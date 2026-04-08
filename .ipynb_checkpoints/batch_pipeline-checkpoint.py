import os
import json
import torch
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

dataset_root = "/root/autodl-tmp/BraTS2020"
model_dir = "/root/autodl-tmp/qwen_weights"
output_json = "/root/autodl-tmp/benchmark_results_full.json"

print("初始化 Qwen2.5-VL 大模型 (学术复现模式)...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)

patient_dirs = []
for root, dirs, files in os.walk(dataset_root):
    for d in dirs:
        if d.startswith("BraTS20_Training_"):
            patient_dirs.append(os.path.join(root, d))

patient_dirs.sort()
test_patient_dirs = patient_dirs

results = []
print(f"开始全量学术评估，共计 {len(test_patient_dirs)} 个样本，请保持后台运行...")

for patient_dir in tqdm(test_patient_dirs):
    patient_id = os.path.basename(patient_dir)
    flair_path = os.path.join(patient_dir, f"{patient_id}_flair.nii")
    seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii")

    if not os.path.exists(flair_path) or not os.path.exists(seg_path):
        continue

    flair_data = nib.load(flair_path).get_fdata()
    seg_data = nib.load(seg_path).get_fdata()

    tumor_pixels = np.sum(seg_data > 0, axis=(0, 1))
    best_z = np.argmax(tumor_pixels)

    if tumor_pixels[best_z] == 0:
        continue

    slice_2d = flair_data[:, :, best_z]
    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
    slice_2d = (slice_2d * 255).astype(np.uint8)

    img_pil = Image.fromarray(slice_2d).rotate(90).convert("RGB")
    img_pil = img_pil.resize((336, 336), Image.Resampling.LANCZOS)

    temp_img_path = f"/root/autodl-tmp/temp_{patient_id}.jpg"
    img_pil.save(temp_img_path)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": temp_img_path},
            {"type": "text", "text": "Identify the bounding box of the region described by the following expression: <|object_ref_start|> brain tumor (abnormal hyperintense lesion) <|object_ref_end|>."}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    results.append({
        "patient_id": patient_id,
        "slice_z_index": int(best_z),
        "qwen_prediction": output_text.strip()
    })

    os.remove(temp_img_path)

    if len(results) % 50 == 0:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)

with open(output_json, "w") as f:
    json.dump(results, f, indent=4)

print(f"\n全量跑通！报告已保存至: {output_json}")