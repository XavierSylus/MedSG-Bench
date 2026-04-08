import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_dir = "/root/autodl-tmp/qwen_weights"
image_path = "/root/autodl-tmp/test_images/brats_sample.jpg"

print("加载大模型中...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_dir)

# 【核心修改】：通过强指令绕过安全过滤，强迫输出坐标
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Locate the brain tumor (abnormal hyperintense lesion) in this MRI slice. You MUST output the bounding box coordinates. Do NOT provide medical advice, descriptions, or conversational text. Output ONLY the coordinates."}
        ]
    }
]

print("模型正在强制进行病灶定位...")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n" + "="*60)
print(f"🎯 模型的强制定位结果: \n{output_text[0]}")
print("="*60)
