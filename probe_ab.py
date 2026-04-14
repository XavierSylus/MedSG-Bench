import json
import os
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore")

model_path = "/root/autodl-tmp/qwen_weights_round3"
data_path = "/root/autodl-tmp/MedSG_Data/MedSG-Bench/Task8.json"
if not os.path.exists(data_path):
    data_path = "/root/autodl-tmp/MedSG_Data/MedSG-Bench/Task8_aligned.json"

print("🚀 加载 Round3 模型中...")
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = [x for x in data if x.get("task") == "referring"][:5]

for idx, item in enumerate(samples):
    print(f"\n{'-'*80}")
    print(f"[Sample {idx+1}/5] ID: {item.get('id', 'N/A')}")
    print(f"🎯 GroundTruth: {item.get('answer', item.get('groundtruth', 'N/A'))}")

    images = []
    for img_path in item.get("images", []):
        # 暴力寻址防御：遍历所有可能的挂载点
        possible_paths = [
            os.path.join("/root/autodl-tmp/MedSG_Data", img_path),
            os.path.join("/root/autodl-tmp/MedSG_Data/MedSG-Bench", img_path),
            os.path.join("/root/autodl-tmp", img_path),
            os.path.join("/root/autodl-tmp/MedSG-Bench", img_path)
        ]
        full_path = next((p for p in possible_paths if os.path.exists(p)), None)
        if not full_path:
            print(f"❌ 警告: 找不到图片 {img_path}")
            continue
        images.append(full_path)

    base_question = item.get("question", "")

    # A 组：官方 Eval 考卷格式 (强硬指令 + ref 标签替换)
    q_A = base_question.replace('<|object_ref_start|>', '<ref>').replace('<|object_ref_end|>', '</ref>')
    q_A += " Output only the coordinates. Format: (x1,y1),(x2,y2). Strictly follow this format. No additional text or explanation."
    
    # B 组：训练方言格式 (保留原生标签，无附加限制)
    q_B = base_question

    def infer(question):
        messages = [
            {"role": "user", "content": [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": question}]}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        if image_inputs is None:
            inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
        else:
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    res_A = infer(q_A)
    print(f"👉 [A 组 - 官方 Eval 考卷]:\n{res_A}")
    res_B = infer(q_B)
    print(f"👉 [B 组 - Round3 训练方言]:\n{res_B}")
print('-'*80)
