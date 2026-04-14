import json
import os
import re

source_file = '/root/autodl-tmp/MedSG_Data/MedSG-Train/Task8.json'
out_file = '/root/autodl-tmp/MedSG-Bench/train/Task8_round2_100.json'
real_base_dir = '/root/autodl-tmp/MedSG_Data/MedSG-Train'

with open(source_file, 'r') as f:
    data = json.load(f)

clean_data = []
for item in data:
    if len(clean_data) >= 100:
        break
        
    # 修复幽灵路径
    new_images = []
    for img_rel_path in item.get('images', []):
        clean_rel_path = img_rel_path.replace("MedSG-Bench/MedSG-Train/", "")
        new_images.append(os.path.join(real_base_dir, clean_rel_path))
    item['images'] = new_images

    # 核心校准：暴力切断任何解释性文本，只保留坐标框
    is_valid = True
    for conv in item['conversations']:
        if conv['from'] == 'gpt':
            # 寻找类似 [x1, y1, x2, y2] 或 (x1, y1), (x2, y2) 的纯坐标
            match = re.search(r'(\[.*?\]|\(.*?\))', conv['value'])
            if match:
                conv['value'] = match.group(0) # 强制覆盖，剥夺模型学习说话的机会
            else:
                is_valid = False # 连坐标都没有的数据直接丢弃
    
    if is_valid:
        clean_data.append(item)

with open(out_file, 'w') as f:
    json.dump(clean_data, f, indent=2, ensure_ascii=False)

# 自动注册到 LLaMA-Factory
info_path = '/root/autodl-tmp/MedSG-Bench/train/dataset_info.json'
with open(info_path, 'r') as f:
    info = json.load(f)
info['Task8_round2'] = {
    "file_name": out_file,
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "images"}
}
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)

print("✅ 第二轮校准训练集 (100条) 已生成并注册！输出纯净度已锁定至极限。")
