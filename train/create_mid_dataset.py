import json
import os

source_file = '/root/autodl-tmp/MedSG_Data/MedSG-Train/Task8.json'
out_file = '/root/autodl-tmp/MedSG-Bench/train/Task8_500.json'
real_base_dir = '/root/autodl-tmp/MedSG_Data/MedSG-Train'

with open(source_file, 'r') as f:
    data = json.load(f)

# 梯度扩容：抽取 500 条
mid_data = data[:500]

for item in mid_data:
    new_images = []
    for img_rel_path in item.get('images', []):
        clean_rel_path = img_rel_path.replace("MedSG-Bench/MedSG-Train/", "")
        new_images.append(os.path.join(real_base_dir, clean_rel_path))
    item['images'] = new_images

with open(out_file, 'w') as f:
    json.dump(mid_data, f, indent=2, ensure_ascii=False)

# 自动注册到 LLaMA-Factory 注册表
info_path = '/root/autodl-tmp/MedSG-Bench/train/dataset_info.json'
with open(info_path, 'r') as f:
    info = json.load(f)
info['Task8_500'] = {
    "file_name": out_file,
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "images"}
}
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)

print("✅ 500 条梯度扩容数据集 Task8_500 已生成并注册完毕！")
