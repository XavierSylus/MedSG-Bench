import json
import os

source_file = '/root/autodl-tmp/MedSG_Data/MedSG-Train/Task8.json'
out_file = '/root/autodl-tmp/MedSG-Bench/train/Task8_round3_100.json'
real_base_dir = '/root/autodl-tmp/MedSG_Data/MedSG-Train'

with open(source_file, 'r') as f:
    data = json.load(f)

clean_data = data[:100] # 直接截取前 100 条，不修改 value
for item in clean_data:
    new_images = []
    for img_rel_path in item.get('images', []):
        clean_rel_path = img_rel_path.replace("MedSG-Bench/MedSG-Train/", "")
        new_images.append(os.path.join(real_base_dir, clean_rel_path))
    item['images'] = new_images

with open(out_file, 'w') as f:
    json.dump(clean_data, f, indent=2, ensure_ascii=False)

# 注册到 LLaMA-Factory
info_path = '/root/autodl-tmp/MedSG-Bench/train/dataset_info.json'
with open(info_path, 'r') as f:
    info = json.load(f)
info['Task8_round3'] = {
    "file_name": out_file,
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "images"}
}
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)
print("✅ Round 3 健康数据集已生成！原汁原味的官方坐标格式已保留。")
