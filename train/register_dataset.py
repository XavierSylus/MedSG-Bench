import json

info_path = '/root/autodl-tmp/MedSG-Bench/train/dataset_info.json'
with open(info_path, 'r') as f:
    info = json.load(f)

# 注册我们的极简验证集
info['Task8_mini'] = {
    "file_name": "/root/autodl-tmp/MedSG-Bench/train/Task8_mini.json",
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "images": "images"
    }
}

with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)

print("✅ 已成功将 Task8_mini 注册到 dataset_info.json 弹药库！")
