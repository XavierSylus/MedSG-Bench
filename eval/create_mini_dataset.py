import json
import os

source_file = '/root/autodl-tmp/MedSG_Data/MedSG-Train/Task8.json'
out_file = '/root/autodl-tmp/MedSG-Bench/train/Task8_mini.json'
# 你服务器上真正的图像物理父目录
real_base_dir = '/root/autodl-tmp/MedSG_Data/MedSG-Train'

with open(source_file, 'r') as f:
    data = json.load(f)

mini_data = data[:50]
missing_images = 0

for item in mini_data:
    new_images = []
    for img_rel_path in item.get('images', []):
        # 暴力切除原作者的幽灵前缀 "MedSG-Bench/MedSG-Train/"
        clean_rel_path = img_rel_path.replace("MedSG-Bench/MedSG-Train/", "")
        # 组装成坚不可摧的绝对路径
        abs_path = os.path.join(real_base_dir, clean_rel_path)
        
        if os.path.exists(abs_path):
            new_images.append(abs_path)
        else:
            new_images.append(abs_path)
            missing_images += 1
            
    item['images'] = new_images

with open(out_file, 'w') as f:
    json.dump(mini_data, f, indent=2, ensure_ascii=False)

print(f"✅ 成功提取 50 条样本，已生成: {out_file}")
if missing_images > 0:
    print(f"❌ [致命警告] 映射后物理磁盘上仍缺失 {missing_images} 张图像！路径组装失败。")
else:
    print("✅ [状态极佳] 150张图像全部精准命中物理地址！随时可以点火起飞！")
