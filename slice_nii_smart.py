import nibabel as nib
import numpy as np
from PIL import Image

# 1. 路径配置（同时加载影像和医生的标注）
flair_path = "/root/autodl-tmp/test_images/BraTS20_Training_007/BraTS20_Training_007_flair.nii"
seg_path = "/root/autodl-tmp/test_images/BraTS20_Training_007/BraTS20_Training_007_seg.nii"
save_path = "/root/autodl-tmp/test_images/brats_sample.jpg"

print("正在通过医生标注寻找最大的肿瘤切面...")
flair_data = nib.load(flair_path).get_fdata()
seg_data = nib.load(seg_path).get_fdata()

# 2. 统计每一层的肿瘤像素数量，找到包含最大肿瘤的 Z 轴索引
tumor_pixels_per_slice = np.sum(seg_data > 0, axis=(0, 1))
best_z = np.argmax(tumor_pixels_per_slice)
print(f"锁定目标！第 {best_z} 层切片包含最大的脑肿瘤。")

# 3. 提取这一层进行归一化和缩放
slice_2d = flair_data[:, :, best_z]

slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
slice_2d = (slice_2d * 255).astype(np.uint8)

img_pil = Image.fromarray(slice_2d).rotate(90).convert("RGB")
img_pil = img_pil.resize((336, 336), Image.Resampling.LANCZOS)
img_pil.save(save_path)

print(f"已重新生成带有明显肿瘤的图片，覆盖原文件: {save_path}")
