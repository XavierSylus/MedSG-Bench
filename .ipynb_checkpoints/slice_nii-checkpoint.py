import nibabel as nib
import numpy as np
from PIL import Image
import os

# 这里在路径中加上了中间那层文件夹的名字
nii_file_path = "/root/autodl-tmp/test_images/BraTS20_Training_007/BraTS20_Training_007_flair.nii" 
save_path = "/root/autodl-tmp/test_images/brats_sample.jpg"

print(f"正在读取 3D 医疗数据: {nii_file_path}")
img = nib.load(nii_file_path)
data = img.get_fdata()

z_mid = data.shape[2] // 2
slice_2d = data[:, :, z_mid]

print("正在执行 Min-Max 归一化处理...")
slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
slice_2d = (slice_2d * 255).astype(np.uint8)

img_pil = Image.fromarray(slice_2d).rotate(90).convert("RGB")
img_pil = img_pil.resize((336, 336), Image.Resampling.LANCZOS)
img_pil.save(save_path)

print(f"2D 切片提取成功！已保存为: {save_path}")
