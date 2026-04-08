from PIL import Image, ImageDraw

image_path = "/root/autodl-tmp/test_images/brats_sample.jpg"
save_path = "/root/autodl-tmp/test_images/result_with_box.jpg"

print("正在将大模型的预测坐标绘制到切片上...")
img = Image.open(image_path)
draw = ImageDraw.Draw(img)

# Qwen-VL 的坐标格式默认为 [ymin, xmin, ymax, xmax]，且归一化到 1000
bbox_qwen = [70, 38, 265, 256]
width, height = img.size  # 论文要求的 336x336

ymin, xmin, ymax, xmax = bbox_qwen

# 转换回真实的像素坐标
abs_xmin = int(xmin / 1000 * width)
abs_ymin = int(ymin / 1000 * height)
abs_xmax = int(xmax / 1000 * width)
abs_ymax = int(ymax / 1000 * height)

# 绘制红色的预测边界框
draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline="red", width=3)

img.save(save_path)
print(f"预测框绘制完成！请在左侧查看文件: {save_path}")
