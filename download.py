from modelscope import snapshot_download
import os

save_dir = '/root/autodl-tmp/qwen_weights'
os.makedirs(save_dir, exist_ok=True)

print("开始从魔搭社区极速下载 Qwen2.5-VL-7B 模型...")
model_dir = snapshot_download('qwen/Qwen2.5-VL-7B-Instruct', local_dir=save_dir)
print(f"模型已成功下载至: {model_dir}")
