import json
import os
import re
import numpy as np
import nibabel as nib

dataset_root = "/root/autodl-tmp/BraTS2020"
json_path = "/root/autodl-tmp/benchmark_results_full.json"

patient_dir_map = {}
for root, dirs, files in os.walk(dataset_root):
    for d in dirs:
        if d.startswith("BraTS20_Training_"):
            patient_dir_map[d] = os.path.join(root, d)

def calculate_iou(boxA, boxB):
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0, yB - yA) * max(0, xB - xA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

with open(json_path, "r") as f:
    results = json.load(f)

total_iou = 0.0
acc_05_count = 0
valid_samples = 0

for res in results:
    patient = res["patient_id"]
    z_idx = res["slice_z_index"]
    qwen_pred_str = res["qwen_prediction"]

    match = re.search(r'\[\s*\{.*?\}\s*\]', qwen_pred_str, re.DOTALL)
    if not match:
        continue
    try:
        pred_data = json.loads(match.group(0))
        qwen_box = pred_data[0]["bbox_2d"]
        qwen_box = [int(v / 1000.0 * 336) for v in qwen_box] 
    except Exception:
        continue

    if patient not in patient_dir_map:
        continue
        
    seg_path = os.path.join(patient_dir_map[patient], f"{patient}_seg.nii")
    if not os.path.exists(seg_path):
        continue
        
    seg_data = nib.load(seg_path).get_fdata()
    slice_mask = seg_data[:, :, z_idx] > 0
    slice_mask = np.rot90(slice_mask)
    
    rows = np.any(slice_mask, axis=1)
    cols = np.any(slice_mask, axis=0)
    if not np.any(rows):
        continue
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    scale = 336.0 / 240.0
    true_box = [int(ymin * scale), int(xmin * scale), int(ymax * scale), int(xmax * scale)]

    iou = calculate_iou(qwen_box, true_box)
    total_iou += iou
    
    if iou >= 0.5:
        acc_05_count += 1
        
    valid_samples += 1

if valid_samples > 0:
    avg_iou = total_iou / valid_samples
    acc_05 = acc_05_count / valid_samples
    print("=" * 60)
    print(" 严谨学术复现评估结果 (MedSG-Bench Task 8: BraTS)")
    print(f"有效测试样本数: {valid_samples}")
    print(f"Average IoU  : {avg_iou * 100:.2f}%")
    print(f"Acc@0.5      : {acc_05 * 100:.2f}%")
    print("=" * 60)
else:
    print("未能成功计算任何样本，请检查。")