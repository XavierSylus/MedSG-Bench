import json
import sys
import re

file_path = sys.argv[1]

with open(file_path, 'r') as f:
    data = json.load(f)

total = len(data)

stats = {
    "pure_bbox": 0, "has_markdown": 0, "has_json": 0, "md_and_json": 0, 
    "fallback_count": 0, "large_box_count": 0
}

orig_ious = []
lenient_ious = []

def flatten_box(box):
    # 第一性原理：无视任何嵌套层级，递归把所有数字抽出来
    flat = []
    def _flatten(b):
        if isinstance(b, (list, tuple)):
            for item in b: _flatten(item)
        elif isinstance(b, (int, float)):
            flat.append(float(b))
    _flatten(box)
    return flat[:4] if len(flat) >= 4 else None

def extract_bbox_lenient(text):
    # 宽松解析：无视外层格式，强制提取最后出现的四个坐标组合
    matches = re.findall(r'\[\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\]', text)
    if matches:
        return [float(x) for x in matches[-1]]
    return None

def calc_iou(box1, box2):
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4: return 0.0
    inter_x1, inter_y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    inter_x2, inter_y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1: return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

for item in data:
    ans = str(item.get('answer', ''))
    
    # 使用安全的展平机制提取原始坐标
    gt = flatten_box(item.get('groundtruth', []))
    pred_orig = flatten_box(item.get('filter_answer', []))

    # 1. 格式统计
    has_md = '```' in ans
    has_js = '{' in ans or 'json' in ans.lower()
    if has_md and has_js: stats["md_and_json"] += 1
    elif has_md: stats["has_markdown"] += 1
    elif has_js: stats["has_json"] += 1
    else: stats["pure_bbox"] += 1

    # 2. 定位兜底统计
    if pred_orig == [0.0, 0.0, 336.0, 336.0]:
        stats["fallback_count"] += 1
    elif pred_orig and len(pred_orig) == 4:
        area = (pred_orig[2] - pred_orig[0]) * (pred_orig[3] - pred_orig[1])
        if area > (336 * 336 * 0.8):
            stats["large_box_count"] += 1

    # 3. 计算 IoU 对比
    orig_ious.append(calc_iou(pred_orig, gt))
    
    pred_lenient = extract_bbox_lenient(ans)
    lenient_ious.append(calc_iou(pred_lenient, gt))

ave_iou_orig = sum(orig_ious) / total * 100
ave_iou_lenient = sum(lenient_ious) / total * 100

print("=== B0: Error Attribution Report ===")
print(f"Total Samples: {total}")
print("\n--- 1. Format Breakdown ---")
print(f"Has Markdown & JSON: {stats['md_and_json']} ({(stats['md_and_json']/total)*100:.1f}%)")
print(f"Pure Bbox (Clean): {stats['pure_bbox']} ({(stats['pure_bbox']/total)*100:.1f}%)")
print("\n--- 2. Grounding Breakdown ---")
print(f"Full Image Fallback [0,0,336,336]: {stats['fallback_count']} ({(stats['fallback_count']/total)*100:.1f}%)")
print(f"Extremely Large BBox (>80% area): {stats['large_box_count']} ({(stats['large_box_count']/total)*100:.1f}%)")
print("\n--- 3. Lenient Parsing Impact ---")
print(f"Official AVE_IOU: {ave_iou_orig:.2f}%  -->  Lenient AVE_IOU: {ave_iou_lenient:.2f}%")
