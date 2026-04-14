import json, re, glob, statistics as st
from collections import Counter

# 自动寻找最新的 Prompt 消融评测结果
files = glob.glob("/root/autodl-tmp/Task8_eval_result_round3_prompt_ablation.json/*.json")
if not files:
    raise FileNotFoundError("找不到评测结果文件")
ROUND3_JSON = files[0]
print(f"📊 正在解剖文件: {ROUND3_JSON}\n")

with open(ROUND3_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = [x for x in data if isinstance(x, dict) and x.get("task") == "referring"]

ious = []
areas_pred = []
areas_gt = []
ratios = []
cx_offsets = []
cy_offsets = []
w_ratios = []
h_ratios = []
zero_like = 0
small_pred = 0

def box_area(b):
    try:
        x1,y1,x2,y2 = b
        return max(0, float(x2)-float(x1)) * max(0, float(y2)-float(y1))
    except:
        return 0

def get_safe_pred(pred_raw):
    if isinstance(pred_raw, list):
        if len(pred_raw) > 0 and isinstance(pred_raw[0], list):
            return pred_raw[0]
        elif len(pred_raw) == 4:
            return pred_raw
    return [0.0, 0.0, 0.0, 0.0]

for x in samples:
    # 兼容脚本的两种字段命名
    pred = get_safe_pred(x.get("prediction", x.get("filter_answer", [0,0,0,0])))
    gt = x.get("groundtruth", [0,0,0,0])
    
    px1,py1,px2,py2 = pred
    gx1,gy1,gx2,gy2 = gt

    p_area = box_area(pred)
    g_area = box_area(gt)

    # 动态计算 IoU，因为原始脚本可能没把 iou 存进 list
    x_left = max(gx1, px1)
    y_top = max(gy1, py1)
    x_right = min(gx2, px2)
    y_bottom = min(gy2, py2)
    if x_right < x_left or y_bottom < y_top:
        iou = 0.0
    else:
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = g_area + p_area - intersection
        iou = intersection / union if union > 0 else 0.0

    ious.append(iou)
    areas_pred.append(p_area)
    areas_gt.append(g_area)

    if p_area == 0:
        zero_like += 1
        continue

    if p_area < 400:
        small_pred += 1

    if g_area > 0:
        ratios.append(p_area / g_area)

    pcx, pcy = (px1+px2)/2, (py1+py2)/2
    gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
    cx_offsets.append(pcx-gcx)
    cy_offsets.append(pcy-gcy)

    gw, gh = max(1, gx2-gx1), max(1, gy2-gy1)
    pw, ph = max(1, px2-px1), max(1, py2-py1)
    w_ratios.append(pw/gw)
    h_ratios.append(ph/gh)

print("TOTAL =", len(samples))
print("mean_iou =", sum(ious)/len(ious))
print("median_iou =", st.median(ious))
print("zero_area_pred =", zero_like)
print("small_pred_area(<400) =", small_pred)

if ratios:
    print("pred/gt area ratio mean =", sum(ratios)/len(ratios))
    print("pred/gt area ratio median =", st.median(ratios))

if cx_offsets:
    print("center offset x mean =", sum(cx_offsets)/len(cx_offsets))
    print("center offset y mean =", sum(cy_offsets)/len(cy_offsets))
    print("center offset x median =", st.median(cx_offsets))
    print("center offset y median =", st.median(cy_offsets))

if w_ratios:
    print("width ratio mean =", sum(w_ratios)/len(w_ratios))
    print("height ratio mean =", sum(h_ratios)/len(h_ratios))
    print("width ratio median =", st.median(w_ratios))
    print("height ratio median =", st.median(h_ratios))

# 额外看前5个低IoU样本（为了终端清爽，展示前5个代表性样本）
worst = sorted(samples, key=lambda x: x.get("iou", 0.0))[:5]
print("\n=== Worst 5 samples ===")
for x in worst:
    print({
        "question": x.get("question", "")[:60] + "...",
        "gt": x.get("groundtruth"),
        "pred": get_safe_pred(x.get("prediction", x.get("filter_answer")))
    })
