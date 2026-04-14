import json
import sys

file_path = sys.argv[1]

with open(file_path, 'r') as f:
    data = json.load(f)

total = len(data)
fallback_count = 0
format_error_count = 0

for item in data:
    # 统计全图兜底 (Mode Collapse)
    pred = item.get('filter_answer', [])
    if pred == [0.0, 0.0, 336.0, 336.0] or pred == [0, 0, 336, 336]:
        fallback_count += 1
        
    # 统计格式幻觉 (Format Hallucination)
    ans = item.get('answer', '')
    if '```' in ans or '{' in ans or '[' in ans:
        format_error_count += 1

print(f"=== Baseline Failure Analysis ===")
print(f"Total Samples: {total}")
print(f"Fallback [0,0,336,336] Count: {fallback_count} ({fallback_count/total*100:.2f}%)")
print(f"Format Error (JSON/Markdown) Count: {format_error_count} ({format_error_count/total*100:.2f}%)")
