import json
import os

mini_result_path = '/root/autodl-tmp/Task8_eval_result_mini.json'

# 自动处理“路径变目录”的坑
if os.path.isdir(mini_result_path):
    json_files = [f for f in os.listdir(mini_result_path) if f.endswith('.json')]
    if not json_files:
        print(f"❌ 目录 {mini_result_path} 下没有找到 json 文件！")
        exit(1)
    # 自动读取里面的真实结果文件
    mini_result_path = os.path.join(mini_result_path, json_files[0])
    print(f"📂 穿透目录，自动锁定真实评测文件: {mini_result_path}\n")

with open(mini_result_path, 'r') as f:
    results = json.load(f)

total = len(results)
multi_box_count = 0
add_criterion_count = 0
json_markdown_count = 0
pure_format_count = 0

for item in results:
    # 兼容有些脚本把预测结果包在不同字段的可能
    pred = str(item.get('prediction', item.get('pred', '')))
    
    if 'addCriterion' in pred:
        add_criterion_count += 1
        
    if pred.count('(') > 2 or pred.count('[') > 2:
        multi_box_count += 1
        
    if '```' in pred or '{' in pred:
        json_markdown_count += 1
        
    if 'addCriterion' not in pred and '{' not in pred and pred.count('(') <= 2 and pred.count('[') <= 2:
        pure_format_count += 1

print("============================================================")
print(f"🎯 失败模式诊断报告 (总样本: {total})")
print(f"-> 包含 'addCriterion' 文本污染: {add_criterion_count} 例 ({add_criterion_count/total:.1%})")
print(f"-> 包含多框/冗余括号输出: {multi_box_count} 例 ({multi_box_count/total:.1%})")
print(f"-> 包含 JSON/Markdown 结构: {json_markdown_count} 例 ({json_markdown_count/total:.1%})")
print(f"-> 相对纯净的单框输出: {pure_format_count} 例 ({pure_format_count/total:.1%})")
print("============================================================")
