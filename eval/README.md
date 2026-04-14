# MedSG-Bench: Multimodal Medical Image Segmentation Reproduction

> **Disclaimer (复现声明)**
> This repository is a **reproduction and analysis** of the official [MedSG-Bench](https://github.com/xxxxx/MedSG-Bench) evaluation pipeline. It is not the original source code. The purpose of this project is to establish a verifiable baseline, conduct rigorous failure analysis, and subsequently explore Domain-Specific Fine-Tuning (PEFT/LoRA) for large vision-language models in medical imaging.

## 🚀 Current Progress: Phase A (Baseline Established)

We have successfully reproduced the official zero-shot evaluation pipeline for **Task 8 (Referring)** using an unmodified evaluation protocol.

* **Task:** Referring Expression Comprehension (Skin Lesion / Tumor Localization)
* **Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (Zero-shot)
* **Environment:** Single NVIDIA RTX 4090 (24GB)

### 📊 Baseline Metrics (Official Protocol)
* **AVE_IOU:** 12.67%
* **IOU@0.7:** 2.16%
* **IOU@0.5:** 6.21%
* **IOU@0.3:** 15.59%

*Note: For full metric breakdown and console logs, see [`results/task8_baseline_summary.md`](results/task8_baseline_summary.md) and [`logs/task8_eval_console.log`](logs/task8_eval_console.log).*

### 🔬 B0: Error Attribution Analysis
Prior to any fine-tuning, a rigorous error attribution analysis was conducted to decouple parsing errors from true model capability deficits. 
**Key Finding:** Lenient parsing (regex extraction) yielded an AVE_IOU of **12.62%**, confirming that the low baseline score is *not* primarily due to formatting constraints, but rather a profound **domain knowledge deficit (Mode Collapse/Grounding Failure)**. Over 32% of predictions defaulted to full-image fallback or extremely large bounding boxes.

*(See [`analysis/failure_analysis_notes.md`](analysis/failure_analysis_notes.md) for the complete B0 report).*

## 🛠️ How to Reproduce
For detailed reproduction steps and environment setup, please refer to [`docs/reproduce_task8.md`](docs/reproduce_task8.md). 
*(Note: Model weights and the 7.1GB dataset are excluded from this repository due to size and licensing constraints).*

---
**Next Steps (Phase B):** Proceeding with dataset formatting and PEFT/LoRA domain adaptation to address the visual grounding deficit identified in Phase A.