# MedSG-Bench Zero-Shot Evaluation: Exposing Spatial Hallucinations in Vision-Language Models

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Overview
This repository contains an independent, rigorous zero-shot evaluation pipeline for **MedSG-Bench** (A Benchmark for Medical Image Sequences Grounding, NeurIPS 2025). 

Specifically, this project reproduces **Task 8: Referring Grounding** on a highly complex subset of brain MRI scans (BraTS 2020) using the state-of-the-art Multimodal Large Language Model (MLLM), **Qwen2.5-VL-7B**. The primary objective is to quantitatively and qualitatively investigate the boundary of generic MLLMs' grounding capabilities in specialized, low-contrast clinical contexts.

## 🎯 Motivation & Core Findings
While modern MLLMs exhibit remarkable performance on natural images, their zero-shot spatial reasoning in medical imaging remains underexplored. 

Through a controlled subset reproduction (N=55), this project successfully verified the critical failure modes documented in the original MedSG-Bench study. The empirical results demonstrate that without specialized medical instruction tuning, generic MLLMs suffer from severe **"Spatial Bias"** and **"Mode Collapse."**

### 📊 Quantitative Results
Under strict zero-shot evaluation protocols using the official prompt templates, the model exhibited near-complete failure in cross-modal localization:
* **Total Valid Samples:** 55
* **Average Intersection over Union (IoU):** 0.21%
* **Accuracy at IoU 0.5 (Acc@0.5):** 0.00%

### 👁️ Qualitative Analysis (Failure Case)
The model consistently failed to localize the hyperintense lesions based on visual cues. Instead, it defaulted to generating bounding boxes anchored to the top-left corner of the image space (approximate coordinates `[20, 10, 80, 90]`), regardless of the actual tumor location. This confirms the hypothesis that the model relies on spatial hallucination rather than semantic visual grounding when facing complex medical contexts.

*(Please refer to `comparison_001.jpg` in this repository for a visual demonstration of the ground truth vs. the hallucinated bounding box).*

## ⚙️ Methodology
This evaluation pipeline strictly adheres to the data preprocessing and prompting guidelines outlined in the MedSG-Bench protocol:
1. **Data Preprocessing:** 3D `.nii` volumes were parsed to extract the 2D slice containing the maximum tumor area. Slices underwent strict Min-Max normalization and were resized to `336x336` to align with the model's visual encoder constraints.
2. **Standardized Prompting:** Employed the official referring grounding template without any forced jailbreak instructions: `Identify the bounding box of the region described by the following expression: <|object_ref_start|> brain tumor (abnormal hyperintense lesion) <|object_ref_end|>.`
3. **Dual-Metric Evaluation:** Automated calculation of both Average IoU and Acc@0.5 based on exact bounding box overlap with physician-annotated ground truth masks.

## 🚀 Repository Structure
* `batch_pipeline.py`: Automated batch inference script for data ingestion, preprocessing, and zero-shot VLM querying.
* `calculate_iou.py`: Evaluation script for parsing model JSON outputs and computing IoU/Acc@0.5 against `.nii` ground truth masks.
* `benchmark_results_full.json`: The raw output logs containing the parsed coordinates from the 55 evaluation samples.
* `comparison_001.jpg`: Visualized failure case demonstrating the mode collapse.

## 💻 Quick Start
**Note:** Due to data privacy and file size constraints, the original BraTS 2020 dataset and Qwen2.5-VL model weights are **not** included in this repository.

1. Clone the repository:
   ```bash
   git clone [https://github.com/XavierSylus/MedSG-Bench.git](https://github.com/XavierSylus/MedSG-Bench.git)
   cd MedSG-Bench
2. Download the BraTS 2020 Dataset and place the extracted folders in /BraTS2020/.

3. Download the Qwen2.5-VL-7B weights into /qwen_weights/.

4. Run the evaluation pipeline:
    ```Bash
    python batch_pipeline.py
    python calculate_iou.py
    
📚 References
Yue, J., et al. "MedSG-Bench: A Benchmark for Medical Image Sequences Grounding." NeurIPS, 2025.

Qwen Team. "Qwen2.5-VL Technical Report." arXiv, 2025.

Menze, B. H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." IEEE Transactions on Medical Imaging, 2014.
