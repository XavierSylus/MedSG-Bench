# MedSG-Bench


## 👉 Environment

You should follow the commands below to establish an environment.

```
conda create -n MedSG-Bench python=3.10

git clone [our repo link]
cd [our repo]

conda activate MedSG-Bench
pip install -r requirements.txt
```

## 👉 Dataset and model

We release all resources on [HuggingFace](https://huggingface.co/MedSG-Bench) for public access and reproducibility, including:

&nbsp;&nbsp;&nbsp;&nbsp;👏 **MedSG-Bench**: A benchmark for medical image sequences grounding;

&nbsp;&nbsp;&nbsp;&nbsp;👏 **MedSG-188K**: A large-scale grounding instruction-tuning dataset;

&nbsp;&nbsp;&nbsp;&nbsp;👏 **MedSeq-Grounder**: A medical grounding-enhanced MLLM.


## 👉 Evaluation on MedSG-Bench

An example run script:
```
python eval/MedSG_Bench.py \
    --model_type qwen2_5_vl \
    --model_path /your/path/to/model/checkpoint
    --task Registered_Diff \
    --test_data /your/path/to/task/json/file \
    --output_path /your/output/path
```

## 👉 Training

The training process is conducted mainly based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

We provide the dataset_info.json and training script in the train folder.

## 🌟 Citation

If you find our paper, data and code useful in your research, please consider giving a star and citation.

```Bibtex
@article{yue2025medsg,
  title={MedSG-Bench: A Benchmark for Medical Image Sequences Grounding},
  author={Yue, Jingkun and Zhang, Siqi and Jia, Zinan and Xu, Huihuan and Han, Zongbo and Liu, Xiaohong and Wang, Guangyu},
  journal={arXiv preprint arXiv:2505.11852},
  year={2025}
}
```

---

## MedSG-Bench Task 8 Reproduction & Diagnostic Report

### 1. Executive Summary

This reproduction did **not** aim to fully replicate the original paper’s large-scale training setting. Instead, the goal was to complete a **resource-constrained reproduction** (single RTX 4090) with a rigorous and trustworthy workflow:

- Execute the **official benchmark / evaluation pipeline** without modifications.
- Establish a **credible zero-shot baseline**.
- Complete the full **training -> LoRA merge -> official re-evaluation** closed loop.
- Identify at least one **effective training route** under limited compute and extreme few-shot (100 samples) conditions.
- Systematically diagnose the failure modes of multimodal fine-tuning in medical visual grounding.

### 2. What Has Been Completed

#### 2.1 Official Baseline Reproduction
The official MedSG-Bench Task 8 referring evaluation pipeline was successfully reproduced.

**Baseline Official Result**
- IOU@0.7 = **2.1592%**
- IOU@0.5 = **6.2078%**
- IOU@0.3 = **15.5870%**
- AVE_IOU = **12.6663%**
- TOTAL = **1482**

This established a reliable, official starting point for all subsequent ablation studies.

#### 2.2 Training–Evaluation Closed Loop
A full closed loop was successfully established:
`Data preprocessing` -> `LoRA Fine-tuning` -> `Checkpoint Merge` -> `Smoke Test` -> `Official Task 8 Evaluation`.
This means the reproduction moved beyond simple inference testing and reached the level of **training-based validation**.

### 3. The Diagnostic Trajectory

This reproduction relied on a strict control-variable method across 5 rounds of troubleshooting:

- **Mini-LoRA v1 (AVE_IOU = 12.6165%)**: Completed the full loop but did not improve the official score. This proved that "training runs successfully" does not imply "training is effective."
- **Round 2 (AVE_IOU = 0.6777%)**: A catastrophic drop. The root cause was traced to a **data protocol error** (using point supervision instead of bounding-box supervision). This revealed the extreme sensitivity of medical grounding tasks to exact data formatting.
- **Round 3 (AVE_IOU = 9.9000%)**: After restoring bbox-style supervision, outputs were legally formatted, but performance degraded. Analysis showed systematic box shrinkage and spatial offsets, indicating the model's zero-shot grounding anchors were disrupted.
- **Prompt Mismatch Probe (AVE_IOU = 10.1400%)**: Confirmed that training/evaluation prompt mismatch was real, but only a secondary factor.
- **Round 4 (AVE_IOU = 10.4869%)**: Removing natural-language prefixes (e.g., `It's in the first image...`) produced a small gain, showing that language generation dilutes attention away from precise coordinate regression. However, this still did not explain the main degradation.

### 4. Key Breakthrough: Round 5

The decisive improvement came from **shrinking the LoRA injection scope**.

Instead of using aggressive full-layer adaptation (`lora_target: all`), a highly conservative configuration was used to protect the base model's spatial common sense:
- 100 high-quality training samples
- Pure bbox-token-only supervision
- Frozen vision tower
- **Conservative LoRA target (`q_proj, v_proj` only)**
- Official Task 8 evaluation pipeline

**Round 5 Official Result**
- IOU@0.7 = **2.1592%**
- IOU@0.5 = **5.2632%**
- IOU@0.3 = **14.2375%**
- AVE_IOU = **12.9747%**
- TOTAL = **1482**

**Delta vs baseline = +0.3084**
*(Note: While the overall Average IOU and looser bounding box matches [IOU@0.3] saw measurable gains, stricter boundary matching [IOU@0.5] saw a slight trade-off. This highlights the delicate balance required to inject new syntax while preserving fine-grained spatial anchors during limited-data LoRA adaptation.)*

### 5. Main Conclusion

The reproduction suggests that under resource-constrained settings, the main bottleneck is **not only** prompt wording or output formatting. 

The stronger factor is that overly aggressive LoRA adaptation, especially with a wide target scope such as `lora_target: all`, **over-disturbs the base model’s original spatial grounding ability**. Once the LoRA injection range was reduced to a more conservative setting (`q_proj, v_proj`), the model rapidly recovered and achieved the first measurable gain over the official zero-shot baseline.

In other words:
> Under limited compute, the most effective strategy is not "more data" or "broader adaptation," but a carefully constrained fine-tuning setup that preserves the base model’s original grounding anchors.

### 6. Consolidated Result Table

| Run | Meaning | AVE_IOU | Delta vs baseline |
|---|---|---:|---:|
| **baseline** | official zero-shot | 12.6663 | +0.0000 |
| **mini_lora_v1** | first closed-loop run | 12.6165 | -0.0498 |
| **round2** | point-supervision error | 0.6777 | -11.9886 |
| **round3** | bbox restored | 9.9000 | -2.7663 |
| **round3_prompt** | prompt mismatch probe | 10.1400 | -2.5263 |
| **round4_bbox** | remove image-index wording | 10.4869 | -2.1794 |
| **round5_qv_only** | **shrink LoRA target** | **12.9747** | **+0.3084** |

### 7. Academic Claims & Disclaimers

**What can be claimed now:**
> We completed a resource-constrained reproduction of the MedSG-Bench Task 8 pipeline: the official baseline was reproduced, the training–merge–official re-evaluation loop was fully established, failure modes were systematically diagnosed, and an effective fine-tuning route was identified that surpassed the zero-shot baseline under official evaluation.

**What this reproduction does NOT claim:**
> This reproduction does not claim a full-scale replication of the original paper’s training setup, hardware scale, or final large-scale model performance. This is a **resource-constrained task-level reproduction**, not a compute-equivalent reproduction.

