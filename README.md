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
