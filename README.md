# $\texttt{MoE-RBench}$: Towards Building Reliable Language Models with Sparse Mixture-of-Experts

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for "MoE-RBench: Towards Building Reliable Language Models with Sparse Mixture-of-Experts". Our implementation is based on [LLMs-Finetuning-Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/tree/main/llama2) and [llama-recipes](https://github.com/meta-llama/llama-recipes)

- Authors: [Guanjie Chen](https://guanjiechen118.github.io/), [Xinyu Zhao](https://zhaocinyu.github.io/), [Tianlong Chen](https://tianlong-chen.github.io/), and [Yu Cheng](https://www.linkedin.com/in/chengyu05/)
- Paper: [OpenReview](https://openreview.net/forum?id=LyJ85kgHFe)
## **Overview**
Mixture-of-Experts (MoE) has gained increasing popularity as a promising framework for scaling up large language models (LLMs). However, the reliability assessment of MoE lags behind its surging applications. Moreover, when transferred to new domains such as in fine-tuning MoEmodelssometimes underperform their dense counterparts. Motivated by the research gap and counter-intuitive phenomenon, we propose MoE-RBench, the first comprehensive assessment of SMoE reliability from three aspects: (i) safety and hallucination, (ii) resilience to adversarial attacks, and (iii) out-of-distribution robustness. Extensive models and datasets are tested to compare the MoE to dense networks from these reliability dimensions. Our empirical observations suggest that with appropriate hyperparameters, training recipes, and inference techniques, we can build the MoE model more reliably than the dense LLM. In particular, we find that the robustness of SMoE is sensitive to the basic training settings. We hope that this study can provide deeper insights into how to adapt the pre-trained MoEmodel to other tasks with higher-generation security, quality, and stability.

## **Setup**
```
# Cuda 11.8 is strongly recommended
conda create -n moe_rbench python=3.9 -y && conda activate moe_rbench
git clone https://github.com/UNITES-Lab/MoE-RBench && cd MoE-RBench
pip install -r requirements
wandb login 
```

## **Datasets**

| Tasks    | Datasets |
| -------- | ------- |
| Safety  | [MaliciousInstructions](https://github.com/vinid/safety-tuned-llamas/tree/main/data/evaluation), [CoNa](https://github.com/vinid/safety-tuned-llamas/tree/main/data/evaluation), [Controversial](https://github.com/vinid/safety-tuned-llamas/tree/main/data/evaluation), [Alpaca](https://github.com/vinid/safety-tuned-llamas/tree/main/data/evaluation)  |
| Hallucination | [TruthfulQA](https://github.com/sylinrl/TruthfulQA), [NQ](ai.google.com/research/NaturalQuestions) |
| OOD Robustness    | [Style-OOD](https://huggingface.co/datasets/AI-Secure/DecodingTrust/viewer/ood), [BOSS](https://github.com/lifan-yuan/OOD_NLP)    |
| Adv Robustness    | [SNLI](https://huggingface.co/datasets/stanfordnlp/snli), [SNLI-hard](https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl), [ANLI](https://huggingface.co/datasets/facebook/anli)   |
1. All datasets except BOSS and SNLI-hard can be downloaded and processed automatically.
2. To evaluate models with BOSS, you should follow the pipeline in the [BOSS-repo])(https://github.com/lifan-yuan/OOD_NLP) and place the processed data in the appropriate folder in `MoE-RBench/robustness-evaluation/ft-datasets`.
3. To evaluate models with SNLI-hard, download raw data from [here](https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl), and follow the instructions of NLI task in [BOSS](https://github.com/lifan-yuan/OOD_NLP) to process the dataset, and palce it under `MoE-RBench/robustness-evaluation/ft_datasets/snli_dataset/data`.

## **Robustness Evaluation**

1. Scripts for robustness evaluation on all models and tasks are listed in `MoE_RBench/robustness-evaluation/scripts`.
2. One example is given as follow:
```shell
torchrun --nnodes 1 --nproc_per_node gpu_num --master-port xxx \
finetuning_ood_style.py \ 
--freeze_router \ # for MoE models only, if set, routers will be frozen
--model_type switch \ # (switch,t5,pythia,molm,open_llama,llama_moe)
--project_name wandb_project --expname wandb_exp \
--run_validation --run_ood \ # for ood evaluation only
--batch_size_training 64 --val_batch_size 128 --weight_decay 0 \
--dropout_rate 1e-1 --expert_dropout_rate 2e-1 --lr 5e-5 \
--num_epochs 2 --enable_fsdp --pure_bf16 \ # or pure_fp16
--dataset dataset \ # see details in scripts
--model_name PATH/TO/MODEL \
--dist_checkpoint_root_folder PATH/TO/DIR \
--dist_checkpoint_folder NAME_OF_CKPT \

```
3. The checkpoints shared by fsdp can be transferred to a whole checkpoint by running `convert_model.sh`.

## **Safety Evaluation**
1. Scripts for instruction fine-tuning of all models are listed in `MoE_RBench/safety-evaluation/scripts`. A example is given as follow:
```python
torchrun --nnodes 1 --nproc_per_node gpu_num --master-port xxx finetuning.py \
--batch_size_training 64 --lr 2e-5 \
--gradient_accumulation_steps 1 --weight_decay 0 \
--num_epochs 1 \
--dataset alpaca_dataset \
--enable_fsdp \
--report \
--dist_checkpoint_root_folder PATH/TO/DIR \
--model_name PATH/TO/MODEL --pure_bf16 \
--dist_checkpoint_folder NAME_OF_CKPT
```
2. The checkpoints shared by fsdp can be transferred to a whole checkpoint by running `convert_model.sh`.

3. For evaluation, please refer to [safety-tuned-llamas](https://github.com/vinid/safety-tuned-llamas).

## **Truthfulness Evaluation**
Please refer to [DoLa](https://github.com/voidism/DoLa).

## Citation
```bibtex
@inproceedings{
chen2024moerbench,
title={MoE-RBench: Towards Building Reliable Language Models with Sparse Mixture-of-Experts},
author={Guanjie Chen and Xinyu Zhao and Tianlong Chen and Yu Cheng},
booktitle={The Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=LyJ85kgHFe}
}
```
