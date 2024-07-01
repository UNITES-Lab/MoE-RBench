# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config:
    baselayer: bool=False
    random_gate: bool=False
    router_topk: int=0
    run_plan1: bool=False
    begin_round: int=1
    task_name: str=None
    test_mode: bool=False
    model_type: str='none'
    model_name: str="ckpts/Llama-2-7b-chat-fp16"
    project_name: str="unkown-project"
    expname: str='expname-unkown'
    scheduler_type: str='step'
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=False
    run_ood: bool=False
    run_bilevel: bool=False
    run_freelb: bool=False
    bilevel_step: int=2
    adv_init: float=2e-2
    adv_lr: float=1e-2
    adv_K: int=3
    prompt_type: int=0
    run_validation_steps: int=10
    freeze_router: bool=False
    freeze_backbone: bool=False
    dropout_rate: float = 0
    expert_dropout_rate: float = 0
    batch_size_training: int=4
    gradient_accumulation_steps: int=1
    num_epochs: int=3
    num_workers_dataloader: int=8
    num_labels: int=3
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "finetuned_models"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = False
    save_every_epoch: bool = False
    dist_checkpoint_root_folder: str="fsdp" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    load_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    aux_loss_weight: float = 0
    router_z_loss_coef: float = 0
    router_aux_loss_coef: float = 0

    
    
    