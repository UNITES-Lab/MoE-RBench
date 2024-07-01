# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os

import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification, AutoConfig, LlamaForSequenceClassification
from moduleformer import ModuleFormerForCausalLM, ModuleFormerConfig, ModuleFormerForSequenceClassification
from t5 import T5Block
from moduleformer.modeling_moduleformer import ModuleFormerConfig, ModuleFormerBlock
from gpt_neox import GPTNeoXLayer, GPTNeoXForSequenceClassification
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from t5 import T5ForSequenceClassification, T5ForConditionalGeneration
from switch_transformers import SwitchTransformersForSequenceClassification, SwitchTransformersForConditionalGeneration
from llama_moe import LlamaMoEConfig, LlamaMoEForSequenceClassification, LlamaMoEDecoderLayer

import policies
from configs import fsdp_config, train_config
from policies import AnyPrecisionAdamW

from utils import fsdp_auto_wrap_policy
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset
from utils.train_utils import (
    train,
    test,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies, 
    get_molm_policies,
    get_switch_policies,
    get_t5_policies,
    get_pythia_policies,
    get_llama_moe_policies
)

import pdb, copy

def main(**kwargs):
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)
    if train_config.model_type == 'molm':
        AutoConfig.register("moduleformer", ModuleFormerConfig)
        AutoModelForCausalLM.register(ModuleFormerConfig, ModuleFormerForCausalLM)
        AutoModelForSequenceClassification.register(ModuleFormerConfig, ModuleFormerForSequenceClassification)
    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        """
        for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        this avoids cpu oom when loading large models like llama 70B, in which case
        model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        overhead and currently requires latest nightly.
        """
        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
            )
            
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else:
        if train_config.model_type == 'molm':
            config = AutoConfig.from_pretrained(train_config.model_name)
            if train_config.router_topk > 0:
                config.k_att = min(train_config.router_topk, 16)
                config.k_mlp = train_config.router_topk
            config.random_gate = train_config.random_gate
            config.num_labels = train_config.num_labels
            config.dropout_rate = train_config.dropout_rate
            config.pad_token_id = config.eos_token_id
            config.expert_dropout_rate = train_config.expert_dropout_rate
            model = AutoModelForSequenceClassification.from_pretrained(
                train_config.model_name,
                # load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                config=config
            )
        elif train_config.model_type == 'pythia':
            config = AutoConfig.from_pretrained(train_config.model_name)
            config.num_labels = 3
            config.dropout_rate = train_config.dropout_rate
            config.pad_token_id = config.eos_token_id
            config.aux_loss_weight = train_config.aux_loss_weight
            model = GPTNeoXForSequenceClassification.from_pretrained(
                train_config.model_name,
                # load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                config = config
            )
        elif train_config.model_type == 't5':
            config = AutoConfig.from_pretrained(train_config.model_name)
            config.num_labels = 3
            config.dropout_rate = train_config.dropout_rate
        
            config.decoder_start_token_id = config.pad_token_id
            model = T5ForConditionalGeneration.from_pretrained(
                train_config.model_name,
                device_map="auto" if train_config.quantization else None,
                config = config
            )
        elif train_config.model_type == 'switch':
            config = AutoConfig.from_pretrained(train_config.model_name)
            config.num_labels = 3
            config.dropout_rate = train_config.dropout_rate
        
            config.router_z_loss_coef = train_config.router_z_loss_coef
            config.router_aux_loss_coef = train_config.router_aux_loss_coef
            config.decoder_start_token_id = config.pad_token_id
            config.expert_dropout_rate = train_config.expert_dropout_rate
            config.classifier_dropout = train_config.dropout_rate
            model = SwitchTransformersForConditionalGeneration.from_pretrained(
                train_config.model_name,
                device_map="auto" if train_config.quantization else None,
                config = config
            )
        elif train_config.model_type == 'llama_moe':
            config = LlamaMoEConfig.from_pretrained(train_config.model_name)
            config.num_labels = train_config.num_labels
            config.expert_dropout_rate = train_config.expert_dropout_rate
            config.pad_token_id = config.eos_token_id
            config.gate_balance_loss_weight = train_config.router_aux_loss_coef
            model = LlamaMoEForSequenceClassification.from_pretrained(
                train_config.model_name,
                device_map="auto" if train_config.quantization else None,
                config = config
            )
        elif train_config.model_type == 'open_llama':
            config = LlamaConfig.from_pretrained(train_config.model_name)
            config.num_labels = train_config.num_labels
            config.pad_token_id = config.eos_token_id
            config.use_cache = use_cache
            model = LlamaForSequenceClassification.from_pretrained(
                train_config.model_name,
                device_map="auto" if train_config.quantization else None,
                config = config,
            )
    if rank == 0:
        import wandb
        wandb.init(project=train_config.project_name, name=train_config.expname)
        train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
        fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
        wandb.config.update(train_config_dict)
        wandb.config.update(fsdp_config_dict)
    init_gate = {}
    for n, p in model.named_parameters():
        if 'gate' in n:
            init_gate[n] = copy.deepcopy(p) 
            
    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)
    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, model_max_length=512,)
    
    if train_config.model_type in ['molm', 'pythia', 'llama_moe']:
        model_max_length = 512
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, model_max_length=model_max_length)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    if train_config.model_type in ['open_llama']:
        model_max_length = 512
        tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name, model_max_length=model_max_length)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        if train_config.model_type == 'molm':
            mixed_precision_policy, wrapping_policy = get_molm_policies(fsdp_config, rank)
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, ModuleFormerBlock)
        elif train_config.model_type == 'pythia':
            mixed_precision_policy, wrapping_policy = get_pythia_policies(fsdp_config, rank)
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, GPTNeoXLayer)
        elif train_config.model_type == 't5':
            mixed_precision_policy, wrapping_policy = get_t5_policies(fsdp_config, rank)
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, T5Block)
        elif train_config.model_type == 'switch':
            mixed_precision_policy, wrapping_policy = get_switch_policies(fsdp_config, rank)
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, SwitchTransformersBlock)
        elif train_config.model_type == 'llama_moe':
            mixed_precision_policy, wrapping_policy = get_llama_moe_policies(fsdp_config, rank)
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaMoEDecoderLayer)
        elif train_config.model_type == 'open_llama':
            mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
        else:
            raise ValueError(f'wrong model type: {train_config.model_type}')
        
        # my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            # TODO: set use_origin_params will get the original param, which can be identified and set to no_grad after FSDP()
            use_orig_params=True,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )

        if train_config.freeze_router:
            for name, param in model.named_parameters():
                if "router" in name or "gate" in name:
                    param.requires_grad = False
                    if rank == 0:
                        print(f'Freeze router: {name}')
                else:
                    param.requires_grad = True
                    
        if train_config.freeze_backbone:
            for name, param in model.named_parameters():
                if "router" in name or "gate" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    if rank == 0:
                        print(f'Freeze backbone: {name}')
        if fsdp_config.fsdp_activation_checkpointing:
            if train_config.model_type == 'molm':
                policies.apply_molm_fsdp_checkpointing(model)
            elif train_config.model_type == 'pythia':
                policies.apply_pythia_fsdp_checkpointing(model)
            elif train_config.model_type == 't5':
               policies.apply_t5_fsdp_checkpointing(model)
            elif train_config.model_type == 'switch':
                policies.apply_switch_fsdp_checkpointing(model)
            elif train_config.model_type == 'llama_moe':
                policies.apply_llamamoe_fsdp_checkpointing(model)
            elif train_config.model_type == 'open_llama':
                policies.apply_fsdp_checkpointing(model)
            else:
                raise ValueError(f'wrong model type: {train_config.model_type}')

    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    if train_config.run_validation:
        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
    else:
        dataset_val = None
    
    if (not train_config.enable_fsdp or rank == 0) and train_config.run_validation:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.run_ood:
        dataset_ood = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="ood",
        ).dataset
    else:
        dataset_ood = None
    
    if (not train_config.enable_fsdp or rank == 0) and train_config.run_ood:
            print(f"--> OOD Validation Set Length = {len(dataset_ood)}")
    
    train_sampler = None
    val_sampler = None
    ood_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
    )

    def collate_fn(batch):
        seq_length = len(batch[0]['input_ids'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        while len(batch) < train_config.val_batch_size:
            added_data = {}
            for k, v in batch[-1].items():
                if k == 'input_ids':
                    added_data[k] = v.clone()
                elif k == 'labels':
                    added_data[k] = v
            added_data['attention_mask'] = torch.zeros(seq_length)
            batch.append(added_data)
        all_input_ids = torch.stack([item['input_ids'] for item in batch])
        all_attention_mask = torch.stack([item['attention_mask'] for item in batch])
        all_labels = torch.stack([item['labels'] for item in batch])

        return {
            'input_ids': all_input_ids,
            'attention_mask': all_attention_mask,
            'labels': all_labels
        }
    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=False,
            collate_fn = collate_fn
        )
    else:
        eval_dataloader = None
    if train_config.run_ood:
        ood_dataloaders = []
        print(dataset_ood)
        for dataset in dataset_ood:
            if train_config.enable_fsdp:
                ood_sampler = DistributedSampler(
                    dataset,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                )
            else:
                ood_sampler = None
            ood_dataloaders.append(torch.utils.data.DataLoader(
                dataset,
                batch_size=train_config.val_batch_size,
                num_workers=train_config.num_workers_dataloader,
                pin_memory=True,
                sampler=ood_sampler if ood_sampler else None,
                drop_last=False,
                collate_fn = collate_fn
            ))
    else:
        ood_dataloaders = None
    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        
    # Load the full optim static states in
    # then shard and load to optim
    try:
        if train_config.load_optimizer:
            full_osd = torch.load(os.path.join(train_config.model_name, 'optimizer.pth'))
            sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, model)
            optimizer.load_state_dict(sharded_osd)
            print(f'Loading optimizer state in {rank}')
        
    except Exception as e:
        print('Fail in Loading sharded optimizer states')
        exit(0)
    
    if train_config.scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    elif train_config.scheduler_type == 'cosine':
        all_steps = train_config.num_epochs * len(train_dataloader) // train_config.gradient_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(optimizer, all_steps*0.1, all_steps, num_cycles=3)
    elif train_config.scheduler_type == 'linear':
        all_steps = train_config.num_epochs * len(train_dataloader) // train_config.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, all_steps*0.1, all_steps)
    
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        ood_dataloaders,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        init_gate = init_gate
    )

if __name__ == "__main__":
    fire.Fire(main)
