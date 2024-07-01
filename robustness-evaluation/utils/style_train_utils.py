# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
from typing import List
import yaml
import time

import fire
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import pdb
import wandb
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from torch.nn import functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
from pkg_resources import packaging
from .memory_utils import MemoryTrace
import model_checkpointing
import torch.cuda.nccl as nccl
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))
from policies import (
    bfSixteen, fpSixteen,bfSixteen_mixed, 
    get_llama_wrapper, get_molm_wrapper, get_pythia_wrapper, get_t5_wrapper, get_switch_wrapper)

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    
# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def sst_train(model, train_dataloader, eval_dataloader, ood_dataloaders, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None,
          init_gate=None, round=0):
    """
    Trains the model on the given dataloader
    
    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons
    
    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler() 
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_acc = float("-inf")
    bilevel_state = None # whether router is freeze
    global_step = 0
    for epoch in range(train_config.num_epochs // (round+1)):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            if not train_config.enable_fsdp or rank == 0: pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length)
            for step, batch in enumerate(train_dataloader):
            # for step, batch in enumerate(tqdm(train_dataloader,colour="blue", desc=f"Training Epoch{epoch}")):
                for key in batch.keys():
                    if train_config.enable_fsdp:
                        batch[key] = batch[key].to(local_rank)
                    else:
                        batch[key] = batch[key].to('cuda:0')  
                global_step += 1          
                loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                if not loss.isnan(): # in case the loss is nan (this may sometimes happen due to cropping the training samples by length = `max_words`)
                    total_loss += loss.detach().float()
                if train_config.use_fp16:
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        if train_config.scheduler_type in ['cosine', 'linear']:
                            lr_scheduler.step()
                        if not train_config.enable_fsdp or rank == 0: pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    if train_config.run_bilevel and bilevel_state:
                        loss = loss / 2 # smaller grad for router
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        if train_config.scheduler_type in ['cosine', 'linear']:
                            lr_scheduler.step()
                        # lr_scheduler.step() # For Huggingface LR schedulers (update lr after each gradient step)
                        if not train_config.enable_fsdp or rank == 0: pbar.update(1)
                
                if not train_config.enable_fsdp or rank == 0: pbar.set_description(f"Training Epoch: {epoch + 1}/{train_config.num_epochs}, step {step + 1}/{len(train_dataloader)} completed (loss: {loss.detach().float()}, lr: {optimizer.param_groups[0]['lr']})")
                if rank==0:
                    wandb.log({"Training Epoch": (epoch + 1)/train_config.num_epochs,
                            "Global step": global_step,
                            "loss": loss.detach().float(),
                            "lr": optimizer.param_groups[0]['lr']}, step=global_step)
                scores = {}
                if train_config.run_validation and global_step % train_config.run_validation_steps == 0:
                    eval_ppl, eval_epoch_loss, eval_acc, eval_not_in_labels = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, 'validation:')
                    ood_ppl, ood_epoch_loss, ood_acc, ood_not_in_labels = 0, 0, 0, 0
                    if train_config.run_ood:
                        for ood_name, ood_dataloader in ood_dataloaders.items():
                            print(f'evaluate ood:{ood_name}')
                            _ood_ppl, _ood_epoch_loss, _ood_acc, _ood_not_in_labels = evaluation(model, train_config, ood_dataloader, local_rank, tokenizer, 'ood-test:')
                            scores['ood/'+ood_name+'_acc'] = _ood_acc
                            ood_ppl += _ood_ppl
                            ood_epoch_loss += _ood_acc
                            ood_acc += _ood_acc
                            ood_not_in_labels += _ood_not_in_labels

                        ood_ppl /= len(ood_dataloaders)
                        ood_epoch_loss /= len(ood_dataloaders)
                        ood_acc /= len(ood_dataloaders)
                        ood_not_in_labels /= len(ood_dataloaders)
                    checkpoint_start_time = time.perf_counter()
                    if train_config.save_model and eval_acc > best_acc and global_step >= total_length*0.5:
                        if train_config.enable_fsdp:
                            dist.barrier()
                        if train_config.use_peft:
                            if train_config.enable_fsdp:
                                if rank==0:
                                    print(f"we are about to save the PEFT modules")
                            else:
                                print(f"we are about to save the PEFT modules")
                            model.save_pretrained(train_config.output_dir)  
                            if train_config.enable_fsdp:
                                if rank==0: 
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")
                            else:
                                print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                
                        else:
                            if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                                
                                model_checkpointing.save_model_checkpoint(
                                    model, optimizer, rank, train_config, epoch=epoch
                                )
                            elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                if local_rank == 0:
                                    print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                
                                if train_config.save_optimizer: 
                                    model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                    if local_rank == 0:
                                        print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                else:
                                    model_checkpointing.save_model_and_optimizer_sharded(model, rank, train_config)
                        if train_config.enable_fsdp:
                            dist.barrier()
                    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                    checkpoint_times.append(checkpoint_end_time)
                    if eval_acc > best_acc:
                        best_acc = eval_acc
                        if train_config.enable_fsdp:
                            if rank==0:
                                print(f"best eval acc and ood acc on global step {global_step} is {eval_acc=}, {ood_acc=}")
                        else:
                            print(f"best eval acc and ood acc on global step {global_step} is is {eval_acc=}, {ood_acc=}")
                    wandb_log = {
                        'eval/eval_acc':eval_acc,
                        'eval/ood_acc':ood_acc,
                        'eval/eval_loss':eval_epoch_loss,
                        'eval/ood_loss':ood_epoch_loss,
                        'eval/acc_sum':eval_acc+ood_acc,
                        'eval/eval_complete_wrong': eval_not_in_labels,
                        'eval/ood_complete_wrong': ood_not_in_labels,
                    }
                    for k,v in scores.items():
                        wandb_log[k] = v
                    if rank==0:
                        wandb.log(wandb_log, step=global_step)
                    val_loss.append(eval_epoch_loss)
                    val_prep.append(eval_ppl)
        
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)    
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Max CUDA memory allocated was {memtrace.peak} GB")
                print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            print(f"Max CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        
        # Update the learning rate as needed
        if train_config.scheduler_type in ['step']:
            if train_config.run_bilevel:
                if (epoch+1) % 2 == 0:
                    lr_scheduler.step()
                pass
            else:
                lr_scheduler.step()
          
        
        
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")

        
    avg_epoch_time = sum(epoch_times)/ len(epoch_times) 
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times)   
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep) 
        avg_eval_loss = sum(val_loss)/len(val_loss) 

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and train_config.save_model:
        save_train_params(train_config, fsdp_config, rank)
        
    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, prefix=""):
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
    model.eval()
    # ct = 0
    eval_preds = []
    eval_labels = []
    eval_loss = torch.zeros(1).to(model.device)  # Initialize evaluation loss
    filtered_examples = 0

    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating "+prefix, disable= True)):
            
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if not loss.isnan(): eval_loss += loss.detach().float()
                # else: ct += 1
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            for mask_id in range(len(batch['attention_mask'])):
                mask = batch['attention_mask'][mask_id]
                if torch.sum(mask) == 0:
                    # filter the padded examples
                    preds = torch.argmax(outputs.logits[:mask_id], -1)
                    batch['labels'] = batch['labels'][:mask_id]
                    filtered_examples += len(batch['attention_mask']) - mask_id
                    break
            
            eval_preds.extend(preds.detach().cpu().numpy())
            eval_labels.extend(batch['labels'].detach().cpu().numpy())
    # print("NaN occurences:", ct)
    if train_config.model_type in ['t5', 'switch']:
        eval_preds = tokenizer.batch_decode(eval_preds, skip_special_tokens=True)
       
        for label in eval_labels:
            for i in range(len(label)):
                if label[i] == -100:
                    label[i] = tokenizer.pad_token_id
        eval_labels = tokenizer.batch_decode(eval_labels, skip_special_tokens=True)
        pres = []
        labels = []
        for pre, label in zip(eval_preds,eval_labels):
            if len(pre.strip().split()) > 0:
                pres.append(pre.strip().split()[0])
            else:
                pres.append('Error')
            labels.append(label.strip().split()[0])
        eval_preds = pres
        eval_labels = labels
        if train_config.task_name == 'stsb':
            eval_labels = [int(label[0]) if label[0] in ['0','1','2','3','4','5'] else -1 for label in eval_labels]
            eval_preds = [int(label[0]) if label[0] in ['0','1','2','3','4','5'] else -1 for label in eval_preds]
    if local_rank == 0:
        print(eval_labels[:3], eval_preds[:3])
    if train_config.task_name == 'stsb':
        from scipy.stats import pearsonr
        import numpy as np
        eval_labels_np = np.array(eval_labels)
        eval_preds_np= np.array(eval_preds)
        Pearson_Corr, _ = pearsonr(eval_labels_np, eval_preds_np)
        correct_rate = Pearson_Corr
    else:
        if train_config.model_type in ['t5', 'switch']:
            correct_predictions = [1 if pred[:len(true_label)] == true_label else 0 for pred, true_label in zip(eval_preds, eval_labels)]
        else:
            correct_predictions = [1 if pred == true_label else 0 for pred, true_label in zip(eval_preds, eval_labels)]
        correct_rate = sum(correct_predictions)/len(correct_predictions)
    not_in_labels = [0 if pred in eval_labels else 1 for pred in eval_preds]
    not_in_labels_rate = sum(not_in_labels) / len(not_in_labels)
    correct_rate = torch.tensor(correct_rate).to(model.device)
    not_in_labels_rate = torch.tensor(not_in_labels_rate).to(model.device)
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_rate, op=dist.ReduceOp.SUM)
        dist.all_reduce(not_in_labels_rate, op=dist.ReduceOp.SUM)
    
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
        correct_rate = correct_rate/world_size
        not_in_labels_rate = not_in_labels_rate/world_size
    eval_ppl = torch.exp(eval_epoch_loss)
    
    eval_epoch_loss = eval_epoch_loss.item()
    correct_rate = correct_rate.item()
    not_in_labels_rate = not_in_labels_rate.item()
    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(prefix + f"{eval_epoch_loss=} {correct_rate=} {not_in_labels_rate=} {filtered_examples=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=} {correct_rate=}")
    if train_config.enable_fsdp:
        dist.barrier()
    return eval_ppl, eval_epoch_loss, correct_rate, not_in_labels_rate

def test(model, train_dataloader, eval_dataloader, ood_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None,
          init_gate=None):
    """
    Trains the model on the given dataloader
    
    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons
    
    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler() 
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"]) 
    with MemoryTrace() as memtrace:  # track the memory usage
        model.eval()
        eval_ppl, eval_epoch_loss, eval_acc, eval_not_in_labels = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, 'validation:')
        ood_ppl, ood_epoch_loss, ood_acc, ood_not_in_labels = evaluation(model, train_config, ood_dataloader, local_rank, tokenizer, 'ood-test:')
        test_result = {}
        test_result['eval_ppl'] = eval_ppl
        test_result['eval_epoch_loss'] = eval_epoch_loss
        test_result['eval_acc'] = eval_acc
        test_result['eval_not_in_labels'] = eval_not_in_labels
        test_result['ood_ppl']  = ood_ppl
        test_result['ood_epoch_loss'] = ood_epoch_loss 
        test_result['ood_acc'] = ood_acc
        test_result['ood_not_in_labels'] = ood_not_in_labels
        if rank == 0:
            print(test_result)

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")
                
                
def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' 
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")




def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        # bf16_ready = False

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    
    wrapping_policy = get_llama_wrapper()
      
    return mixed_precision_policy, wrapping_policy

def get_llama_moe_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        # bf16_ready = False

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    
    # wrapping_policy = get_llama_moe_wrapper()
    wrapping_policy = None
    return mixed_precision_policy, wrapping_policy

def get_molm_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        # bf16_ready = False

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    
    wrapping_policy = get_molm_wrapper()
      
    return mixed_precision_policy, wrapping_policy

def get_pythia_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        # bf16_ready = False

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    
    wrapping_policy = get_pythia_wrapper()
      
    return mixed_precision_policy, wrapping_policy

def get_t5_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        # bf16_ready = False

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    
    wrapping_policy = get_t5_wrapper()
      
    return mixed_precision_policy, wrapping_policy

def get_switch_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""
    
    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        # bf16_ready = False

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    
    wrapping_policy = get_switch_wrapper()
      
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries, 
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    # + "-"
    # + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")