# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import torch
import os
import sys
import yaml
from transformers import LlamaTokenizer, GPTNeoXTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification,GPTNeoXConfig, GPTNeoXForCausalLM
from moduleformer import ModuleFormerForCausalLM, ModuleFormerConfig, ModuleFormerForSequenceClassification
from switch_transformers import SwitchTransformersForConditionalGeneration,SwitchTransformersConfig
from transformers import T5ForConditionalGeneration, AutoConfig

def load_pythia_from_config(config_path):
    model_config = GPTNeoXConfig.from_pretrained(config_path)
    model = GPTNeoXForCausalLM(config=model_config)
    return model

def load_molm_from_config(config_path):
    AutoConfig.register("moduleformer", ModuleFormerConfig)
    AutoModelForCausalLM.register(ModuleFormerConfig, ModuleFormerForCausalLM)
    AutoModelForSequenceClassification.register(ModuleFormerConfig, ModuleFormerForSequenceClassification)
    model_config = AutoConfig.from_pretrained(config_path)
    model_config.num_labels = 3
    model = AutoModelForSequenceClassification.from_config(config=model_config)
    return model

def load_switch_from_config(config_path):
    model_config = SwitchTransformersConfig.from_pretrained(config_path)
    model_config.num_labels = 3
    model = SwitchTransformersForConditionalGeneration(config=model_config)
    return model

def load_t5_from_config(config_path):
    model_config = AutoConfig.from_pretrained(config_path)
    model_config.num_labels = 3
    model = T5ForConditionalGeneration(config=model_config)
    return model
# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)
from model_checkpointing import load_sharded_model_single_gpu

def main(
    mtype="", # type of hf model
    fsdp_checkpoint_path="", # Path to FSDP Sharded model checkpoints
    consolidated_model_path="", # Path to save the HF converted model checkpoints
    HF_model_path_or_name="" # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
    ):
    
    try:
        file_name = 'train_params.yaml'
        # Combine the directory and file name to create the full path
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        # Open the file
        with open(train_params_path, 'r') as file:
            # Load the YAML data
            data = yaml.safe_load(file)
            # Access the 'model_name' field
            HF_model_path_or_name = data.get('model_name')

            print(f"Model name: {HF_model_path_or_name}")
    except FileNotFoundError:
        if HF_model_path_or_name == "":
            print(f"The file {train_params_path} does not exist.")
            HF_model_path_or_name = input("Please enter the model name: ")
            print(f"Model name: {HF_model_path_or_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    try:
        file_name = 'optimizer.pth'
        if not os.path.exists(consolidated_model_path):
            os.mkdir(consolidated_model_path)
        # Combine the directory and file name to create the full path
        _from = os.path.join(fsdp_checkpoint_path, file_name)
        _to = os.path.join(consolidated_model_path, file_name)
        import shutil
        shutil.copyfile(_from, _to)
        print(f'optimizer copyed  {_from=}, {_to=}')
    except Exception as e:
        print(f'Can not find optimi static state {_from=}, {_to=}')
        print(e)
        exit(0)
        
    #load the HF model definition from config
    if mtype == 'pythia':
        model_def = load_pythia_from_config(HF_model_path_or_name)
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(HF_model_path_or_name)
    elif mtype == 'molm':
        model_def = load_molm_from_config(HF_model_path_or_name)
        tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
    elif mtype == 'switch':
        model_def = load_switch_from_config(HF_model_path_or_name)
        tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
    elif mtype == 't5':
        model_def = load_t5_from_config(HF_model_path_or_name)
        tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
    else:
        print(f'not exist type {mtype=}')
        raise NotImplementedError
    
    print("model is loaded from config")
    #load the FSDP sharded checkpoints into the model
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("model is loaded from FSDP checkpoints")
    #loading the tokenizer form the  model_path
    
    tokenizer.save_pretrained(consolidated_model_path)
    #save the FSDP sharded checkpoints in HF format
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")
if __name__ == "__main__":
    fire.Fire(main)
