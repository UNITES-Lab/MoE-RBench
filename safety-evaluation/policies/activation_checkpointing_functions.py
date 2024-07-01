# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
import os
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from transformers.models.t5.modeling_t5 import T5Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from moduleformer.modeling_moduleformer import ModuleFormerBlock
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from LLaMA_MoE.modeling_llama_moe_hf import LlamaMoEDecoderLayer
from functools import partial

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
molm_check_fn = lambda submodule: isinstance(submodule, ModuleFormerBlock) #or isinstance(submodule, nn.Embedding))
pythia_check_fn = lambda submodule: isinstance(submodule, GPTNeoXLayer) #or isinstance(submodule, nn.Embedding))
llama_moe_check_fn = lambda submodule: isinstance(submodule, LlamaMoEDecoderLayer) #or isinstance(submodule, nn.Embedding))

def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

def apply_molm_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying molm fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=molm_check_fn
    )

def apply_llama_moe_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying llama-moe fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=llama_moe_check_fn
    )

def apply_pythia_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying pythia fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=pythia_check_fn
    )
