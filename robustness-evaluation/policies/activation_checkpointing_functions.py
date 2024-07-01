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

from t5 import T5Block
from llama_moe import LlamaMoEDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from moduleformer.modeling_moduleformer import ModuleFormerBlock
from gpt_neox import GPTNeoXLayer
from switch_transformers import SwitchTransformersBlock
from functools import partial

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
molm_check_fn = lambda submodule: isinstance(submodule, ModuleFormerBlock) #or isinstance(submodule, nn.Embedding))
pythia_check_fn = lambda submodule: isinstance(submodule, GPTNeoXLayer) #or isinstance(submodule, nn.Embedding))
switch_check_fn = lambda submodule: isinstance(submodule, SwitchTransformersBlock)
t5_check_fn = lambda submodule: isinstance(submodule, T5Block)
llamamoe_check = lambda submodule: isinstance(submodule, LlamaMoEDecoderLayer)


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

def apply_pythia_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying pythia fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=pythia_check_fn
    )
def apply_switch_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying switch fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=switch_check_fn
    )

def apply_t5_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying t5 fsdp activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=t5_check_fn
    )

def apply_llamamoe_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying llama-moe activation checkpointing...")

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=llamamoe_check
    )
