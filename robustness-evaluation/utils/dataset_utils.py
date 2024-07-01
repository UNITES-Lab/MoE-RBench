# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets import (
    get_ood_nli_dataset,
    get_adv_nli_dataset,
    get_ood_sa_dataset,
    get_ood_td_dataset,
    get_adv_snli_dataset,
    get_ood_style_dataset
)
from typing import Optional


DATASET_PREPROC = {
    "ood_nli_dataset_seq2class": partial(get_ood_nli_dataset, for_decoder=True),
    "ood_nli_dataset_seq2seq": partial(get_ood_nli_dataset, for_decoder=False),
    "adv_snli_dataset_seq2class": partial(get_adv_snli_dataset, for_decoder=True),
    "adv_snli_dataset_seq2seq": partial(get_adv_snli_dataset, for_decoder=False),
    "adv_dataset_seq2class": partial(get_adv_nli_dataset, for_decoder=True, prompt_type=0),
    "adv_dataset_seq2seq": partial(get_adv_nli_dataset, for_decoder=False, prompt_type=0),
    "ood_sa_dataset_seq2class": partial(get_ood_sa_dataset, for_decoder=True),
    "ood_sa_dataset_seq2seq": partial(get_ood_sa_dataset, for_decoder=False),
    "ood_td_dataset_seq2class": partial(get_ood_td_dataset, for_decoder=True),
    "ood_td_dataset_seq2seq": partial(get_ood_td_dataset, for_decoder=False),
    "ood_style_dataset_seq2class": partial(get_ood_style_dataset, for_decoder=True),
    "ood_style_dataset_seq2seq": partial(get_ood_style_dataset, for_decoder=False),
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        split,
    )
