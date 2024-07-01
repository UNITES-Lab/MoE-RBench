# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_dataset/alpaca_data_no_safety.json"


@dataclass
class alpaca500safe_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_dataset/alpaca_500safe.json"


@dataclass
class alpaca1ksafe_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/alpaca_dataset/alpaca_1ksafe.json"


@dataclass
class dolly_dataset:
    dataset: str = "dolly_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl"

    
@dataclass
class aoa_dataset:
    dataset: str =  "aoa_dataset"
    data_path: str = "ft_datasets/aoa_dataset"
    train_split: str = "train.json"


@dataclass
class pure_bad_dataset:
    dataset: str =  "pure_bad_dataset"
    # train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_100.jsonl"
    # train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_50.jsonl"
    # train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_10.jsonl"
    train_split: str = "ft_datasets/pure_bad_dataset/pure_bad_10_demo.jsonl"



@dataclass
class ood_nli_dataset_seq2class:
    dataset: str = "ood_nli_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/nli_dataset/data"


@dataclass
class ood_nli_dataset_seq2seq:
    dataset: str = "ood_nli_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/nli_dataset/data"

class adv_snli_dataset_seq2class:
    dataset: str = "adv_snli_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/snli_dataset/data"


@dataclass
class adv_snli_dataset_seq2seq:
    dataset: str = "adv_snli_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/snli_dataset/data"

@dataclass
class ood_sa_dataset_seq2class:
    dataset: str = "ood_sa_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/sa_dataset/data"


@dataclass
class ood_sa_dataset_seq2seq:
    dataset: str = "ood_sa_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/sa_dataset/data"
    
@dataclass
class ood_nli_dataset_seq2class_2:
    dataset: str = "ood_nli_dataset_seq2class_2"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/nli_dataset_2/data"

@dataclass
class ood_nli_dataset_seq2seq_2:
    dataset: str = "ood_nli_dataset_seq2seq_2"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/nli_dataset_2/data"

@dataclass
class adv_dataset_seq2seq_0:
    dataset: str = "adv_dataset_seq2seq_0"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/adv_dataset/data"

@dataclass
class adv_dataset_seq2seq_1:
    dataset: str = "adv_dataset_seq2seq_1"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/adv_dataset/data"

@dataclass
class adv_dataset_seq2seq_2:
    dataset: str = "adv_dataset_seq2seq_2"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/adv_dataset/data"

@dataclass
class adv_dataset_seq2seq_decoder_0:
    dataset: str = "adv_dataset_seq2seq_0"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/adv_dataset/data"

@dataclass
class adv_dataset_seq2seq_decoder_1:
    dataset: str = "adv_dataset_seq2seq_1"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/adv_dataset/data"

@dataclass
class adv_dataset_seq2seq_decoder_2:
    dataset: str = "adv_dataset_seq2seq_2"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/adv_dataset/data"

@dataclass
class adv_dataset_seq2class:
    dataset: str = "adv_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/adv_dataset/data"

@dataclass
class ood_td_dataset_seq2class:
    dataset: str = "ood_td_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/td_dataset/data"

@dataclass
class ood_td_dataset_seq2seq:
    dataset: str = "ood_td_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/td_dataset/data"


@dataclass
class ood_qa_dataset_seq2class:
    dataset: str = "ood_qa_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/qa_dataset/data"

@dataclass
class ood_qa_dataset_seq2seq:
    dataset: str = "ood_qa_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/qa_dataset/data"
@dataclass
class gluex_cola_dataset_seq2class:
    dataset: str = "gluex_cola_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"


@dataclass
class gluex_cola_dataset_seq2seq:
    dataset: str = "gluex_cola_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"

@dataclass
class gluex_stsb_dataset_seq2class:
    dataset: str = "gluex_stsb_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"


@dataclass
class gluex_stsb_dataset_seq2seq:
    dataset: str = "gluex_stsb_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"

@dataclass
class gluex_qnli_dataset_seq2class:
    dataset: str = "gluex_qnli_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"


@dataclass
class gluex_qnli_dataset_seq2seq:
    dataset: str = "gluex_qnli_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"

@dataclass
class gluex_rte_dataset_seq2class:
    dataset: str = "gluex_rte_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"


@dataclass
class gluex_rte_dataset_seq2seq:
    dataset: str = "gluex_rte_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"

@dataclass
class gluex_mrpc_dataset_seq2class:
    dataset: str = "gluex_mrpc_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"


@dataclass
class gluex_mrpc_dataset_seq2seq:
    dataset: str = "gluex_mrpc_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"

@dataclass
class gluex_qqp_dataset_seq2class:
    dataset: str = "gluex_qqp_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"


@dataclass
class gluex_qqp_dataset_seq2seq:
    dataset: str = "gluex_qqp_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"

@dataclass
class ood_style_dataset_seq2class:
    dataset: str = "ood_style_dataset_seq2class"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"


@dataclass
class ood_style_dataset_seq2seq:
    dataset: str = "ood_style_dataset_seq2seq"
    train_split: str = "train"
    test_split: str = "val"
    ood_split: str = "ood"
    data_path: str = "ft_datasets/gluex_dataset/data"