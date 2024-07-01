import os
from datasets import load_dataset,load_from_disk
from datasets import Dataset
import torch
task_map = {'NLI':['mnli, cnli, anli, wanli'],}

def load_tsv_dataset(base_path, report=False):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(base_path, delimiter='\t')
    data_dict = df.to_dict(orient='list')
    if 'Text' in data_dict.keys():
        data_ls = [[i,str(int(j))] for i,j in zip(data_dict['Text'], data_dict["Label"]) if isinstance(i, str)]
        data_np = np.array(data_ls)
        data_dict['Label'] = data_np[:,1]
        data_dict['Text'] = data_np[:,0]
    else:
        abonded_sum = 0
        data_ls = [[i,j,str(k)] for i,j,k in zip(data_dict['Hypothesis'], data_dict['Premise'], data_dict["Label"]) if isinstance(i, str) and isinstance(j, str) and not k == '-']
        abonded_sum = len(data_dict['Label']) - len(data_ls)
        if report:
            print('abondened examples:',abonded_sum)
        data_np = np.array(data_ls)
        
        data_dict["Label"] = data_np[:,2]
        data_dict['Hypothesis'] = data_np[:,0]
        data_dict['Premise'] = data_np[:,1]
    return Dataset.from_dict(data_dict)

def get_ood_sa_dataset(dataset_config, tokenizer, partition, for_decoder=False):
    return OOD_Dataset(dataset_config, tokenizer, partition, for_decoder)

class OOD_Dataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", for_decoder=False):
        if partition == "train":
            pth = os.path.join(dataset_config.data_path, 'amazon', 'train.tsv')
            self.dataset = load_tsv_dataset(pth)
        elif partition == 'test':
            pth = os.path.join(dataset_config.data_path, 'amazon', 'test.tsv')
            self.dataset = load_tsv_dataset(pth)
        elif partition == 'ood':
            pths = [os.path.join(dataset_config.data_path, 'dynasent','test.tsv'),os.path.join(dataset_config.data_path, 'semeval','test.tsv'),os.path.join(dataset_config.data_path, 'sst','test.tsv')]
            ood_datasets = []
            for pth in pths:
                ood_datasets.append(load_tsv_dataset(pth))
            self.dataset = ood_datasets
        
        if isinstance(self.dataset, list):
            for i in range(len(self.dataset)):
                self.dataset[i] = self.dataset[i].map(
                    lambda x: self.preprocess(tokenizer=tokenizer, batch=x, for_decoder=for_decoder),
                    num_proc=16,
                    remove_columns=self.dataset[i].column_names,
                    batched=True,
                    desc='preprocess and tokenize'
                )
                self.dataset[i].set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        else:
            self.dataset = self.dataset.map(
                lambda x: self.preprocess(tokenizer=tokenizer, batch=x, for_decoder=for_decoder),
                num_proc=16,
                remove_columns=self.dataset.column_names,
                batched=True,
                desc='preprocess and tokenize')
            self.dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        self.tokenizer = tokenizer
        
    def preprocess(self, tokenizer, batch, for_decoder=False):
        targets = batch['Label']
        labels = []
        prompt = ["\n All in all , the sentiment was"]
        inputs = [text + prompt[0] for text in batch['Text']]
        candidate_seq2seq = {0: "negative", 2: "neutral", 1: "positive", "0": "negative", "2": "neutral", "1": "positive"}
        candidate_seq2class = {'2':2,'1':1,'0':0,"negative":0, "neutral":2, "positive":1}
        for target in targets:
            if for_decoder:
                labels.append(candidate_seq2class[target])
            else:
                labels.append(candidate_seq2seq[target])
        inputs = tokenizer(inputs, truncation=True, padding='max_length', return_attention_mask=True, add_special_tokens=True)
        if for_decoder:
            pass
        else:
            labels = tokenizer(labels, truncation=True, padding=True, return_attention_mask=False, add_special_tokens=True)['input_ids']
            labels = torch.tensor(labels, dtype=torch.long)
            labels[labels==tokenizer.pad_token_id] = -100
            labels = labels.tolist()
        ret = {"input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels}
        # for k,v in ret.items():
        #     print(k, v[:2])
        return ret
    
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
