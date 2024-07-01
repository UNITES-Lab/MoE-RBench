import os
from datasets import load_dataset,load_from_disk
from datasets import Dataset
import torch
import copy

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

def get_adv_nli_dataset(dataset_config, tokenizer, partition, for_decoder=False, prompt_type=0):
    return adv_Dataset(dataset_config, tokenizer, partition, for_decoder, prompt_type=prompt_type)

class adv_Dataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", for_decoder=False, prompt_type=0):
        self.prompt_type = prompt_type
        self.partition = partition
        if 'dev' in partition:
            self.dataset = load_dataset('snli', split='validation')
            self.dataset = self.dataset.filter(lambda example: example['label'] != -1)
            print(f"validation dataset:{self.dataset}")
        else:
            self.dataset = load_dataset('anli', split=partition)
        self.dataset = self.dataset.map(
            lambda x: self.preprocess(tokenizer=tokenizer, batch=x, for_decoder=for_decoder),
            num_proc=32,
            remove_columns=self.dataset.column_names,
            batched=True,
            desc='preprocess and tokenize',
            load_from_cache_file=False
            )
        if 'train' in partition:
            from datasets import concatenate_datasets
            esnli = load_dataset('snli', split='train').shuffle(42)
            esnli = esnli.filter(lambda example: example['label'] != -1)
            # esnli = esnli.rename_column('explanation_1', 'reason')
            if partition[-1]=='1':
                esnli = esnli[:len(esnli)//9]
            elif partition[-1]=='2':
                esnli = esnli[len(esnli)//3: len(esnli)//3+1*len(esnli)//6]
            else:
                esnli = esnli[2*len(esnli)//3:]
            esnli = Dataset.from_dict(esnli)
            esnli = esnli.map(
                lambda x: self.preprocess(tokenizer=tokenizer, batch=x, for_decoder=for_decoder),
                num_proc=32,
                remove_columns=esnli.column_names,
                batched=True,
                batch_size=128,
                desc='preprocess and tokenize',
                load_from_cache_file=False)
            print(f"{esnli=}, anli={self.dataset}")
            self.dataset = concatenate_datasets([self.dataset, esnli]).shuffle(42)
        self.dataset.set_format("torch")
        self.tokenizer = tokenizer
        
    def preprocess(self, tokenizer, batch, for_decoder=False):
        labels = []
        prompt_1 = ['Is this true ?\n','\n implies \n','\n Answer: ']
        inputs = []
        labels = []
        for premise, hypothesis,label in zip(batch['premise'], batch['hypothesis'],batch['label']):
            prompt = prompt_1
            candidate_seq2seq = {2:'No',1:'Maybe',0:'Yes'}
            input = prompt[0]+premise+prompt[1]+hypothesis+prompt[2]
            if not for_decoder:
                label = candidate_seq2seq[label]
            inputs.append(input)
            labels.append(label)
        inputs = tokenizer(inputs, truncation=True, padding='max_length', max_length=256, return_attention_mask=True, add_special_tokens=True)
        if not for_decoder:
            labels = tokenizer(labels, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=True)['input_ids']
            labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.tolist()
            
        ret = {"input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels}
        return ret
    
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
