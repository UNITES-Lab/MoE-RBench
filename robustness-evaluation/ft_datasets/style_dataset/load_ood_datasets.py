import os
from datasets import load_dataset,load_from_disk
from datasets import Dataset
import torch
task_map = {'NLI':['mnli, cnli, anli, wanli'],}

def get_ood_style_dataset(dataset_config, tokenizer, partition, for_decoder=False):
    return OOD_Dataset(dataset_config, tokenizer, partition, for_decoder)

class OOD_Dataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", for_decoder=False):
        self.tokenizer = tokenizer
        if partition == "train":
            self.dataset = load_dataset('glue','sst2',split='train')
        elif partition == 'test':
            self.dataset = load_dataset('glue','sst2',split='validation')
        elif partition == 'ood':
            style_dataset = load_dataset('AI-Secure/DecodingTrust', 'ood', split='style')
            dataset_dict = {}
            for example in style_dataset:
                if not example['split'] == 'dev':
                    continue
                category = example['category']
                if category == 'base':
                    continue
                if category not in dataset_dict.keys():
                    dataset_dict[category] = {'sentence':[],'label':[]}
                dataset_dict[category]['sentence'].append(example['question_sentence'])
                dataset_dict[category]['label'].append(example['answer'])
            for ood_name, ood_dataset in dataset_dict.items():
                ood_dataset = Dataset.from_dict(ood_dataset)
                dataset_dict[ood_name] = ood_dataset.map(
                    lambda x: self.preprocess(tokenizer=tokenizer, batch=x, for_decoder=for_decoder),
                    num_proc=16,
                    remove_columns=ood_dataset.column_names,
                    batched=True,
                    desc='preprocess and tokenize')
                dataset_dict[ood_name].set_format("torch", columns=["input_ids", "attention_mask", "labels"])
                print(f"{ood_name}:{ood_dataset}")
            self.dataset = dataset_dict
            
        if not partition == 'ood':
            self.dataset = self.dataset.map(
                lambda x: self.preprocess(tokenizer=tokenizer, batch=x, for_decoder=for_decoder),
                num_proc=16,
                remove_columns=self.dataset.column_names,
                batched=True,
                desc='preprocess and tokenize')
            self.dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        
    def preprocess(self, tokenizer, batch, for_decoder=False):
        targets = batch['label']
        labels = []
        prompt = ['Given the sentences: ','\n what is the sentiment of this sentence ? ']
        inputs = [prompt[0]+sentence+prompt[1] for sentence in batch['sentence']]
        candidate_seq2seq = {0:'negtive', 1:'postive','0':'negtive', '1':'postive'}
        candidate_seq2class = {0:0, 1:1,'negtive':0, 'postive':1, '0':0, '1':1}
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
        return ret
    
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
