'''
Some of the code refer to
https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import random
import numpy as np
import torch

from datasets import load_dataset,load_from_disk
from torch.utils.data.dataset import Dataset

def get_wikitext2():
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    return traindata, testdata

def get_ptb():
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    return traindata, valdata

def get_c4():
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', use_auth_token=False)
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', use_auth_token=False)

    return traindata, valdata

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

def process_data(samples, tokenizer, seq_len, field_name, use_chat=False, use_vicuna=False):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]#.replace("\n", " ")
    if use_chat:
        test_ids = test_ids[1:] # ignore the <s>
        if use_vicuna:
            start, end = "<s>USER: ", " \nASSISTANT: "
        else:
            start, end = "<s>[INST] ", " [/INST] "
        start_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(start))).to(test_ids.device)
        print(start_ids)
        end_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(end))).to(test_ids.device)
        print(end_ids)
        seq_len -= len(start_ids)+len(end_ids)
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        if use_chat:
            batch = torch.cat((start_ids,batch[:len(batch)//2],end_ids,batch[-len(batch)//2:]))
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)


def process_c4data(samples, tokenizer, seq_len, field_name, use_chat=False, use_vicuna=False):
    if use_chat:
        if use_vicuna:
            start, end = "<s>USER: ", " \nASSISTANT: "
        else:
            start, end = "<s>[INST] ", " [/INST]"
        start_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(start)))
        print(start_ids)
        end_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(end)))
        print(end_ids)
        seq_len -= len(start_ids)+len(end_ids)
    
    random.seed(0)
    test_ids_batch = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(samples) - 1)
            tmp = tokenizer(samples[i][field_name], return_tensors='pt')#
            if use_chat:
                tmp.input_ids = tmp.input_ids[:,1:] # ignore the <s>
            if tmp.input_ids.shape[1] > seq_len:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seq_len - 1)
        j = i + seq_len
        batch = tmp.input_ids[0, i:j]
        if use_chat:
            batch = torch.cat((start_ids,batch[:len(batch)//2],end_ids,batch[-len(batch)//2:]))
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)

def get_loaders(name, tokenizer, seq_len=2048, batch_size = 8, use_chat=False, use_vicuna=False):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2()
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text', use_chat, use_vicuna)
    if 'ptb' in name:
        train_data, test_data = get_ptb()
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence', use_chat, use_vicuna)
    if 'c4' in name:
        train_data, test_data = get_c4()
        test_dataset = process_c4data(test_data, tokenizer, seq_len, 'text', use_chat, use_vicuna)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader

def get_ppl_datasets(name):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2()
        test_data = test_data['text']
    if 'ptb' in name:
        train_data, test_data = get_ptb()
        test_data = test_data['sentence']
    
    return train_data, test_data