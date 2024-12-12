import json
import random
import pandas as pd

def prepare_safety_anchor_datasets(anchor_size = 64):
    advbench = pd.read_csv('datasets/AdvBench.csv')
    harmful_texts = advbench['goal'].tolist()
    truthfulqa = pd.read_csv('datasets/TruthfulQA.csv')
    harmless_texts = truthfulqa['Question'].tolist()
    
    harmless_texts_cut = random.sample(harmless_texts, len(harmful_texts)) # for balance 
    harmless_texts_anchor = random.sample(harmless_texts_cut, anchor_size)
    harmful_texts_anchor = random.sample(harmful_texts, anchor_size)
    
    harmless_texts_test = list(set(harmless_texts)-set(harmless_texts_anchor))
    harmful_texts_test = list(set(harmful_texts)-set(harmful_texts_anchor))

    return harmful_texts_anchor, harmless_texts_anchor, harmful_texts_test, harmless_texts_test 

def load_xstest(data_path): # unbalance
    xstest = pd.read_csv(data_path)
    harmful_texts = xstest[xstest['type'].str.contains('contrast_')]['prompt'].tolist() # num: 200
    harmless_texts = xstest[~xstest['type'].str.contains('contrast_')]['prompt'].tolist() # num: 250
    
    return harmful_texts, harmless_texts

def load_malicious_instruct_with_100_heldout_harmless(data_path): # balance
    data_path = data_path.split(',')
    if 'MaliciousInstruct' in data_path[0]: # num: 100
        data_path = [data_path[1], data_path[0]]
        
    with open(data_path[0]) as f:
        harmless_texts = f.readlines()
    with open(data_path[1]) as f:
        harmful_texts = f.readlines()

    return harmful_texts, harmless_texts

def load_llm_safeguard_200_heldout(data_path): # balance
    with open(data_path) as f:
        texts = f.readlines()
        
    return texts[:100], texts[100:]

def load_oktest_with_HarmfulQ(data_path): # unbalance
    data_path = data_path.split(',')
    if 'HarmfulQ' in data_path[0]:
        data_path = [data_path[1], data_path[0]]

    oktest = pd.read_csv(data_path[0]) # num: 300
    ok_texts = oktest['prompt'].tolist()
    with open(data_path[1], "r") as file: # num: 200
        harm_texts = json.load(file)
        
    return harm_texts, ok_texts

def load_RE_dataset(data_path):
    re_dataset = pd.read_parquet(data_path)
    re_dataset = re_dataset['sentence'].tolist()
    harmful_texts = []
    harmless_texts = []
    for data in re_dataset:
        harmful_texts.append(data[1])
        harmless_texts.append(data[0])
        
    return harmful_texts, harmless_texts
