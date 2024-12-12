# Below is the implementaion of our SCANS({S}afety-{C}onscious {A}ctivatio{n} {S}teering).
import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import List
from utils.llama_wrapper import LlamaWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.modeling_utils import get_vicuna_hidden_states, get_vicuna_hidden_states_with_inst_pos, is_reject
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score
from utils.load_safety_dataset import prepare_safety_anchor_datasets, load_xstest, load_oktest_with_HarmfulQ, load_llm_safeguard_200_heldout, load_malicious_instruct_with_100_heldout_harmless, load_RE_dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def get_safety_vectors(model, tokenizer, harmful_texts, harmless_texts):

    hidden_states_harm=[]
    for text in harmful_texts:
        hidden_states = get_vicuna_hidden_states(model, tokenizer, text)
        hidden_states_harm.append(hidden_states)
    hidden_states_harmless=[]
    for text in harmless_texts:
        hidden_states = get_vicuna_hidden_states(model, tokenizer, text)
        hidden_states_harmless.append(hidden_states)    
    
    # the mean activation difference of the last token between harmful and harmless prompts
    vectors = [] # safety steering vectors 
    for layer_index in range(model.config.num_hidden_layers):
        neg_hs = []
        for i in range(len(hidden_states_harm)):
            neg_hs.append(hidden_states_harm[i][layer_index+1][-1])
        neg_hs = torch.stack(neg_hs)
        pos_hs=[]
        for i in range(len(hidden_states_harmless)):
            pos_hs.append(hidden_states_harmless[i][layer_index+1][-1])
        pos_hs = torch.stack(pos_hs)
        steering_v = torch.mean(neg_hs,dim=0) - torch.mean(pos_hs,dim=0)
        steering_v = steering_v.to(model.device)
        steering_v /= torch.norm(steering_v)
        vectors.append(steering_v)
    
    return vectors


def get_unsafe_reference_transition_vector(model, tokenizer, unsafe_set):
    hidden_states_unsafe = []
    s1_index_unsafe = []
    for text in unsafe_set:
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
        text = system_prompt + ' USER: '+ text +' \nASSISTANT: Sure'
        # text = '<s>USER: '+ text +' \nASSISTANT: Sure'
        hidden_states, s1_index = get_vicuna_hidden_states_with_inst_pos(model, tokenizer, text, 'cuda:0')
        hidden_states_unsafe.append(hidden_states)
        s1_index_unsafe.append(s1_index)
    v_unsafe_reference = []
    for layer_index in np.arange(model.config.num_hidden_layers):
        s1_unsafe = []
        s2_unsafe = [] 
        for i in range(len(hidden_states_unsafe)):
            s1_unsafe.append(hidden_states_unsafe[i][layer_index+1][s1_index_unsafe[i]])
            s2_unsafe.append(hidden_states_unsafe[i][layer_index+1][-1])
        v_unsafe = []
        for i in range(len(s1_unsafe)):
            v_unsafe.append(s1_unsafe[i]-s2_unsafe[i])
        v_unsafe = torch.stack(v_unsafe)
        v_unsafe_reference.append(torch.mean(v_unsafe, dim=0))
    
    return v_unsafe_reference

def predict_safety(model, tokenizer, text, v_unsafe_reference, layer_start, layer_end = 32, threshold = 0.75):
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
    text = system_prompt + ' USER: '+ text +' \nASSISTANT: Sure'
    
    hidden_states, s1_index = get_vicuna_hidden_states_with_inst_pos(model, tokenizer, text, 'cuda:0')
    # print(s1_index)
    cosines = []
    for layer_index in np.arange(layer_start,layer_end):
        s1_safe = hidden_states[layer_index+1][s1_index]
        s2_safe = hidden_states[layer_index+1][-1]
        v_safe = s1_safe - s2_safe
        # print(layer_index)
        v_reference = v_unsafe_reference[layer_index]
        cosine = torch.nan_to_num(F.cosine_similarity(v_safe.unsqueeze(0), 
                                                      v_reference.unsqueeze(0)))
        cosines.append(cosine.item())
        
    cosine_mean = sum(cosines)/len(cosines)
    # 0: unsafe, 1: safe
    safety_label = 0 if cosine_mean >=threshold else 1
    
    return cosine_mean, safety_label

def greedy_predict_safety(model, tokenizer, text, v_unsafe_reference, layer_start, layer_end = 32, threshold = 0.6):
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
    text = system_prompt + ' USER: '+ text +' \nASSISTANT: Sure'
    # text = '<s>USER: '+ text +' \nASSISTANT: Sure'
    hidden_states, s1_index = get_vicuna_hidden_states_with_inst_pos(model, tokenizer, text, 'cuda:0')
    # 0: unsafe, 1: safe
    safety_label = 0
    
    cosines = []
    for layer_index in np.arange(layer_start,layer_end):
        s1_safe = hidden_states[layer_index+1][s1_index]
        s2_safe = hidden_states[layer_index+1][-1]
        v_safe = s1_safe - s2_safe
        # print(layer_index)
        v_reference = v_unsafe_reference[layer_index]
        cosine = torch.nan_to_num(F.cosine_similarity(v_safe.unsqueeze(0), 
                                                      v_reference.unsqueeze(0)))
        cosines.append(cosine.item())
        if cosine.item() < threshold:
            safety_label = 1
        
    cosine_mean = sum(cosines)/len(cosines)
    
    return cosine_mean, safety_label

  
def main(args):
    
    harmful_texts_anchor, harmless_texts_anchor, harmful_texts_test, harmless_texts_test  = prepare_safety_anchor_datasets(args.anchor_size)
    chat_model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    chat_model = chat_model.to(device)
    steering_v = get_safety_vectors(chat_model, tokenizer, harmful_texts_anchor, harmless_texts_anchor)
    v_unsafe_reference = get_unsafe_reference_transition_vector(chat_model, tokenizer, harmful_texts_anchor)
    model = LlamaWrapper(chat_model, tokenizer, size=args.model_size, use_vicuna=args.use_vicuna)
    
    if args.load_testdata != "default":
        if 'xstest' in args.load_testdata:
            harmful_texts_test, harmless_texts_test = load_xstest(args.load_testdata) # use xstest datasets as example
        elif 'MaliciousInstruct' in args.load_testdata:
            harmful_texts_test, harmless_texts_test = load_malicious_instruct_with_100_heldout_harmless(args.load_testdata)
        elif 'LLMSafeGuard' in args.load_testdata:
            harmful_texts_test, harmless_texts_test = load_llm_safeguard_200_heldout(args.load_testdata)
        elif 'OKTest' in args.load_testdata:
            harmful_texts_test, harmless_texts_test = load_oktest_with_HarmfulQ(args.load_testdata)
        elif 'representation-engineering' in args.load_testdata:
            harmful_texts_test, harmless_texts_test = load_RE_dataset(args.load_testdata)
    
    true_labels = [0]*len(harmful_texts_test)+[1]*len(harmless_texts_test)
    reject_num = [0,0]
    predicted_labels = []
    cos_all = []
    layer_start = args.layers[0]
    layer_end = chat_model.config.num_hidden_layers if args.model_size == '7b' else args.layers[-1]
    
    results = pd.DataFrame()
    inputs = []
    outputs = []
    for i, texts in enumerate([harmful_texts_test, harmless_texts_test]):    
        for text in texts:
            model.set_save_internal_decodings(False)
            model.reset_all()
            if 'LLMSafeGuard' in args.load_testdata or 'Held-out' in args.load_testdata:
                threshold = 0.68 if args.model_size == "7b" else 0.6
            elif 'OKTest' in args.load_testdata:
                threshold = 0.65 if args.model_size == "7b" else 0.62
            elif 'representation-engineering' in args.load_testdata and args.model_size=="13b":
                threshold = 0.65 
                layer_start = 37
                layer_end = 39
            elif 'default' in args.load_testdata and args.model_size=="13b":
                threshold = 0.7
                layer_start = 10
                layer_end = 20
            elif 'xstest' in args.load_testdata and args.model_size=="13b":
                threshold = 0.6
                layer_start = 23
                layer_end = 33
            else:
                threshold = 0.75 if args.model_size == "7b" else 0.70
            if 'xstest' in args.load_testdata and args.model_size == "7b":
                cosine_mean, safety_label = greedy_predict_safety(chat_model, tokenizer, text, v_unsafe_reference, layer_start=layer_start, layer_end=layer_end)
            else:
                cosine_mean, safety_label = predict_safety(chat_model, tokenizer, text, v_unsafe_reference, layer_start=layer_start, layer_end=layer_end, threshold=threshold)
            predicted_labels.append(safety_label)
            cos_all.append(1-cosine_mean)
            for layer in args.layers:
                steering_v[layer] = steering_v[layer].to(model.model.device)
                if safety_label:
                    model.set_steering_vectors(layer, -args.multiplier * steering_v[layer]) 
                else:
                    model.set_steering_vectors(layer, args.multiplier * steering_v[layer])
            response = model.generate_text(
                user_input = text,
                max_new_tokens = 256, # default: 50
            )
            response = response.split("ASSISTANT:")[-1].strip()
            inputs.append(text)
            outputs.append(response)
            if is_reject(response):
                reject_num[i] += 1.
            else:
                print(response)
    print('Refusal Rate of Unsafe Dataset:', reject_num[0]/len(harmful_texts_test))
    print('Refusal Rate of Safe Dataset:', reject_num[1]/len(harmless_texts_test))

    results["input"] = inputs
    results["output"] = outputs
    if args.load_testdata == "default":
        dataset_name = "advbench_truthfulQA"
    else:
        datasets = args.load_testdata.split(',')
        dataset_name = ''
        for i, data in enumerate(datasets):
            if i: dataset_name += '_'
            dataset_name += data.split('.')[-2].split('/')[-1]
    os.makedirs(f"{args.output_path}/{args.model_path.split('/')[-1]}", exist_ok=True)
    results.to_csv(f"{args.output_path}/{args.model_path.split('/')[-1]}/{dataset_name}.csv")
    
    precision, recall, thresholds = precision_recall_curve(true_labels, cos_all)
    auprc = auc(recall, precision)

    print(predicted_labels)
    print('Wrong Classification of Unsafe Dataset:', sum(predicted_labels[:len(harmful_texts_test)])/len(harmful_texts_test))
    print('Wrong Classification of Safe Dataset:', 1-sum(predicted_labels[-len(harmless_texts_test):])/len(harmless_texts_test))
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUPRC:", auprc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="vicuna-7b-v1.5",)
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--use_vicuna", type=bool, default=True)
    parser.add_argument("--multiplier", type=float, default=4)
    parser.add_argument("--output_path", type=str, default='./outputs')
    parser.add_argument("--layers", nargs='+', type=int, default=list(np.arange(10,20)), help="which layers to modify via safety-conscious activation steering")
    parser.add_argument("--anchor_size", type=int, default=64, help="the size of dataset used to get the safety steering vector and unsafe reference transition vector")
    parser.add_argument("--load_testdata", type=str, default='default', help="default: advbench(unsafe)&truthfulqa(safe); otherwise, load the test dataset path")
    args = parser.parse_args()
    
    main(args)
