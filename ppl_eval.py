import os
import torch
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.llama_wrapper import LlamaWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.modeling_utils import get_all_hidden_states, get_all_hidden_states_with_inst_pos,tokenize_llama_chat, tokenize_llama_base, tokenize_vicuna_v1_5, find_instruction_end_postion
from utils.load_safety_dataset import prepare_safety_anchor_datasets
from eval.ppl_dataset import get_ppl_datasets, get_loaders

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def get_safety_vectors( model, tokenizer, harmful_texts, harmless_texts):

    hidden_states_harm=[]
    for text in harmful_texts:
        hidden_states = get_all_hidden_states(model, tokenizer, text)
        hidden_states_harm.append(hidden_states)
    hidden_states_harmless=[]
    for text in harmless_texts:
        hidden_states = get_all_hidden_states(model, tokenizer, text)
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


def get_unsafe_reference_transition_vector(model, tokenizer, unsafe_set, use_vicuna=False):
    hidden_states_unsafe = []
    s1_index_unsafe = []
    for text in unsafe_set:
        if use_vicuna:
            text = '<s>USER: '+ text +' \nASSISTANT: Sure'
        else:
            text = '<s>[INST] '+ text +' [/INST] Sure'
        hidden_states, s1_index = get_all_hidden_states_with_inst_pos(model, tokenizer, text, 'cuda:0')
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

def predict_safety(model, tokenizer, text, v_unsafe_reference, layer_start, layer_end = 32, threshold = 0.75, use_vicuna=False):
    if use_vicuna:
        text = '<s>USER: '+ text +' \nASSISTANT: Sure'
    else:
        text = '<s>[INST] '+ text +' [/INST] Sure'
    hidden_states, s1_index = get_all_hidden_states_with_inst_pos(model, tokenizer, text, 'cuda:0')

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

    
def main_ori(args):
    
    harmful_texts_anchor, harmless_texts_anchor, _, _  = prepare_safety_anchor_datasets(args.anchor_size)
    chat_model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    chat_model = chat_model.to(device)
    
    steering_v = get_safety_vectors(chat_model, tokenizer, harmful_texts_anchor, harmless_texts_anchor)
    layer_end = chat_model.config.num_hidden_layers if args.model_size == '7b' else args.layers[-1]
    v_unsafe_reference = get_unsafe_reference_transition_vector(chat_model, tokenizer, harmful_texts_anchor, use_vicuna=args.use_vicuna)
    
    model = LlamaWrapper(chat_model, tokenizer, size=args.model_size, use_chat=args.use_chat, use_vicuna=args.use_vicuna)


    predicted_labels = []
    cos_all = []    
    layer_start = args.layers[0]

    _, test_dataloader = get_loaders(args.ppl_data, tokenizer, seq_len=2048, batch_size = 1, use_chat=args.use_chat, use_vicuna=args.use_vicuna)
    with torch.no_grad():
        nlls = []
        for batch in tqdm(test_dataloader):  
            batch = batch.to(device)     
            text = tokenizer.decode(batch[0]) 
            
            model.set_save_internal_decodings(False)
            model.reset_all()
            threshold = 0.75
            cosine_mean, safety_label = predict_safety(chat_model, tokenizer, text, v_unsafe_reference, layer_start=layer_start, layer_end=layer_end, threshold=threshold, use_vicuna=args.use_vicuna)
            predicted_labels.append(safety_label)
            cos_all.append(1-cosine_mean)
            for layer in args.layers:
                steering_v[layer] = steering_v[layer].to(model.model.device)
                if safety_label:
                    model.set_steering_vectors(layer, -args.multiplier * steering_v[layer]) 
                else:
                    model.set_steering_vectors(layer, args.multiplier * steering_v[layer])
            
            instr_pos = find_instruction_end_postion(batch[0], model.END_STR)
            model.set_after_positions(instr_pos) # instr_p-100: suppose the last 100 tokens is answer

            target_ids = batch.clone()
            # target_ids[:, :-(instr_pos+len(model.END_STR))] = -100 # suppose the tokens after "[/INST]/ASSISTANT:" is answer
            
            outputs = model.model(batch, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
            # print(neg_log_likelihood)
            nlls.append(neg_log_likelihood.detach().cpu())
            
        #print(torch.cat(nlls, dim=-1).mean())
        # ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppl = np.exp(np.mean(nlls).item())

    print(predicted_labels)
    print('Wrong Classification of Safe Dataset:', 1-sum(predicted_labels)/len(predicted_labels))
    print('PPL:',ppl)

def baseline(args):
    chat_model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    chat_model = chat_model.to(device)
    _, test_data = get_ppl_datasets(args.ppl_data)
    
    with torch.no_grad():
        nlls = []
        for text in test_data:        
            if args.use_chat:
                if args.use_vicuna:
                    tokens = tokenize_vicuna_v1_5(tokenizer=tokenizer, user_input=text)
                else:
                    tokens = tokenize_llama_chat(tokenizer=tokenizer, user_input=text)
            else:
                tokens = tokenize_llama_base(tokenizer=tokenizer, user_input=text)
            tokens = torch.tensor(tokens).unsqueeze(0).to(device)
            lm_logits = chat_model(tokens).logits
            # print(lm_logits.shape)
            # print(tokens.shape)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss)
        #print(torch.cat(nlls, dim=-1).mean())
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())

    print('PPL:',ppl)
    
def baseline_ori(args):
    chat_model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    chat_model = chat_model.to(device)
    _, test_dataloader = get_loaders(args.ppl_data, tokenizer, seq_len=2048, batch_size = 1, use_chat=args.use_chat, use_vicuna=args.use_vicuna)
    
    with torch.no_grad():
        nlls = []
        for batch in tqdm(test_dataloader):        
            batch = batch.to(device)
            target_ids = batch.clone()
            # if args.use_chat:
            #     if args.use_vicuna: # \nASSISTANT:
            #         END_STR = torch.tensor(tokenizer.encode("\nASSISTANT:")[1:]).to(device)
            #     else: # [/INST]
            #         END_STR = torch.tensor(tokenizer.encode("[/INST]")[1:]).to(device)
            # instr_pos = find_instruction_end_postion(batch[0], END_STR)
            # target_ids[:, :-(instr_pos+len(END_STR))] = -100 # suppose the tokens after "[/INST]/ASSISTANT:" is answer
            
            outputs = chat_model(batch, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
            # print(neg_log_likelihood)
            nlls.append(neg_log_likelihood.detach().cpu())
   
        ppl = np.exp(np.mean(nlls).item())

    print('PPL:',ppl)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Llama-2-7b-chat-hf",)
    parser.add_argument("--model_size", type=str, default="7b")
    parser.add_argument("--use_chat", type=bool, default=True)
    parser.add_argument("--use_vicuna", type=bool, default=False)
    parser.add_argument("--multiplier", type=float, default=3.5)
    parser.add_argument("--layers", nargs='+', type=int, default=list(np.arange(10,20)), help="which layers to modify via safety-conscious activation steering")
    parser.add_argument("--anchor_size", type=int, default=64, help="the size of dataset used to get the safety steering vector and unsafe reference transition vector")
    parser.add_argument("--ppl_data", type=str, default='wikitext2', choices=('wikitext2','ptb','c4'))
    parser.add_argument("--test_baseline", "-b", type=bool, default=False)
    
    args = parser.parse_args()
    
    # main(args)
    # baseline(args)
    if args.test_baseline:
        baseline_ori(args)
    else:
        main_ori(args)