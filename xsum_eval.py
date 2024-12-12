import torch
import random
import argparse
import numpy as np
import pandas as pd
from rouge import Rouge
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.llama_wrapper import LlamaWrapper
from utils.load_safety_dataset import prepare_safety_anchor_datasets
from ppl_eval import get_safety_vectors, get_unsafe_reference_transition_vector,predict_safety


def main(args):
    chat_model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if not args.test_baseline:
        model = LlamaWrapper(chat_model,tokenizer,size=args.model_size,use_chat=args.use_chat, use_vicuna=args.use_vicuna)
        harmful_texts_anchor, harmless_texts_anchor, _, _  = prepare_safety_anchor_datasets(args.anchor_size)
        steering_v = get_safety_vectors(model.model, tokenizer, harmful_texts_anchor, harmless_texts_anchor)
        layer_start = args.layers[0]
        layer_end = model.model.config.num_hidden_layers if args.model_size == '7b' else args.layers[-1]
        v_unsafe_reference = get_unsafe_reference_transition_vector(model.model, tokenizer, harmful_texts_anchor, use_vicuna=args.use_vicuna)

    path='Xsum/dev.csv'
    xsum=pd.read_csv(path).sample(200)

    xsum_texts = xsum['dialogue'].tolist()
    summary = xsum['summary'].tolist()
    
    rouge = Rouge()
    rouge1_score_list = []
    rouge2_score_list = []
    rougel_score_list = []
    predicted_labels = []
    for i, text in enumerate(xsum_texts):
        input_text = f'Document:{text.strip()}\nBased on the previous text, provide a brief single summary:'
        
        if args.test_baseline:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            chat_model = chat_model.to(device)
            if args.use_vicuna:
                input_text = 'USER: '+ input_text +' \nASSISTANT:'
            else:
                input_text = '[INST] '+ input_text +' [/INST]'
            input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(chat_model.device)
            response = tokenizer.batch_decode(chat_model.generate(
                inputs = input_ids, 
                max_new_tokens = 128, 
                top_k=1, 
                repetition_penalty=1.1,
            ))[0]
        else:    
            model.set_save_internal_decodings(False)
            model.reset_all()
            threshold = 0.75
            _, safety_label = predict_safety(model.model, tokenizer, input_text, v_unsafe_reference, layer_start=layer_start, layer_end=layer_end, threshold=threshold, use_vicuna=args.use_vicuna)
            predicted_labels.append(safety_label)

            for layer in args.layers:
                steering_v[layer] = steering_v[layer].to(model.model.device)
                if safety_label:
                    model.set_steering_vectors(layer, -args.multiplier * steering_v[layer]) 
                else:
                    model.set_steering_vectors(layer, args.multiplier * steering_v[layer])
            response = model.generate_text(
                user_input = input_text,
                max_new_tokens = 128, 
            )
        if args.use_vicuna:
            response = response.split("ASSISTANT:")[-1].strip()
        else:
            response = response.split("[/INST]")[-1].strip()
        
        scores = rouge.get_scores(response, summary[i])[0]

        rouge1_score_list.append(scores['rouge-1']['f'])
        rouge2_score_list.append(scores['rouge-2']['f'])
        rougel_score_list.append(scores['rouge-l']['f'])
        
        print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))
    
    print("FINAL RESULTS")
    print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path","-m",type=str,default="Llama-2-7b-chat-hf")
    parser.add_argument("--model_size", "-ms", type=str, default="7b")
    parser.add_argument("--use_chat", "-c", type=bool, default=True)
    parser.add_argument("--use_vicuna", type=bool, default=False)
    parser.add_argument("--multiplier", type=float, default=3.5)
    parser.add_argument("--layers", nargs='+', type=int, default=list(np.arange(10,20)), help="which layers to modify via safety-conscious activation steering")
    parser.add_argument("--anchor_size", type=int, default=64, help="the size of dataset used to get the safety steering vector and unsafe reference transition vector")
    parser.add_argument("--test_baseline", "-b", type=bool, default=False)
    
    args = parser.parse_args()
    main(args)
    