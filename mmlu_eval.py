import argparse
import os
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from eval.mmlu_categories import subcategories, categories
from transformers import AutoTokenizer,AutoModelForCausalLM
from utils.load_safety_dataset import prepare_safety_anchor_datasets
from utils.llama_wrapper import LlamaWrapper, find_instruction_end_postion
from ppl_eval import get_unsafe_reference_transition_vector, get_safety_vectors, predict_safety

choices = ["A", "B", "C", "D"]
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


# for llama2 model use
@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    
    harmful_texts_anchor, harmless_texts_anchor, _, _  = prepare_safety_anchor_datasets(args.anchor_size)
    steering_v = get_safety_vectors(model.model, tokenizer, harmful_texts_anchor, harmless_texts_anchor)
    layer_start = args.layers[0]
    layer_end = model.model.config.num_hidden_layers if args.model_size == '7b' else args.layers[-1]
    v_unsafe_reference = get_unsafe_reference_transition_vector(model.model, tokenizer, harmful_texts_anchor, use_vicuna=args.use_vicuna)
    predicted_labels = []
    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        # print(prompt)
        # input_text = "[INST] "+prompt+" [/INST]"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            # input_text = "[INST] "+prompt+" [/INST]"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]
        
        model.set_save_internal_decodings(False)
        model.reset_all()
        
        threshold = 0.75
        _, safety_label = predict_safety(model.model, tokenizer, prompt, v_unsafe_reference, layer_start=layer_start, layer_end=layer_end, threshold=threshold, use_vicuna=args.use_vicuna)
        predicted_labels.append(safety_label)

        for layer in args.layers:
            steering_v[layer] = steering_v[layer].to(model.model.device)
            if safety_label:
                model.set_steering_vectors(layer, -args.multiplier * steering_v[layer]) 
            else:
                model.set_steering_vectors(layer, args.multiplier * steering_v[layer])
        # ""[/INST]
        END_STR = torch.tensor(tokenizer.encode("\nAnswer:")[1:]).to(model.model.device)
        instr_pos = find_instruction_end_postion(input_ids[0], END_STR)
        model.set_after_positions(instr_pos) 
        
        logits = model.model(
            input_ids=input_ids,
        ).logits[:,-1].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    print(predicted_labels)
    return cors, acc, all_probs

@torch.no_grad()
def eval_baseline(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]
        
        logits = model(
            input_ids=input_ids,
        ).logits[:,-1].flatten()

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def cal_results(args):
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "mmlu_test"))
            if "_test.csv" in f
        ]
    )    
    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    
    args.model = args.model.split('/')[-1]
    if not args.test_baseline:
        args.model = args.model + '_steer'
    
    for subject in subjects:
        test_df = pd.read_csv(os.path.join(
            args.save_dir, "mmlu_results_{}".format(args.model), "{}.csv".format(subject)
        ))
        cors = test_df["{}_correct".format(args.model)]
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)
        
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    
def main(args):
    chat_model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if not args.test_baseline:
        model = LlamaWrapper(chat_model,tokenizer,size=args.model_size,use_chat=True, use_vicuna=args.use_vicuna)

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "mmlu_test"))
            if "_test.csv" in f
        ]
    )
    
    args.model = args.model.split('/')[-1]
    if not args.test_baseline:
        args.model = args.model + '_steer'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "mmlu_results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "mmlu_results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "mmlu_dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "mmlu_test", subject + "_test.csv"), header=None
        )
        if args.test_baseline:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            chat_model = chat_model.to(device)
            cors, acc, probs = eval_baseline(args, subject, chat_model, tokenizer, dev_df, test_df)
        else:
            cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "mmlu_results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=8)
    parser.add_argument("--data_dir", "-d", type=str, default="datasets")
    parser.add_argument("--save_dir", "-s", type=str, default="outputs")
    parser.add_argument("--model", "-m", type=str,  default="Llama-2-7b-chat-hf")
    parser.add_argument("--model_size", "-ms", type=str,default="7b")
    parser.add_argument("--multiplier", type=float, default=4)
    parser.add_argument("--layers", nargs='+', type=int, default=list(np.arange(10,20)), help="which layers to modify via safety-conscious activation steering")
    parser.add_argument("--anchor_size", type=int, default=64, help="the size of dataset used to get the safety steering vector and unsafe reference transition vector")
    parser.add_argument("--use_vicuna", type=bool, default=False)
    parser.add_argument("--test_baseline", "-b", type=bool, default=False)
    
    args = parser.parse_args()
    main(args)
    # cal_results(args)