# SCANS
Official repository for our AAAI 2025 paper ["SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering"](https://arxiv.org/abs/2408.11491).

## Overview
Motivated by the intuition of representation engineering to steer model behavior, the key idea behind our SCANS is to extract the refusal behavior vectors, and anchor the safety-critical layers for steering. SCANS then evaluates the harmfulness of inputs to guide output distribution against or consistent with the refusal behavior, which achieves a balance between adequate safety and exaggerated safety.
![overview](https://github.com/user-attachments/assets/1a97cfe2-db17-4136-bf65-dd9b3a7b5622)

## Recommended software environment

> conda env create -n scans_env -f scans_env.yml
  
- python == 3.10.13
- torch == 2.2.0
- transformers >= 4.36.2
- scikit-learn >= 1.2.1
- numpy >= 1.25.2

## Description
- The implementations of SCANS on Llama2 and vicuna can be referred to ```SCANS_llama.py``` and ```SCANS_vicuna.py```, respectively.

  #### Parameters:

  ```--model_path```: To prevent instability in remote access, our code uses local model loading. You need to download the model you need to deploy (e.g., Llama2-7b-chat) into the ``model_path`` folder and specify this parameter when running the code.

  ```--model_size```: Our experiments are primarily based on Llama2-7b-chat, Llama2-13b-chat, vicuna-7b-v1.5 and vicuna-13b-v1.5 and we have tuned some hyper-parameters for each model. Thus, you can set this parameter to tell which model is used, like ```model_size="7b"``` and ```model_size="13b"``` in ``SCANS_llama.py``` ref to Llama2-7b-chat and Llama2-13b-chat model, respectively.

  ```--use_chat```: The default value is ```True``` since our work focuses on the mitigation of exaggerated safety issue which is common in aligned LLMs.

  ```--multiplier```: Ref to hyperparameter α that controls the strength of steering. For exmple, the steering vector multiplier α for Llama2 family models is all set to ```3.5```.

  ```--layers```: Decide which layers to modify via safety-conscious activation steering. The default value is ```list(np.arange(10,20))```.

  ```--anchor_size```: The size of dataset used to get the safety steering vector and unsafe reference transition vector. The default value is ```64```.

  ```--load_testdata```: The default value is ```"default"``` which represents two datasets: advbench(unsafe)&truthfulqa(safe); otherwise, load the test dataset path.

  ```--output_path```: Save output (input prompts, LLM outputs after SCANS, etc.) of each sample to ```"./outputs"```(the default saving path, you can modify as needed).

- ```MATCH_STRINGS``` in ```utils/modeling_utils.py``` list some example refusal string keywords. We adopt string matching to judge whether the model response refuses the query because we find that after activation steering, models may use some more fixed phrases to refuse that can be well covered by a manually defined string set. You can also modify the given ```MATCH_STRINGS``` according to your model outputs to guarantee the accuracy of judgement results.

- ```utils/load_safety_dataset.py``` provides the loading methods of some safety-related datasets in our paper. When you want to test other new safety-related datasets, you might need to add new dataset loading function here.

- ```utils/llama_wrapper.py``` is inspired by this [work](https://github.com/nrimsky/CAA) (Thanks!). When you want to test other models (except Llama2, vicuna), you might need to modify this file accordingly (e.g., different chat templates).

- The used datasets in our paper is provided in  ```datasets``` folder.

  We use [AdvBench](https://github.com/ltroin/llm_attack_defense_arena) as the harmful queries and [TruthfulQA](https://github.com/sylinrl/TruthfulQA) as the benign ones to generate the refusal steering vector.
  
  We select [XSTest](https://github.com/paul-rottger/exaggerated-safety) and [OKTest](https://github.com/InvokerStark/OverKill) which are two prominent benchmarks focusing on the exaggerated safety phenomenon in LLMs.

  We use [RepE-Data](https://huggingface.co/datasets/justinphan3110/harmful_harmless_instructions), the remaining AdvBench, [MaliciousInstruct](https://github.com/Princeton-SysML/Jailbreak_LLM) to evaluate the security.

- We also evaluate whether the SCANS would result in a model capability decline. (a) multi-choice question answer-
ing task ```mmlu_eval.py```: we choose MMLU (Hendrycks et al. 2020) since it is considered comprehensive and challenging due to the extensive knowledge needed. (b) generation task ```xsum_eval.py```: taking text summaries as an example, we use XSum (Narayan, Cohen, and Lapata 2018) to evaluate the quality of generated summaries when employing activation steering. Besides, we include two perplexity-based tasks ```ppl_eval.py```, WikiText-2 (Merity et al. 2017) and C4 (Raffel et al. 2020).

- We additionally provide a ```classification_eval.py``` file for comparing the our classification method σ(q) with some state-of-the-art baselines (Llama Guard, Perspective API, etc.).

## Usage 
The below script is one example of using our SCANS on Llama2-7b-chat model. 

```sh
# for llama2_7b_chat
python SCANS_llama.py

python SCANS_llama.py \
    --load_testdata datasets/xstest_v2_prompts.csv

python SCANS_llama.py \
    --load_testdata datasets/OKTest.csv,datasets/HarmfulQ.json

python SCANS_llama.py \
    --load_testdata datasets/representation-engineering/data/test-00000-of-00001-e88521c3da183185.parquet

python SCANS_llama.py \
    --load_testdata datasets/MaliciousInstruct.txt,datasets/Held-outHarmless.txt
```
TODO: add more hyper-parameters to help reproduction.

## Citation
If you use our technique or are inspired by our work, welcome to cite our paper and provide valuable suggestions.
```
@inproceedings{cao2025scans,
  title={SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering},
  author={Cao, Zouying and Yang, Yifei and Zhao, Hai},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={22},
  pages={23523--23531},
  year={2025}
}
```

> [!NOTE]  
> This repo is under construction.
