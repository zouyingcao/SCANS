# SCANS: Mitigating the Exaggerated Safety for LLMs via Safety-Conscious Activation Steering
![overview](https://github.com/user-attachments/assets/1a97cfe2-db17-4136-bf65-dd9b3a7b5622)

## Recommended software environment

> conda env create -n scans_env -f scans_env.yml
  
- python == 3.10.13
- torch == 2.2.0
- transformers >= 4.36.2
- scikit-learn >= 1.2.1
- numpy >= 1.25.2

## Description
The implementations of SCANS on Llama2 and vicuna can be referred to ```SCANS_llama.py``` and ```SCANS_vicuna.py```, respectively.

### Parameters:

```--model_path```: To prevent instability in remote access, our code uses local model loading. You need to download the model you need to deploy (e.g., Llama2-7b-chat) into the ``model_path`` folder and specify this parameter when running the code.

```--model_size```: Our experiments are primarily based on Llama2-7b-chat, Llama2-13b-chat, vicuna-7b-v1.5 and vicuna-13b-v1.5 and we have tuned some hyper-parameters for each model. Thus, you can set this parameter to tell which model is used, like ```model_size="7b"``` and ```model_size="13b"``` in ``SCANS_llama.py``` ref to Llama2-7b-chat and Llama2-13b-chat model, respectively.

```--use_chat```: The default value is ```True``` since our work focuses on the mitigation of exaggerated safety issue which is common in aligned LLMs.

```--multiplier```:  Refs to hyperparameter α that controls the strength of steering. For exmple, the steering vector multiplier α for Llama2 family models is all set to ```3.5```.

```--layers```: Decide which layers to modify via safety-conscious activation steering. The default value is ```list(np.arange(10,20))```.

```--anchor_size```: The size of dataset used to get the safety steering vector and unsafe reference transition vector. The default value is ```64```.

```--load_testdata```: The default value is ```"default"``` which represents two datasets: advbench(unsafe)&truthfulqa(safe); otherwise, load the test dataset path.

```--output_path```: The default value is ```"./outputs"```.

## Citation
If you use our technique or are inspired by our work, welcome to cite our paper and provide valuable suggestions.
```
```

> [!NOTE]  
> This repo is under construction.
