The used datasets in our paper is provided in  ```datasets``` folder.

- We use [AdvBench](https://github.com/ltroin/llm_attack_defense_arena) as the harmful queries and [TruthfulQA](https://github.com/sylinrl/TruthfulQA) as the benign ones to generate the refusal steering vector. Note that we just randomly sample 64 harmful questions and 64 harmless questions to extract the steering vectors as mentioned in Section 3.1.
- [XSTest](https://github.com/paul-rottger/exaggerated-safety) and [OKTest](https://github.com/InvokerStark/OverKill) are two prominent benchmarks focusing on the exaggerated safety phenomenon in LLMs. The remaining TruthfulQA is also utilized for safety evaluation.
- [RepE-Data](https://huggingface.co/datasets/justinphan3110/harmful_harmless_instructions), the remaining AdvBench, [MaliciousInstruct](https://github.com/Princeton-SysML/Jailbreak_LLM) are used to evaluate the security.
- [MMLU](https://github.com/hendrycks/test) is to evaluate whether the SCANS would result in a model capability decline.
