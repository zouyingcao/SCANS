The used datasets in our paper is provided in  ```datasets``` folder.

- We use [AdvBench](https://github.com/ltroin/llm_attack_defense_arena) as the harmful queries and [TruthfulQA](https://github.com/sylinrl/TruthfulQA) as the benign ones to generate the refusal steering vector.
- [XSTest](https://github.com/paul-rottger/exaggerated-safety) and [OKTest](https://github.com/InvokerStark/OverKill) are two prominent benchmarks focusing on the exaggerated safety phenomenon in LLMs.
- [RepE-Data](https://huggingface.co/datasets/justinphan3110/harmful_harmless_instructions), the remaining AdvBench, [MaliciousInstruct](https://github.com/Princeton-SysML/Jailbreak_LLM) are used to evaluate the security.
- [MMLU](https://github.com/hendrycks/test) is to evaluate whether the SCANS would result in a model capability decline.
