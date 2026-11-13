---
license: apache-2.0
task_categories:
- text-generation
tags:
- agent
- llm
- benchmark
---

# Long Horizon Execution

This project contains the dataset accompanying the paper "[The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](https://arxiv.org/abs/2509.09677)"

## Abstract
Does continued scaling of large language models (LLMs) yield diminishing returns? Real-world value often stems from the length of task an agent can complete. We start this work by observing the simple but counterintuitive fact that marginal gains in single-step accuracy can compound into exponential improvements in the length of a task a model can successfully complete. Then, we argue that failures of LLMs when simple tasks are made longer arise from mistakes in execution, rather than an inability to reason. We propose isolating execution capability, by explicitly providing the knowledge and plan needed to solve a long-horizon task. We find that larger models can correctly execute significantly more turns even when small models have 100% single-turn accuracy. We observe that the per-step accuracy of models degrades as the number of steps increases. This is not just due to long-context limitations -- curiously, we observe a self-conditioning effect -- models become more likely to make mistakes when the context contains their errors from prior turns. Self-conditioning does not reduce by just scaling the model size. In contrast, recent thinking models do not self-condition, and can also execute much longer tasks in a single turn. We conclude by benchmarking frontier thinking models on the length of task they can execute in a single turn. Overall, by focusing on the ability to execute, we hope to reconcile debates on how LLMs can solve complex reasoning problems yet fail at simple tasks when made longer, and highlight the massive benefits of scaling model size and sequential test-time compute for long-horizon tasks.

**GitHub:** [https://github.com/long-horizon-execution/measuring-execution/](https://github.com/long-horizon-execution/measuring-execution/)

## Description
![Our task.](task_overview.png)

This dataset is a synthetic benchmark designed to measure the pure execution capability of LLMs over long horizons. The core task is **key-value dictionary addition**. A fixed, in-context dictionary mapping five-letter English words (keys) to integer values is provided in `dictionary.json`. The model's goal is to maintain a running sum. In each turn, it receives one or more keys (defined by the turn complexity, `K`), retrieves their corresponding values from the dictionary, adds them to the running sum, and outputs the new sum. The primary metric for evaluation is the **task length**: the number of steps a model can execute before its accuracy drops below a certain threshold.

The dataset is designed to be programmatically generated and thus contamination-free. We only provide 100 samples here for ease of access, but more can be generated using the script [here](https://github.com/long-horizon-execution/measuring-execution/blob/main/generate_dataset_json.py).

## Using the dataset
`test.jsonl` contains the individual samples that can be used to prompt the LLM.
- _"input"_: contains the keys to be processed.
- _"values"_: contains the values mapped to the corresponding keys as described in `dictionary.json`.
- _"output"_: contains the expected running sum answers.

The provided dataset is configured with a turn complexity of `K=1` (one key per turn). To evaluate models on a higher turn complexity, such as `K=N`, you can post-process the data by grouping every `N` consecutive turns:
- _"input"_: Concatenate every `N` items into a single comma-separated string.
- _"output"_: The new running sum for the grouped turn is simply the **last** running sum from the original group of `N` turns.

## Sample Usage

To load and inspect the `test.jsonl` dataset:

```python
import json

file_path = "test.jsonl"
samples = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples.")
print("First sample:")
print(samples[0])
```

## Benchmark
![Benchmark of Frontier models.](benchmark.png)

## Citation
Find out more about our work at https://arxiv.org/abs/2509.09677. Consider citing us if you use our dataset.
```
@misc{
      sinha2025illusiondiminishingreturnsmeasuring,
      title={The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs}, 
      author={Akshit Sinha and Arvindh Arun and Shashwat Goel and Steffen Staab and Jonas Geiping},
      year={2025},
      eprint={2509.09677},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.09677}, 
}
```