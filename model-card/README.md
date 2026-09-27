---
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
language:
  - en
  - ko
pipeline_tag: text-classification
tags:
  - routing
  - coding-agents
  - claude-code
  - laya
  - mmbert
---

# pointsman-router

A small local classifier that decides which subagent tier should handle a coding task. It reads a task description and answers three questions with calibrated probabilities: what kind of work it is (intent), how much reasoning it needs (difficulty) and how risky it is (risk). It does not generate text.

It is the model behind [pointsman](https://github.com/jadeonstudio/pointsman), which uses it inside Claude Code to move a subagent to a lighter or stronger role (for example Haiku, Sonnet or Opus) before the subagent starts. The model was fine-tuned so that its answers agree with Claude Opus reference judgements, and it runs locally in about 0.1 seconds.

## What it predicts

The model answers the three route questions defined in pointsman `src/routing.mjs`, in one request:

| Question | Type | Answers |
|---|---|---|
| intent | choice | `explain`, `edit`, `debug`, `operate`, `research`, `architecture`, `other` |
| difficulty | score | 0–4 (displayed as 1–5: mechanical edit … deep, repository-wide reasoning) |
| risk | choice | `safe`, `caution`, `high`, `unknown` |

pointsman turns the three probability distributions into a probability for each tier (keep the host's choice, economy, standard, strong). It routes only when the chance that the host's own choice must be kept is at most 20% and the cheapest sufficient tier reaches 80% cumulative probability. These two values were fitted on held-out calibration data (see Evaluation) and ship with the model in `pointsman.json`.

## How to use

With pointsman (recommended). Pin a full commit id from this repository's history; branch names are refused.

```bash
pointsman laya pull --repo jadeonstudio/pointsman-router --revision <40-character commit id> --python <absolute path to a Python with laya==0.3.4>
pointsman laya adopt --candidate <checkpoint hash printed by pull>
pointsman on
```

`pull` checks every file against the sizes and SHA-256 hashes in `pointsman.json` and refuses the model if its fingerprint differs from the published checkpoint. `adopt` activates it with the published thresholds, and `pointsman laya rollback` undoes it. For a private copy of this repository, set `HF_TOKEN` in the environment; the token is never passed as an argument.

The files follow the official Laya 0.3.4 checkpoint layout (`model.safetensors`, `encoder/config.json`, `tokenizer/`, `rl_agent_config.json`), so the Laya runtime can also load the directory directly. The `training` block and `model_name` inside `rl_agent_config.json` are carried over from the upstream format and the training kit. They do not describe this fine-tune; see Training below.

## Training

- Base: `convaiinnovations/laya-multilingual` (mmBERT-base encoder, about 322M parameters). All parameters were trained.
- Data: 7,219 synthetic coding-task descriptions in Korean and English, written by Claude models under a written generation spec. The split is 6,619 tasks for training (19,857 question samples), 292 for calibration and 308 for test.
- Labels: every task was labelled by Claude Opus 5.5 (`claude-opus-5-5`) running as Claude Code subagents under a written labelling spec. No labels from any other model were used. The data is not released.
- Procedure: 6 epochs; the best epoch was selected by agreement on the calibration split (epoch 6). Micro-batch 4 with 16-step gradient accumulation (effective batch 64). AdamW with learning rate 2.5e-5 for the encoder and 1e-4 for the heads, weight decay 0.01, cosine schedule to 1e-6. Dropout 0.1, seed 42, long inputs truncated at the task head. Post-hoc temperature scaling was fitted on the calibration split.
- Hardware: Apple M4 Pro (MPS), about 12.8 hours.

## Evaluation

The test split has 308 tasks (159 Korean, 149 English) that were never used for training, epoch selection or threshold fitting. All numbers measure **agreement with Claude Opus reference labels**, not task success. For comparison, two independent Claude labelling passes agreed with each other at 0.977 (intent), 0.913 (difficulty) and 0.940 (risk) on a related 300-task set.

| Model | intent | difficulty | difficulty ±1 | risk | high→safe errors | warm latency p50 / p95 |
|---|---|---|---|---|---|---|
| **pointsman-router** | **0.880** | **0.718** | **0.994** | **0.692** | **2** | **101 / 167 ms** |
| laya-multilingual (base) | 0.308 | 0.292 | 0.727 | 0.451 | 43 | 113 / 193 ms |

Latency was measured on an Apple M4 Pro, MPS, fp16, with three questions per request and the model already loaded. Loading the model takes about 5.5 s.

By language, intent agreement is 0.918 for Korean and 0.839 for English, and risk agreement is 0.667 for Korean and 0.718 for English.

Routing decisions (the gate shipped in `pointsman.json`, tier probability 0.80 and keep-host probability 0.20), measured on the test split:

| | Tasks |
|---|---|
| Routed | 101 of 308 (32.8%) |
| Same tier Claude's labels imply | 72 |
| More expensive tier than Claude's labels imply | 15 |
| Unsafe (cheaper than Claude's tier, or routed where Claude would keep the host's choice) | 12 (11.9%, 95% upper bound 19.6%) |

Even Claude's own labels route only 50.6% of these tasks; the rest should keep the host's choice or go to the strongest tier.

## Limitations

- Agreement with Claude's judgement is not the same as a cheaper model succeeding. Measure outcomes in your own workflow before relying on the savings.
- Most unsafe decisions come from the risk head calling a task `safe` with high confidence when Claude labelled it `caution`. Thresholds cannot filter these out, so keep hard rules for production, credential and payment work outside the model.
- The training data is synthetic and covers Korean and English software-engineering requests. Other languages and domains are untested.
- The thresholds were fitted for this checkpoint only. Do not reuse them with another model.

## License and attribution

Apache License 2.0. This model is derived from Laya (Apache-2.0) and mmBERT-base (MIT); see `NOTICE`. The training labels are outputs of Anthropic's Claude.
