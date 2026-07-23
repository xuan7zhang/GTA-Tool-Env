# Can answer confidence (entropy / log-likelihood) guide tool-space optimization?

**Question.** Behavioral probes need a live tool server. Is there a purely *model-side*,
label-free signal — the model's own confidence in its final answer — that ranks tool
environments by quality (so we could pick the config that maximizes it)? We swept 15
entropy/likelihood variants at two levels (token and answer) against a clean
three-config accuracy gradient.

**Answer: no.** No variant reliably ranks tool configs, and per-task correctness
prediction is weak at best (AUC ≈ 0.60). The negative result is informative — it
explains why a reliability signal must probe the *tool*, not the model's confidence.

## Setup

Qwen2.5-7B, GTA-Atomic full 229. Three configs spanning a clear accuracy gradient
(all clean-pool except corrupt):

| config | menu | tool outputs | true AnsAcc (greedy) |
|---|---|---|---|
| oracle | per-task GT tools | real | **22.0** |
| corrupt | per-task GT tools | proxy-corrupted | 13.5 |
| full | all-14 forced (clutter) | real | 10.9 |

Two judges per signal: **per-task AUC** (does the signal predict this answer's
correctness?) and **config-level rank-match** (does the signal's config-mean rank the
three configs like true accuracy?). Token-level signals come from per-token logprobs +
top-20 entropy captured during greedy runs (`GTA_LL_LOG` patch in `models/openai_api.py`);
answer-level semantic signals from K=5 temperature-0.8 samples per config.

## Results — every signal fails

**Token level (greedy).** mean/min/sum logprob, perplexity, entity-token min-logprob,
low-confidence-token fraction, per-token entropy (mean/max/min), top1–top2 margin
(mean/min): all per-task |AUC−0.5| ≤ 0.056, and config-means are indistinguishable
(e.g. mean-logprob −0.012 / −0.009 / −0.010 for oracle/full/corrupt — all ≈ −0.01).
Greedy decoding flattens the token distribution; the argmax token is ~100% confident
whether the answer is right or wrong.

**Answer level (semantic, K=5 @ T=0.8).** Naive string-clustered entropy/agreement:
n_distinct ≈ 4.7/5 for *every* config (free-form answers are almost all string-distinct),
so no signal. Proper **semantic entropy** (mpnet embedding clustering, cos ≥ 0.75):

| | oracle (22.0) | corrupt (13.5) | full (10.9) |
|---|---|---|---|
| semantic entropy (mean) | 0.869 | 0.865 | 0.829 |

- **Config-level: inverted.** The *best* config (oracle) has the *highest* entropy, the
  worst (full) the lowest → rank ✗. Semantic entropy tracks answer *diversity* (a good
  config gives richer tool evidence → more varied phrasings), not config *quality*; the
  between-config entropy is dominated by task difficulty / tool richness, not by
  confidence in correctness.
- **Per-task: weak.** semantic entropy → correctness AUC 0.403 (i.e. low entropy → more
  likely correct, AUC 0.597 in that direction); agreement → correctness 0.595. So
  within a config, answer consistency weakly predicts correctness — but it is the
  strongest signal found and still only ≈ 0.60.

**mean-logprob config ranking is unstable.** In one earlier 3-config run it happened to
rank oracle > full; here it ranks inverted. The config-means all sit at ≈ −0.01, so the
ordering is noise — the apparent "it works" was luck at n=3.

## Why this matters

The model is **confidently wrong** under a bad tool environment: its internal
uncertainty reflects "how familiar is this answer string," not "how trustworthy was the
tool evidence." Confidence/entropy/likelihood therefore **cannot** compare tool
environments. This is consistent with the project's central split:

- **Tool reliability** is a property *of the tool* (does output depend on input) →
  measurable by a behavioral probe, label-free, regime-invariant. This is what our
  method uses.
- **Environment quality / task-helpfulness** is a property of *tool × task × model* →
  not recoverable from the model's own answer confidence.

So a model-side confidence signal is not a shortcut around the behavioral probe; the
probe's information (tool input-dependence) is genuinely not present in the model's
answer distribution.

## Caveats / open

Single model (7B), single dataset, greedy + T=0.8 only. A stronger elicitation
(verbalized confidence, a trained probe on hidden states, or NLI-based semantic-entropy
clustering) is untested and could recover a weak per-task signal, but the config-level
inversion — driven by answer diversity, not quality — is unlikely to reverse. Signals,
per-task rows and per-config means: `results/ll_signal/`, `results/{ll7b,sc}_*`.
Scripts: `experiments/{ll_battery.sh, ll_analyze_full.py, semantic_entropy.py}` and the
`GTA_LL_LOG` capture patch.
