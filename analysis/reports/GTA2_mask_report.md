# Optimizing the Tool Environment of a Multimodal Agent — Mask-Axis Report

## Overview

We study an agent's **tool environment** as a first-class, optimizable object
**E = (m, C, Φ)**, independent of the model:

- **m** — the tool **mask** (which tools are available)
- **C** — **composition** (fusing tools into macro-tools)
- **Φ** — tool **schema / description** rewrites

Motivation: several recent papers argue that raw agent accuracy hides what tools
actually contribute (Tool-Use Tax, VisualNeedle, "Do Multimodal Agents Really Benefit
from Tool Use"). We put a controllable probe-proxy in front of the tool server so we
can intervene on E at run time and read out the causal effect on accuracy.

This report covers the **mask axis (m)**: whether per-task masking helps, why, and
whether it can be done **without ground truth**.

---

## 0. Testbed — the GTA-Atomic benchmark

**GTA** (General Tool Agents, NeurIPS 2024), Atomic setting: **229 real multimodal
tasks**. Each task is a natural-language request plus 1–2 real images (**249 unique
images**, avg 1.1 per task) that the agent must answer by **calling tools** (e.g.
"how much should I pay for the beer according to the menu?", "what breed is the dog in
the middle of the picture?").

### Tool pool: 14 tools in 4 categories

| Category | Tools |
|---|---|
| **Perception** (4) | OCR, ImageDescription, RegionAttributeDescription, TextToBbox |
| **Logic** (5) | Calculator, Solver, Plot, MathOCR, CountGivenObject |
| **Operation** (3) | DrawBox, AddText, GoogleSearch |
| **Creativity** (2) | TextToImage, ImageStylization |

Tools are served by an AgentLego tool server backed by real models (GLIP detection,
LLaVA VQA, EasyOCR, etc.) running on GPU.

### Ground truth each task carries

Every task is annotated with four kinds of GT:

1. **Relevant tool set** (`tools`): which tools the task needs — **1–4 tools, avg
   2.3 of 14** (distribution: 1 tool → 17 tasks, 2 → 147, 3 → 51, 4 → 14). This is the
   GT for our **oracle mask** and for **tool-selection accuracy**.
2. **Reference trajectory** (`dialogs`): a full GT solution chain with the **exact tool
   calls and arguments** at each step (**avg 2.4 tool-call steps per task**, range 1–7).
   This is the GT for step-level metrics.
3. **Gold answer** (`gt_answer`): **156 objective** (whitelist aliases, word-boundary
   regex match), **16 subjective** (semantic similarity ≥ threshold, mpnet), **57
   generative** (image-output tasks scored on generation-tool arguments).

### Native metrics — the benchmark provides tool-selection GT

Because GTA ships step-aligned reference trajectories, **tool-selection accuracy is a
benchmark-native metric**. Two evaluation modes:

- **Step mode** (aligned to the GT trajectory): **InstAcc** (instruction alignment),
  **ToolAcc = tool-selection accuracy** (did the agent pick the same tool as GT at
  each step), **ArgAcc** (argument accuracy), and per-category P/R/F1. Our reproduced
  7B **ToolAcc = 33.4** (public leaderboard 32.85) — i.e. the 7B strictly selects the
  correct tool at only ~⅓ of steps.
- **End mode** (end-to-end): **AnsAcc** — final-answer accuracy (objective whitelist /
  subjective simscore). This is the main metric for our mask and composition work.

### Why this matters for us

Because the benchmark provides both *relevance* GT (which tools a task needs) and
*trajectory* GT (which tool to call at each step), we can (a) build an **oracle mask**
and measure **selection accuracy**, (b) **score a no-GT selector** against that GT
while the selector itself never touches it, and (c) separate "picked the right tool"
(ToolAcc / selection) from "answered correctly" (AnsAcc). The mask axis' causal chain
is precisely: *selection accuracy ↑ → answer accuracy ↑*.

> **Note on noise.** A ±2.5–3 AnsAcc run-to-run noise floor (tool-server
> nondeterminism even under greedy decoding) applies throughout. Every accuracy claim
> is multi-seed; the load-bearing evidence is the deterministic mechanism metrics
> (selection accuracy, coverage/all-kept), not the noisy AnsAcc means.

---

## 1. What "mask" means, mechanically

Each task's agent menu is the set of tools it may call (`action_executor.actions`);
masking = which tools are in that set. It is composed as

```
menu = (per-task resources ∩ deployed tools) + force-added − hidden
       [optionally restricted per task by a predicted keep-set]
```

Since GTA annotates each task with its 1–4 relevant tools (avg 2.3), **`per_task` = the
oracle mask** and **`full_everywhere` (force all 14 onto every task) = mask OFF**. All
of this is pure environment-side intervention: no change to tools, data, or model.

## 2. The oracle mask helps — and it is a capability effect

`per_task` (oracle) vs `full_everywhere` (mask off), full 229, AnsAcc:

| Model | Full (mask off) | Oracle mask | Δ (oracle gain) |
|---|---|---|---|
| Qwen2.5-7B | 11.5 (n=3) | 15.9 (n=3) | **+4.4** |
| Qwen2.5-3B | 10.6 (n=3) | 11.8 (n=2) | +1.2 (noisy) |
| Qwen2.5-14B | *(running)* | | |

Consistent with an earlier multi-model sweep (Qwen 3B/7B/14B/32B + Llama 3B/8B) in
which mask gain scaled inversely with capability (weakest models gain most; 32B ≈ 0):
strong models ignore honest clutter, weak models cannot. **The per-task mask carries
real, capability-dependent signal.** (Oracle gain is itself noisy — the 7B seed-1 was
+9.5, the 3-seed mean +4.4.)

## 3. Mechanism — the mask raises tool-selection accuracy

Why does a smaller menu help? Weak models are bad at *picking* tools, so extra options
cause misselection. Measuring, per tool call, whether the agent hit a GT-relevant tool:

| Model | Full (14 tools) | Oracle mask |
|---|---|---|
| Qwen2.5-3B | **44%** (263/593 calls) | 54% |
| Qwen2.5-7B | **59%** (69/116) | **82%** |

With the mask off the 7B misselects 41% of its calls; the oracle mask lifts the hit
rate to 82%. The 3B is worse on both counts (44%) and calls tools **5× more often**
(593 vs 116) — weak models are both trigger-happy and inaccurate. This is the direct
mechanism: **fewer irrelevant options → fewer misselections → higher answer accuracy.**

## 4. Global (pool-wide) mask optimization is NULL

Optimizing the *global* pool (a single mask applied to all tasks, e.g. leave-one-out
tool removal) did **not** move accuracy — apparent single-seed effects failed to
replicate across seeds and task-bootstrap. Which tools are in the *pool* is inert; only
the *per-task* mask carries signal. So the real target is per-task selection.

## 5. No-GT per-task selector — can we predict the oracle mask?

Honest-clutter relevance is **query-dependent** (unlike injected "poison" tools, whose
uselessness is tool-intrinsic and detectable by input-degeneracy probing — that signal
does not apply to honest tools). We therefore combine three no-GT signals; GT is used
only to *score* them, never to build the mask.

- **LLM router** — ask a model "which tools does this request need?" (reasoning signal)
- **Embedding match** — mpnet cosine between the query and each tool's description
  (semantic signal), keep top-4
- **Frequency prior** — always keep {OCR, ImageDescription}, the two most common tools
  (relevant to 128/97 of 229 tasks; base-rate signal)

**Offline coverage vs GT** (all-kept = fraction of tasks whose every GT tool is kept;
pool = avg menu size; oracle pool 2.3, full 14):

| Selector | all-kept | pool |
|---|---|---|
| Embedding (query↔tool, top-4) | 21% | 4.0 |
| LLM router — 7B | 22% | 1.9 |
| LLM router — 14B | 40% | 3.1 |
| Frequency prior {OCR, ImageDescription} | 11% | 2.0 |
| **hybridA** = 7B-router ∪ emb ∪ freq | **80%** | 6.0 |
| **hybridStrong** = 14B-router ∪ emb ∪ freq | **93%** | 6.7 |

Single signals are weak (~20% all-kept). The **frequency prior is the highest-leverage
recall booster** (router 22% → 64% when unioned with it), because the two common
perception tools cover exactly what router/embedding miss. A stronger router routes
better (7B 22% → 14B 40%). The hybrid reaches 80–93% coverage at pool 6.

## 6. Online recovery — how much oracle gain does the no-GT mask recover?

`recovery = (predicted − full) / (oracle − full)`, applied per task, same session as
full/oracle (fair):

| Model | Full | Oracle | Predicted (hybridA) | Recovery |
|---|---|---|---|---|
| Qwen2.5-7B | 11.5 | 15.9 | 13.5 (n=3) | **~45%** |
| Qwen2.5-3B | 10.6 | 11.8 | 14.6 (n=2) | predicted **> oracle** (noisy) |
| Qwen2.5-14B | *(running)* | | | |

- **7B: the no-GT mask recovers ~45% of the oracle gain** — a real but partial lever.
  It lifts selection accuracy 59% → 67% (vs oracle's 82%), consistent with recovering
  part, not all, of the gain (all-kept 80% and pool 6 ≫ oracle 2.3 → less decluttering).
- **3B: predicted (14.6) exceeds both full (10.6) and oracle (11.8)** — the 6-tool
  hybrid helps the very weak model *more* than its 2.3-tool GT mask (the extra common
  tools appear genuinely useful to it). Provisional: 3B oracle gain is tiny (+1.2, n=2).
- **Counterintuitive:** the higher-coverage hybridStrong (93%) recovered *less* than
  hybridA (80%) on the 7B, because a bigger pool declutters less — coverage does not
  monotonically buy recovery. The signal ablation isolates this.

## 7. Signal ablation *(in progress)*

A leave-one-out over {router, embedding, frequency} on the 7B (online) to measure which
signal drives *recovery* — in particular whether the router/embedding "intelligence"
adds anything beyond the trivial frequency prior plus pruning. Offline coverage already
shows the frequency prior is the recall backbone; the online marginal-contribution
table will be filled in when the run completes.

## 8. Limitations and open questions

- AnsAcc noise floor ±2.5–3 → per-model oracle gains are noisy (n = 2–3); the robust
  signals are selection accuracy and coverage.
- The no-GT selector **partially** recovers the oracle gain (~45% on 7B); it does not
  reach oracle, because predicting per-task relevance without GT is hard (coverage caps
  at 80–93%, pool stays ~6 vs oracle 2.3).
- Open: does recovery scale with capability (14B pending)? Which signal drives online
  recovery (ablation pending)? Can a stronger router (32B) or a learned selector close
  the remaining gap?

## Bottom line

On GTA-Atomic, the **per-task tool mask carries real, capability-dependent signal** —
the oracle mask lifts a weak agent's accuracy by driving a measurable jump in
tool-selection accuracy (7B: 59% → 82%) — while the **global pool mask is inert**. A
**ground-truth-free selector** built from query↔tool routing + embedding + a frequency
prior **recovers ~45% of the oracle gain on the 7B** — the first no-GT lever on the
honest-clutter mask (prior no-GT selection only worked on injected poison tools).
Closing the remaining gap to the oracle is the open problem.

---

*Models run across the mask/composition series: Qwen2.5-3B / 7B / 14B / 32B and
Llama-3.2-3B / 3.1-8B (Qwen3-8B was dropped — its thinking mode is incompatible with
the ReAct harness). Reproducible code and per-run outputs are in the project repo.*
