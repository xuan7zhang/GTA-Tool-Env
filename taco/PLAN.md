# TACO — Tool Attribute Causal Optimization: repository audit + execution plan

**Written before any TACO experiment is launched.** Frozen reference for run
accounting; deviations get logged in `results/taco/reports/TACO_REPORT.md` §2.

---

## 1. Repository audit (what already exists, what TACO must add)

### 1.1 Execution stack (reusable as-is — TACO changes none of it)

```
OpenCompass / Lagent ReAct (max_turn 10, greedy temp 0)
   ├── LLM     lmdeploy OpenAI-compatible endpoint            :12580
   └── tools ─► proxy/proxy.py  (mask / Φ / corrupt / log)    :16281
                   └─► agentlego-server (14 GTA tools)        :16181
```

| component | path | status for TACO |
|---|---|---|
| eval config (env-var driven) | `configs/gta_atomic_env.py` | reuse unchanged |
| agent + menu control | `GTA/opencompass/opencompass/models/lagent.py` | reuse unchanged |
| dataset (229 tasks) | `$GTA_BIG/data/gta_dataset/dataset.json` | reuse; never mutated |
| tool server | `scripts/start_toolserver.sh` | reuse unchanged |
| **proxy** | `proxy/proxy.py` | **extend**: add TACO transform layer |
| logprob capture | `GTA_LL_LOG` hook in `models/openai_api.py` | reuse (mean/min logprob, per-token entropy, top-1/2 margin, tokens) |
| split | `results/tool_graph_facility/split.json` | **reuse** (80 calib / 149 heldout, seed 42, `dataset_sha256` recorded) |

Menu algebra already supported:
`menu(q) = (per-task tools ∩ proxy.mask) + GTA_EXTRA_TOOLS − GTA_HIDE_TOOLS`,
optionally narrowed by `GTA_PREDICTED_MASK` (per-task JSON).
⇒ **a fixed global environment** `M` is realised as `proxy.mask = M` **and**
`GTA_EXTRA_TOOLS = M` (every query sees exactly `M`).
⇒ **a query-level selector** is realised with `GTA_PREDICTED_MASK` — structurally
distinct, and labelled as such everywhere.

### 1.2 Tool inventory (14) and TACO eligibility

| tool | native output | tasks (of 229) | TACO role |
|---|---|---|---|
| OCR | `(x1,y1,x2,y2) TEXT` lines, multi-line, long | 128 | **primary** (text extraction) |
| ImageDescription | free prose paragraph (~60–110 tok) | 97 | **primary** (image description) |
| Calculator | bare scalar (`36.39`) | 74 | **primary** (calculation) |
| CountGivenObject | bare int (`6`) | 34 | **primary** (regional/counting) |
| TextToBbox | `(x1,y1,x2,y2), score 88.5` lines | 19 | **primary** (localization) |
| RegionAttributeDescription | short phrase | 8 | **primary** (regional inspection), low-n |
| Solver | `[-2, 6]` | 16 | secondary (text) |
| GoogleSearch | — no API key → proxy `unavailable` | 40 | excluded from interventions (symmetric across conditions) |
| MathOCR | — no API key → proxy `unavailable` | 20 | excluded, as above |
| Plot, DrawBox, AddText, TextToImage, ImageStylization | image path | 15/15/18/19/18 | image-output ⇒ format/length interventions **not applicable**; participate only in mask/subset design |

**Consequence for RQ1**: format/length interventions have a real surface on
6 text tools (OCR, ImageDescription, Calculator, CountGivenObject, TextToBbox,
RegionAttributeDescription) + Solver. `RegionAttributeDescription` has only 8
tasks total (≈3 in calibration) — below the ≥20-pair target of §10 of the spec.
**Recorded as a dataset limitation, not silently dropped**: it is intervened on,
and every per-tool claim carries its n.

### 1.3 Prior findings that constrain TACO's design (from `analysis/reports/`)

These are *this repository's own* established results; TACO must not re-derive
them and must not contradict them without evidence:

1. **Capability scissor**: mask gain (oracle − full) = +2.4 (3B), **+8.7 (7B)**,
   ≈0 (14B), ≈−1 (32B). ⇒ 7B is the sensitive model, 14B the "attributes
   should matter less" contrast. Directly motivates H6 / attribute×model.
2. **Global pool mask is inert; per-task menu carries the signal.** ⇒ the
   global-mask arm of TACO's search is *a priori* the weak arm; TACO's novel
   lever is Φ (format/length), not M. Preregistered expectation.
3. **Likelihood is answer-echo**: gold-Δlogprob AUC 0.580 → **0.483** once OCR
   tasks are removed. ⇒ RQ3's controls (§14) are mandatory, and H5 is the
   preregistered expectation.
4. **Clutter cliff**: oracle 20.9 → +2 distractors 13.7 → full-14 10.6 at 7B.
   ⇒ menu-size is a first-order covariate in every model; format effects must
   be estimated *within* a menu-size stratum (both minimal and full contexts).
5. **Noise floor**: greedy decoding ⇒ ±0.8 AnsAcc run-to-run from tool-server
   nondeterminism. ⇒ **every** attribute claim is a *paired per-task* delta with
   a paired bootstrap CI, never a difference of two run means.

### 1.4 What TACO must build (nothing else exists for it)

1. `taco/transform.py` — canonical-representation extractor + deterministic
   renderers for F0–F6, L0–L5, P∈{front,middle,back}; factual-preservation
   checker; transformation log.
2. `proxy/proxy.py` hook — `taco_spec` state key; `transform_output()` applied
   after upstream response, before returning to the agent. Default off ⇒
   byte-identical to today's behaviour for every prior experiment.
3. `taco/run.py` / `taco/runner.sh` — condition runner with `DONE` markers,
   per-run manifest, paired/interleaved ordering.
4. `taco/features.py`, `taco/model.py`, `taco/search.py`, `taco/report.py`.

---

## 2. Cost model (measured from existing runs)

| model | tasks | measured wall time | source |
|---|---:|---:|---|
| Qwen2.5-7B | 229 (end only) | **27 min** (≈7 s/task) | `ll7b_oracle` 05:29→05:56 |
| Qwen2.5-7B | 229 (step+end) | 66 min | `baseline_greedy` |
| Qwen2.5-32B | 229 (end only) | 43 min | `clean_keepall_s1` |

Derived unit costs (end-only, greedy):

| task set | 7B | 14B (×1.5) |
|---|---:|---:|
| 30 tasks (pilot) | 3.5 min | 5.3 min |
| 80 tasks (calibration) | **9.5 min** | **14 min** |
| 149 tasks (held-out) | 17.5 min | 26 min |
| tool-relevant subset (~25 avg) | 3 min | 4.5 min |

**Lanes.** kn064 (4×L40S, 2 d 16 h remaining) runs 2 independent lanes
(lane1: LLM gpu0 + toolserver gpu2 + proxy :16281; lane2: LLM gpu1 + toolserver
gpu3 + proxy :26281). kn063 (2 h 45 m left) and kn069 (2 idle GPUs, 18 h) are
opportunistic overflow only. Effective throughput ≈ **2 × 6 runs/hour** at
80-task granularity.

---

## 3. Staged plan with run counts

A **run** = one eval pass of one environment over one task set. Every run
writes `results/taco/raw_runs/<run_id>/` with `DONE`, `proxy.jsonl`,
`transform.jsonl`, `ll.jsonl`, predictions, metric JSON.

### Stage 0 — audit (no GPU beyond smoke tests)
Harness/leakage checks → `results/taco/phase0_audit/` + `PHASE0_AUDIT.md`.
**0 eval runs** (≈6 single-task smoke calls).

### Stage 1 — causal pilot (gate)
7B, 30 calibration tasks, tools {OCR, ImageDescription, TextToBbox,
RegionAttributeDescription}, conditions {F0, F2 concise, F4 JSON} × {L-short,
L-full}, contexts {minimal = per-task menu, full = forced 14}.

Cells: F0·L4, F2·L4, F4·L4, F0·L0, F0·L2 (=5) × 2 contexts = **10 runs**
≈ 35 min lane-time.

**Gate**: factual preservation ≥ 99% on the transform log; paired execution
works; ≥1 measurable behavioural or accuracy effect; infra failure < 10%.

### Stage 2 — full attribute study (7B + 14B, 80 calibration tasks)

| block | conditions | contexts | models | runs |
|---|---|---|---:|---:|
| A. uniform format policy (all text tools) | F0…F6 (7) | 2 | 2 | 28 |
| B. uniform length policy | L0,L1,L2,L3,L5 (5; L4≡F0) | 2 | 2 | 20 |
| C. length-mechanism decomposition @2× budget | relevant / irrelevant / redundant (3) | 2 | 2 | 12 |
| D. evidence position (secondary) | front / middle / back (3, back≡default) → 2 new | 1 (full) | 2 | 4 |
| E. single-tool interventions (target tool only, others native) | F2, F4, L0, L5 (4) × 6 tools | 1 (per-task menu) | 2 | 48 |
| F. distractor context (target + 2 matched irrelevant) | F2, L5 (2) × 6 tools | 1 | 1 (7B) | 12 |
| **Stage 2 total** | | | | **124** |

Lane-time: blocks A–D 64 runs × ~12 min = 12.8 h; E–F 60 runs × ~4 min = 4 h
⇒ **≈17 h single-lane, ≈8.5 h on 2 lanes.**

### Stage 2b — subset design (Phase 7 of the spec)

Balanced incomplete block design over the 14 tools, `|S| ∈ {1,2,3,5,8,14}`.
Spec target is 101 subsets; at 80 calibration tasks that is 20 h/model/lane.
**Budgeted**: 60 subsets at 7B + 24 at 14B = **84 runs** (≈14 h ⇒ 7 h on
2 lanes). Design saved to `results/taco/subset_design.json` with the
coverage table (per-tool appearances, pairwise co-occurrence) so the shortfall
against the spec's 101 is explicit and auditable, not silent.

### Stage 3 — utility model (gate, **0 eval runs**)
Fit TACO-Intrinsic + TACO-Conditional on Stage 1/2/2b paired effects with
grouped CV (group = task; also leave-one-tool-out, leave-one-category-out).
**Gate**: grouped AUC for `Δacc>0` ≥ 0.60 **or** stable, meaningful R²; LOTO
not catastrophic.

### Stage 4 — global search on calibration only

| search | runs |
|---|---:|
| greedy forward, empirical (calibration accuracy), 3 stages × {14,13,12} | 39 (7B only) |
| beam / random / coordinate descent over Φ — **model-predicted**, no eval | 0 |
| calibration verification of the top-3 candidate environments per model | 6 |
| random same-size mask reference distribution on calibration (10 draws) | 10 (7B) |
| **Stage 4 total** | **55** |

≈9 h single-lane. **Gate**: TACO-selected envs beat the calibration
random-mask median; stable across calibration folds; no single task dominates.

### Stage 5 — held-out (149 tasks), frozen environments, **run once**

Environments per model (all frozen before this stage):

global: `KeepAll`, `KeepNone`, `random-same-size ×3`, `global top-k utility`,
`greedy-empirical`, `coverage/facility-location`, `uniform-concise`,
`uniform-JSON`, `uniform-truncated`, `TACO-Intrinsic`, `TACO-Conditional-avg`
= **13**
query-level baselines (labelled): `embedding top-k`, `LLM router`,
`oracle relevant`, `TACO-Conditional per-query top-k` = **4**

7B: 17 runs × 17.5 min = 5 h. 14B: 13 runs (global only) × 26 min = 5.6 h.
**Stage 5 total = 30 runs ≈ 10.6 h ⇒ 5.3 h on 2 lanes.**

### Stage 6 — cross-model / cross-family robustness (only if time remains)
Frozen 7B environment evaluated on 14B and on Llama-3.1-8B, held-out:
**4 runs** (≈1.5 h). Predictor transfer (train 7B → test 14B) is analysis-only.

### Total

| stage | runs | lane-hours | 2-lane wall |
|---|---:|---:|---:|
| 0 | 0 | 0.3 | 0.3 |
| 1 | 10 | 0.6 | 0.6 |
| 2 | 124 | 17 | 8.5 |
| 2b | 84 | 14 | 7 |
| 3 | 0 | 1 | 1 |
| 4 | 55 | 9 | 4.5 |
| 5 | 30 | 10.6 | 5.3 |
| 6 | 4 | 1.5 | 0.8 |
| **total** | **307** | **54** | **≈28 h** |

Against kn064's 2 d 16 h remaining this fits with ~50% slack for failures.
**Priority order if compute is lost**: Stage 1 → 2A/2B (format+length core,
7B) → 3 → 4 → 5 (7B) → 2 (14B) → 2b → 6. A truncated run is reported as
truncated (see §25 "negative but useful"), never as a completed stage.

---

## 3b. Measured power reality (audit finding — changes the run allocation)

Three numbers measured from the dataset and from prior runs, before launching:

**(i) Only 65 of the 80 calibration tasks are scorable.** GTA-Atomic's
`gt_answer` is `null` for 57 of 229 tasks (image-generation tasks, which the
evaluator excludes from AnsAcc) and a list for 16 (sentence-embedding
simscore, excluded from the paired *binary* analysis). Calibration: 63 dict +
2 list ⇒ **n = 63** for paired binary effects. Held-out: 93 + 14 ⇒ n = 93.

**(ii) The agent is tool-lazy, so an intervention often never fires.**
Measured on prior 7B runs, calibration tasks with ≥1 tool call:

| context | engaged / 80 calibration tasks |
|---|---:|
| per-task (minimal) menu | **39** |
| forced full 14-tool menu | **19** |

An output intervention is a no-op on a task where the tool is never called.
⇒ every effect is reported twice: **ITT** over all scored tasks, and
**treated** over tasks where the transform actually fired (`fired` flag in
`paired_effects.parquet`). ⇒ the **minimal context is promoted to primary**
and the full context to secondary; the distractor context (block F) is the
first thing cut if compute runs short.

**(iii) Per-tool n is very uneven.** Scorable calibration tasks whose menu
contains the tool: OCR 43, Calculator 29, ImageDescription 16,
CountGivenObject 16, Solver 7, TextToBbox 3, RegionAttributeDescription 2.
⇒ the spec's "≥20 relevant pairs per major tool" is reachable for **OCR and
Calculator only**; ImageDescription and CountGivenObject are exploratory;
**TextToBbox and RegionAttributeDescription cannot support a per-tool claim
at all** and appear only inside the uniform policy. This is a property of
GTA-Atomic, is recorded here in advance, and is reported as a limitation
rather than papered over with an underpowered per-tool table.

Consequence for the verdict: with n ≈ 63 paired binary outcomes and a subset
of those treated, a single condition can only resolve large effects. The
hierarchical model pooling across conditions, tools and both models is what
carries the attribute claims — not any individual cell.

## 4. Preregistration (frozen here, before any TACO run)

- **H1** length is non-monotonic (inverted-U / saturating), not universally
  increasing.
- **H2** structured format (JSON/KV) is model-dependent: helps 14B more than
  7B; may hurt 7B.
- **H3** evidence density (evidence tokens / total tokens) predicts utility
  better than raw length.
- **H4** tool utility is context-dependent (menu size, redundancy, scale,
  relevance).
- **H5** likelihood is *not* a sufficient utility signal; its intervention
  delta weakens after controlling for length, answer overlap, OCR, tool-call
  occurrence. *(Repo prior: gold-Δlogprob AUC 0.580 → 0.483 without OCR.)*
- **H6** model-specific environments beat one universal environment.
- **Repo-prior H0 (added)**: the global-**mask** arm contributes little
  (global pool mask is inert here); if TACO wins, it should win through **Φ**
  (format/length), not through M. Stated in advance so a Φ-driven win is not
  retrofitted as a mask result, and vice versa.

Interactions preregistered for the hierarchical model: length×model,
format×model, length×relevance, format×relevance, length×menu-size,
format×menu-size, position×length, relevance×clutter, category×model,
overlap×likelihood.

## 4b. Execution log (deviations from §3, recorded as they happen)

| when | event | effect on the plan |
|---|---|---|
| pre-Stage-1 | prediction keys are *positions* after `GTA_TASK_IDS` filtering, not dataset ids | fixed in `runner.per_task_outcomes`; the first pilot's numbers were discarded, not reported |
| pre-Stage-1 | `X-GTA-Task-Id` never reaches the proxy (pre-existing: prior runs' logs also carry an empty task_id) | per-task attribution now via `GTA_CURTASK_FILE`; proxy reads it when the header is absent |
| Stage 1 | **lmdeploy segfault** in `Sampling::Update()` under sustained `top_logprobs=20`; both lanes' LLMs died and every task returned a connection error | `GTA_LL_TOPK` lowered to 5, `start_llm_supervised.sh` restarts the server, and the runner **halts a lane** rather than record a serving failure as an accuracy effect (`conn_errors` in every record) |
| Stage 1 | a second session began running this same code into `results/taco/` | `TACO_ROOT` added (`taco/paths.py`); this session writes to `results/taco_b/`, the default is left alone so the other session is unaffected |
| Stage 2 setup | pairing key used the full block name, so blocks 2B/2C/2D had no control (their F0/L4 control lives in 2A) | key shortened to the stage prefix; length, mechanism and position blocks now pair correctly |
| Stage 2 setup | `tool_exec_failures` was counting the by-design `unavailable` routing of GoogleSearch/MathOCR | split into `unavailable_calls` / `masked_calls` / real `tool_exec_failures` |
| Stage 4 | empirical greedy cut from 3 stages (39 runs) to 2 (27 runs) | recorded here rather than reported as a full greedy search |

**Stage 1 gate: PASSED.** 0 factual-preservation failures over 16 runs, 0
connection errors, 14 paired conditions built, and at least one measurable
effect (7B / minimal context / L2, treated +33.3 [8.3, 58.3]).

**Analysis pipeline verified end to end on pilot data**: effects → stats →
model → plots all run clean; TACO-Conditional grouped AUC 0.654 on pilot-only
data (the Stage-3 gate is 0.60 on the full data, not on this).

## 5. Leakage rules (enforced mechanically, audited in Phase 0)

1. The 149 held-out ids never enter feature fitting, hyperparameter choice,
   subset search, threshold choice, or environment freezing. The runner
   **refuses** to launch a non-`heldout/` run whose task list intersects them.
2. Annotation-derived features (`annotated relevance`, `reference-chain
   membership`) exist **only** in the analysis model; the deployable predictor
   is fitted on a feature matrix from which those columns are physically
   dropped, and that is asserted in code.
3. Gold-answer likelihood is retrospective analysis only; never an input to a
   search objective or a frozen environment.
4. Grouped CV by task (and by tool / category for LOTO / LOCO). Transformed
   variants of one task never straddle a fold boundary.
5. Frozen environments are written to
   `results/taco/frozen_environments/frozen_envs.json` with a hash **before**
   Stage 5, and Stage 5 asserts the hash.
