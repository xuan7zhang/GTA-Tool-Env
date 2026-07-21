# gta2-envlab — the tool **environment** as a first-class research object

This project studies an LLM agent's **tool environment** as an object that can be
optimized independently of the model:

**E = (m, C, Φ)**
- **m** — the tool **mask** (which tools are in the pool / on each task's menu)
- **C** — **composition** (fusing several tools into one macro-tool)
- **Φ** — tool **schema / description** rewrites

We put a configurable **probe-proxy** in front of the tool server so we can intervene
on E at run time — mask, corrupt, reorder, rewrite descriptions, inject tools — without
touching the agent or the dataset, and read out the causal effect on task accuracy.

**Testbed:** GTA-Atomic (NeurIPS'24 GTA — 229 multimodal tasks, 14 real executable
tools) served by an AgentLego tool server, driven by an OpenCompass / Lagent ReAct
agent (tested with Qwen2.5 3B–32B and Llama-3.x families).

> **New here? Read in this order:** this README → [`RUNBOOK.md`](RUNBOOK.md) (full
> bring-up, every command) → [`analysis/reports/`](analysis/reports/) (findings) →
> [`versions.lock.md`](versions.lock.md) (dependency pins + deviations from upstream).

---

## Architecture

```
OpenCompass / Lagent ReAct agent  (max_turn 10)
   ├── LLM         OpenAI-compatible endpoint (e.g. lmdeploy / vLLM)     :$LLM_PORT
   └── tools ───►  proxy.py  (mask / corrupt / Φ-rewrite / inject / log) :$PROXY_PORT
                      └────►  AgentLego tool server (14 GTA tools)        :$TOOL_PORT
```

Every environment intervention happens at the **proxy**, reconfigurable at run time via
`POST /proxy_config`. The agent and the dataset are never modified.

---

## Repository layout

| path | contents |
|---|---|
| `scripts/` | bring-up: env setup, `common_env.sh` (all `GTA_*` knobs), start LLM / tool server / proxy, `Makefile` targets |
| `proxy/proxy.py` | the probe-proxy: passthrough / tool_free / corrupt_output / unreliable modes, per-tool masks, Φ rewrites, per-task call log |
| `envgen/` | generators for E-variants (masks, Φ rewrites, macro / distractor / poison tools) — write files, never mutate the dataset |
| `experiments/` | one runnable script per experiment ([`experiments/README.md`](experiments/README.md) indexes them) |
| `analysis/` | scorers + `reports/` (write-ups and result JSONs) |
| `configs/` | `gta_atomic_env.py` (env-var-driven eval config) + run manifests |
| `GTA/` | vendored [open-compass/GTA](https://github.com/open-compass/GTA) (agentlego + opencompass) with our patches applied in place; `patches/` documents the isolated diffs |

**Code vs. heavy assets.** Code lives in this repo (`$GTA_LAB`). Heavy assets — conda
envs, model checkpoints, the dataset, and run outputs — live on a separate large volume
(`$GTA_BIG`) and are **never** committed. `results/` is a symlink into `$GTA_BIG`.

---

## Setup

### 1. Point the two location variables at your machine

Override the two location variables when needed (the code path otherwise
defaults to the current checkout):

```bash
export GTA_LAB=/path/to/this/repo          # optional; defaults to this checkout
export GTA_BIG=/path/to/large/volume/gta2  # defaults to /datasets/omni_pretraining/gta2
```

All other scripts `source common_env.sh`, so exported overrides propagate to them.
GPU/port defaults also live here and are overridable via env:

```bash
GTA_LLM_GPUS=0   GTA_LLM_PORT=12580     # LLM (one L40S fits 7B; use "0,1" tp=2 for 32B)
GTA_TOOL_GPU=2   GTA_TOOL_PORT=16181    # AgentLego tool server
GTA_PROXY_PORT=16281                    # probe-proxy (CPU)
```

### 2. Build the three conda environments (one-time)

```bash
scripts/setup_env_lmdeploy.sh      # LLM serving
scripts/setup_env_agentlego.sh     # tool server (perception models)
scripts/setup_env_opencompass.sh   # agent + eval harness
```

These write to `$GTA_BIG/envs/{lmdeploy,agentlego,opencompass}`. Exact pins and the
(deliberate) deviations from GTA's upstream instructions are in
[`versions.lock.md`](versions.lock.md); the key one is **lagent 0.2.3** (not the
repo-pinned 0.1.2).

### 3. Get the dataset

GTA-Atomic = the original `gta_dataset` (v0.1.0, 229 tasks) from the GTA release /
`Jize1/GTA` on HuggingFace. Place it at `$GTA_BIG/data/gta_dataset/`
(`dataset.json`, `toolmeta.json`, `image/`).

**API keys (optional).** `GoogleSearch` needs a Serper key and `MathOCR` a Mathpix
key. Without them the proxy routes those two tools to an "unavailable" response and
logs which tasks touch them (~60 of 229). All comparisons stay valid because the two
stubbed tools are symmetric across conditions.

---

## Quickstart

Run these on a GPU node (with SLURM: `srun --overlap --jobid=<JOB> --pty bash`; or any
machine with the GPUs). From the repo root:

```bash
# bring the stack up (three tmux sessions: gta_llm / gta_tool / gta_proxy)
make services

# reproduce the leaderboard baseline (step + end metrics)
make baseline

# run an experiment (examples)
bash experiments/mask_model.sh qwen2.5-7b-instruct 12580 mask7b 3   # mask axis
bash experiments/regionread_exp.sh                                  # composition axis
```

The services are held by `srun --overlap` inside tmux, so they survive your shell
exiting; they die when the SLURM allocation ends. Rebuild on a new node if reallocated.

### Decoding is deterministic by default

The eval config uses **greedy (temperature 0) decoding by default**, so runs are
reproducible. (The OpenCompass `OpenAI` default is 0.7, which silently sampled and was
the main source of the ±2.5–3 run-to-run AnsAcc noise; greedy tightens the residual
floor to ~±0.8, from tool-server nondeterminism only.) Set `GTA_TEMP=0.7` to opt back
into sampled decoding.

---

## The knobs that drive experiments

Read at run time by [`configs/gta_atomic_env.py`](configs/gta_atomic_env.py):

| variable / call | effect |
|---|---|
| `GTA_EXTRA_TOOLS=A,B` | force tools onto **every** task's menu (inject a macro / distractor) |
| `GTA_HIDE_TOOLS=A,B` | hide tools from the agent **menu** only (still forwardable, e.g. for a macro's internal calls) |
| `GTA_PREDICTED_MASK=f.json` | restrict each task's menu to a predicted per-task keep-set |
| `GTA_TASK_IDS=9,10,11` / `GTA_TASK_SUBSET=N` | run a chosen id list / the first N tasks |
| `GTA_TEMP=0` | greedy decoding |
| `POST /proxy_config` | `{mode, mask, per_tool_modes, phi_toolmeta, unavailable_tools}` at run time |

The agent's per-task menu = (per-task GTA tools ∩ proxy `mask`) `+ GTA_EXTRA_TOOLS`
`− GTA_HIDE_TOOLS`, optionally narrowed by `GTA_PREDICTED_MASK`.

---

## Our patches to GTA

All experiment-gated (no-op for the stock baseline), applied in place in `GTA/` and
documented as diffs under `patches/`:

- `models/lagent.py` — menu control (`GTA_EXTRA_TOOLS` / `GTA_HIDE_TOOLS` /
  `GTA_PREDICTED_MASK`), per-task `set_task_id()` for proxy logging, a dotted-tool-name
  parse fix, and a JSON-action parser that tolerates a hallucinated trailing
  observation (recovers otherwise-dropped tool calls).
- `datasets/gta_bench.py` — `GTA_TASK_IDS` / `GTA_TASK_SUBSET` slicing.
- `icl_inferencer/icl_agent_inferencer.py` — `set_task_id(index)` hooks.
- `configs/gta_atomic_env.py` — the env-var-driven eval config (temperature, model,
  tool server, modes, task subset — all from `GTA_*`).

To reproduce from a clean GTA checkout instead of the vendored copy, apply the files
under `patches/` per `patches/APPLY.md`.

---

## Findings (see `analysis/reports/` for details and caveats)

- **Mask (m).** A global (pool-wide) mask is inert, but the **per-task** mask carries
  a large, capability-dependent signal — an oracle per-task mask lifts a weak agent's
  accuracy by raising its tool-selection accuracy. Predicting that mask without ground
  truth is only partially solved.
  → [`analysis/reports/MASK.md`](analysis/reports/MASK.md), `FINAL_REPORT.md`.
- **Trust / injection (Φ + C).** Injecting unreliable tools measurably harms the
  agent; a ground-truth-free **input-grounding** probe identifies them with no false
  positives and is robust across attack regimes where outcome-based selectors are
  fragile. → [`analysis/reports/SUMMARY.md`](analysis/reports/SUMMARY.md),
  [`REPORT.md`](analysis/reports/REPORT.md).
- **Composition (C).** *Parallel* fusion (concatenating independent tool outputs)
  hurts; *dependency* fusion (chaining a tool into the next, hiding the intermediate)
  helps a weak agent by removing an orchestration barrier it otherwise won't cross.
  → [`analysis/reports/COMPOSITION.md`](analysis/reports/COMPOSITION.md).

**Measurement note.** Accuracy carries a run-to-run noise floor from tool-server
nondeterminism (and, under sampled decoding, from the LLM). Use `GTA_TEMP=0` and
multi-seed paired designs; the load-bearing evidence is deterministic detector /
selection properties, not single-run means.

---

## Reproducing a result end to end

1. `make services` (or bring up LLM + tool server + proxy manually — see `RUNBOOK.md`).
2. Pick an experiment script in `experiments/` (indexed in `experiments/README.md`).
3. Run it; outputs land under `$GTA_BIG/results/<run_id>/` (`predictions/` =
   per-sample trajectories with every tool call and return; `results/` = metric JSONs;
   `proxy_calls.jsonl` = the raw per-task tool-call log).
4. Aggregate with the scorers in `analysis/` (e.g. `pertask_score.py`).
