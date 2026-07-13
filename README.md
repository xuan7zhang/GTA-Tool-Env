# gta2-envlab — the tool **environment** as a first-class research object

We treat an agent's tool environment as an object **E = (m, C, Φ)** and ask whether it
can be *optimized*, independent of the model:

- **m** — the tool **mask** (which tools are in the pool)
- **C** — **composition** (fuse several tools into one macro-tool)
- **Φ** — tool **schema / description** rewrites

Testbed: **GTA-Atomic** (NeurIPS'24 GTA, 229 tasks, 14 real tools), served through a
FastAPI probe-proxy in front of the AgentLego tool server, driven by an
OpenCompass / Lagent ReAct agent (Qwen2.5 / Llama families on Killarney L40S).
Motivation: three papers argue raw agent accuracy hides what tools actually do
(Tool-Use Tax 2605.00136, VisualNeedle 2605.26380, "Do Multimodal Agents Really
Benefit from Tool Use" 2606.02357). The proxy lets us intervene on E at runtime and
read out the causal effect.

> **New here? Read in this order:** this README → [`RUNBOOK.md`](RUNBOOK.md) (full
> bring-up + every command) → [`analysis/reports/`](analysis/reports/) (findings) →
> [`versions.lock.md`](versions.lock.md) (env pins + deviations from upstream).

---

## What lives where

| path | what |
|---|---|
| `GTA/` | **vendored** open-compass/GTA (agentlego + opencompass) **with our patches applied in place** — committed so the repo clones-and-runs. `patches/` documents the isolated diffs. |
| `scripts/` | bring-up: env setup, `common_env.sh` (all `GTA_*` knobs), start LLM / tool server / proxy, `on_node.sh`, `Makefile` targets |
| `proxy/proxy.py` | the probe-proxy: passthrough / tool_free / corrupt_output / unreliable modes, per-tool masks, Φ rewrites, per-task call log. Runtime-reconfigurable via `POST /proxy_config`. |
| `envgen/` | generators for E-variants (masks, Φ rewrites, macro/distractor/poison tools) — write files, never mutate the dataset in place |
| `experiments/` | one runnable script per axis experiment ([`experiments/README.md`](experiments/README.md) indexes them) |
| `analysis/` | scorers + `reports/` (the write-ups and result JSONs) |
| `configs/` | `gta_atomic_env.py` (env-var-driven eval config) + run manifests |
| `results` | symlink → `/datasets/omni_pretraining/gta2/results` (gitignored; heavy outputs) |

**Heavy assets** (conda envs, models, dataset, results) live on
`/datasets/omni_pretraining/gta2` (`$GTA_BIG`), **never** in the repo. Code is on
`/project/.../gta2-envlab` (`$GTA_LAB`). Both are pre-provisioned on Killarney.

---

## Quickstart (on Killarney)

```bash
# 0. one-time: build the three conda envs on a compute node (see versions.lock.md)
#    scripts/setup_env_{lmdeploy,agentlego,opencompass}.sh   ->  $GTA_BIG/envs/*
#    (already built on the shared volume; rebuild only if $GTA_BIG is fresh)

# 1. get a shell on the GPU node holding your SLURM allocation
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/killarney/slurm.conf
srun --overlap --jobid=<YOUR_JOBID> --pty bash
cd /project/6101776/xzhan576/gta2-envlab

# 2. bring up the stack (tmux sessions gta_llm / gta_tool / gta_proxy)
make services
#    LLM  lmdeploy Qwen2.5-7B  GPU0 :12580
#    tools agentlego-server 14 tools  GPU2 :16181
#    proxy proxy.py  CPU :16281  (upstream 16181)
# no Serper/Mathpix keys -> GoogleSearch/MathOCR route to "unavailable" (fine)

# 3. reproduce the leaderboard baseline (accept within ±5 of AnsAcc 9.06)
make baseline

# 4. run an axis experiment (examples)
bash experiments/mask_model.sh qwen2.5-7b-instruct 12580 mask7b 3   # m-axis
bash experiments/regionread_exp.sh                                  # C-axis (RegionRead)
```

Services persist as long as the SLURM job lives (they die on job expiry, not on your
shell exiting — they are held by `srun --overlap` inside login-node tmux). Rebuild on
a new node if the job is re-allocated.

### Second stack (parallel lane / a new macro)

Some C-axis experiments run a **second** tool-server+proxy on a free GPU so the old
stack is untouched. `experiments/launch_regionread_stack.sh` (tool server 16182 on
GPU1 + proxy 16282, with the RegionRead macro) is the template; hold it in its own
tmux (`_ts2_fg.sh` / `_proxy2_fg.sh` are the foreground wrappers). See the RegionRead
section below.

---

## The knobs that drive everything

All in `scripts/common_env.sh` (GPUs/ports) and read by the eval config
`configs/gta_atomic_env.py` at runtime:

| env var | effect |
|---|---|
| `GTA_EXTRA_TOOLS=A,B` | force tools onto **every** task's menu (inject a macro/distractor) |
| `GTA_HIDE_TOOLS=A,B` | hide tools from the agent **menu** only (they stay forwardable, e.g. for a macro's internal calls) |
| `GTA_TASK_IDS=9,10,11` | run only these dataset ids (cheap subsets) |
| `GTA_TASK_SUBSET=N` | run only the first N tasks |
| proxy `POST /proxy_config` | `{mode, mask, per_tool_modes, phi_toolmeta, unavailable_tools}` at runtime |

The proxy `mask` filters `/openapi.json`, so the agent's built tool set = per-task GTA
resources ∩ mask, `+ GTA_EXTRA_TOOLS`, `− GTA_HIDE_TOOLS`.

---

## Findings so far (pointers into `analysis/reports/`)

- **m (mask)** — gain ∝ 1/capability. Per-task masking helps weak models
  (Qwen 3B +6.3 → 32B ~0; Llama 3B +3.4 → 8B +0.8); strong models ignore honest
  clutter. Mechanism: weak models misselect and can't recover.
  → [`FINAL_REPORT.md`](FINAL_REPORT.md), [`experiments/README.md`](experiments/README.md).
- **Φ + injection (poison)** — inject authoritative-wrong tools, then recover. An
  **input-grounding** selector (probe each tool for input-dependence) matches the
  oracle tool-set with 0 false-prunes and is regime-robust where frequency/echo
  baselines are fragile. → [`analysis/reports/SUMMARY.md`](analysis/reports/SUMMARY.md),
  [`REPORT.md`](analysis/reports/REPORT.md).
- **C (composition)** — *parallel* fusion loses, *dependency* fusion wins.
  `PerceiveAll` (ImageDescription‖OCR) is **−3.98** (redundant, anti-mask);
  `RegionRead` (TextToBbox→RegionAttributeDescription) is **+28.8** because it removes
  an orchestration barrier the weak model can't cross (clean tool-execution 5% → 55%).
  → [`analysis/reports/COMPOSITION.md`](analysis/reports/COMPOSITION.md).

**Caveat for all of it:** a ±2.5–3 AnsAcc run-to-run noise floor (tool-server
nondeterminism even greedy) → every accuracy claim is multi-seed; the load-bearing
evidence is the *deterministic* properties (selection sets, tool-execution rates), not
the noisy means. The RegionRead work also fixed a ReAct action-parsing bug that had
depressed tool fidelity in the earlier m/Φ runs — see COMPOSITION.md; those are worth
a re-check under the fix.

---

## Our patches to GTA (all experiment-gated, no-op for the stock baseline)

Documented as diffs under `patches/` and applied in place in `GTA/`:

- `models/lagent.py` — `GTA_EXTRA_TOOLS` / `GTA_HIDE_TOOLS` menu control,
  `set_task_id()` (X-GTA-Task-Id header for per-task proxy logging), a dotted-tool-name
  parse fix, and `_patch_lagent_json_parser_trailing()` (recover a valid JSON action
  when the LLM appends a hallucinated observation — critical for tool fidelity).
- `datasets/gta_bench.py` — `GTA_TASK_IDS` / `GTA_TASK_SUBSET` slicing.
- `icl_inferencer/icl_agent_inferencer.py` — `set_task_id(index)` hooks.
- `configs/gta_atomic_env.py` — the env-var-driven eval config (new).

Deps: **lagent 0.2.3** (not the repo-pinned 0.1.2). Full pins + upstream deviations in
[`versions.lock.md`](versions.lock.md).

Repo: `git@github.com:xuan7zhang/GTA-Tool-Env.git`.
