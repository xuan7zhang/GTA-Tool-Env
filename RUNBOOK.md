# gta2-envlab RUNBOOK

GTA-Atomic (NeurIPS'24 GTA, 229 tasks) deployment on Killarney + causal
tool-contribution probes. Code: `/project/6101776/xzhan576/gta2-envlab`.
Heavy assets: `/datasets/omni_pretraining/gta2` (envs, models, data, results).

## Topology

```
opencompass env (agent loop, Lagent ReAct max_turn=10)
   ├── LLM:   lmdeploy api_server  Qwen2.5-7B-Instruct   GPU 0   :12580
   └── tools: proxy/proxy.py  (probes/noise/mask)  CPU   :16281
                └── agentlego-server (14 GTA tools)  GPU 2   :16181
```

GPU/port knobs all live in `scripts/common_env.sh` (GTA_* env vars).
Second sweep lane: set `GTA_LLM_GPUS=1 GTA_TOOL_GPU=3 GTA_LLM_PORT=22580
GTA_TOOL_PORT=26181 GTA_PROXY_PORT=26281` and start a second service stack.

## Services (run on the GPU node, e.g. kn064)

```bash
scripts/on_node.sh bash          # shell on the node via srun --overlap
cd /project/6101776/xzhan576/gta2-envlab
make services                    # tmux: gta_llm, gta_tool, gta_proxy
```

Or individually: `make llm` / `make toolserver` / `make proxy`.

API keys: export `SERPER_API_KEY`, `MATHPIX_APP_ID`, `MATHPIX_APP_KEY` before
`make toolserver`. If absent (**--no-external-api mode**): add
`"unavailable_tools": ["GoogleSearch", "MathOCR"]` to the proxy config —
tools stay in the registry, calls get the "unavailable" response, and the
proxy log records exactly which tasks touched them.

## Task 0 — baseline

```bash
make baseline        # = GTA_EVAL_MODES=step,end scripts/run_eval.sh baseline
```

Metrics: step mode → InstAcc/ToolAcc/ArgAcc/SummAcc; end mode → AnsAcc + P/O/L/C F1.
Reference (public leaderboard, qwen2.5-7b-instruct): Inst 56.38, Tool 32.85,
Arg 5.57, Summ 65.75, Ans 9.06. Acceptance: ±5 pts.

Outputs: `/datasets/omni_pretraining/gta2/results/baseline/<ts>/`
(`predictions/` = per-sample trajectories incl. every tool call + return;
`results/` = metric JSONs; plus `proxy_calls.jsonl` raw-return log).

## Task 1 — causal probes

Proxy modes: passthrough / tool_free / corrupt_output (=format_only) /
result_only (stub) / unreliable(p_fail, seed). Switch at runtime:

```bash
curl -X POST localhost:16281/proxy_config -H 'Content-Type: application/json' \
     -d '{"mode": "tool_free"}'
```

Health-check runs: `make baseline` (A), `make causal-b` (B), `make causal-c` (C), then

```bash
conda activate /datasets/omni_pretraining/gta2/envs/opencompass
python analysis/causal_2x2.py \
  --run-a  $GTA_BIG/results/baseline \
  --run-b  $GTA_BIG/results/causal_B_toolfree \
  --run-c  $GTA_BIG/results/causal_C_corrupt \
  --dataset $GTA_BIG/data/gta_dataset/dataset.json \
  --out analysis/causal_report.md
```

## Task 2 — environment variants

Generators (files only, dataset never modified in place):

```bash
cd envgen
python3 gen_masks.py --toolmeta $GTA_BIG/data/gta_dataset/toolmeta.json --out variants/masks
python3 gen_phi.py   --toolmeta $GTA_BIG/data/gta_dataset/toolmeta.json --out variants/phi --seed 0
python3 gen_extra_tools.py --out variants/compose --macros --n-distractors 5 --seed 0
```

- **m (mask)**: 14 leave-one-out + random 25/50/75% × 3 seeds + full. Masks are
  applied *at the proxy* (`/proxy_config {"mask": [...]}` filters openapi.json,
  so RemoteTool only builds the kept subset — no tool-server restart), plus the
  pruned `toolmeta.json` via `GTA_TOOLMETA`.
- **Φ (schema rewrite)**: paraphrase / degraded / misleading toolmeta variants.
  The agent-visible descriptions come from the tool server's openapi summaries,
  so apply Φ through the proxy: `/proxy_config {"phi_toolmeta": "<variant>/toolmeta.json"}`
  (also set `GTA_TOOLMETA` to the same file for consistency of dummy-tool gaps).
- **C (compose+distractors)**: `extra_tools.py` for
  `agentlego-server start --extra`; macro-tools call primitives *through the
  proxy* so probes apply to inner calls. Expose to the agent with
  `GTA_EXTRA_TOOLS=<names>` (GTA tasks otherwise only see their per-task tools).

Sweep: `make pilot-sweep` (manifest `configs/manifest_pilot_loo.jsonl`,
14 LOO × {passthrough, tool_free} × seed 0). Aggregate: `make aggregate`.

**Analysis convention**: for every variant E report
ΔTool(E) = AnsAcc(E, passthrough) − AnsAcc(E, tool_free), not raw accuracy alone.

## Known facts / gotchas

- GTA repo's "GTA-2" README dataset link (`gta_dataset_v2.zip`) is stale — no
  such release exists. GTA-Atomic == original 229-task `gta_dataset` (v0.1.0).
  The shipped `eval_gta_bench_v2.py` targets GTA-Workflow (end.json, GPT-judge)
  — different benchmark, not used here.
- Each GTA task exposes only its 1–4 `resources` tools to the agent; masked
  tools are skipped gracefully (patch in `opencompass/models/lagent.py`);
  extra tools are appended via `GTA_EXTRA_TOOLS`.
- Compute Canada login shells poison pip (`PIP_CONFIG_FILE` constraints +
  wheelhouse) and leak `PYTHONPATH`; `common_env.sh` neutralizes both.
- Step-wise (`every_with_gt`) mode predicts single steps from gt prefixes and
  does not execute predicted tools — causal probes act on end-to-end mode.
- Evaluator downloads `all-mpnet-base-v2` (pre-fetched into `$GTA_BIG/hf_home`).
```
