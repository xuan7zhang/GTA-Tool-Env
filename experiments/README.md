# experiments/ — GTA tool-environment experiment scripts

Scripts copied back from `/datasets/omni_pretraining/gta2/scripts_extra/`.
They run against the live kn0** stack (LLM + AgentLego tool server + proxy) and
write results to `/datasets/omni_pretraining/gta2/results/`. Heavy assets
(models, conda envs, dataset, results) stay on `/datasets` — only code lives here.
Reports + selector JSONs are in `../analysis/reports/`.

## E = (m, C, Φ) axis experiments

### m — mask / tool-set (the accuracy-gain result)
- `mask_model.sh <model> <port> <prefix>` — injection-free mask experiment:
  `full_everywhere` (every task sees all 14 tools) vs `per_task` (only its 1–4
  resources). Result: **mask gain ∝ 1/capability** — Qwen2.5 3B +6.3 / 7B +4.9 /
  14B +2.9 / 32B ~0; Llama 3B +3.4 / 8B +0.8 (direction universal, magnitude
  family-dependent). Mechanism: weak models misselect (3B 42% of calls) and can't
  recover from irrelevant evidence; per-task mask filters for them.
- `mask_7b.sh` / `mask_C.sh` — the original single-model versions (7B / 32B).
- `mask_acc.sh` (clean pool full14 vs lean9 → null) / `mask_accB.sh` (polluted
  pool full22 vs pruned14) — earlier mask iterations.

### injection → optimization (poison) + selectors
- `subtle_poison_run.sh` — attractive vs subtle poison, two-regime study.
- `grounding_selector.py` — **the winning selector**: input-grounding
  (black-image contrast for image tools + output-cardinality for text tools) →
  catches 8/8 poison, 0 false-prune, regime-invariant, no ground truth.
- `grounding_plus.py` — grounding poison ∪ image-output declutter.
- `answer_echo_selector.py` — answer-echo baseline (regime-fragile).
- `cf_seeds.sh` / `cf_att_more.sh` — call_frequency baseline multi-seed.
- `grounding_seeds.sh` / `run_grounding.sh` / `rerun_methods.sh` / `iter_run.sh`
  — multi-seed drivers for the method comparison.

### C — composition (in progress)
- `macro_tools.py` — `PerceiveAll` macro = ImageDescription ⊕ OCR in one call
  (chains primitives through the proxy). Registered via
  `start_toolserver.sh` with `GTA_MACRO=1`.
- `compose_exp.sh` — baseline (2 primitives) vs composed (PerceiveAll) on the 42
  tasks needing both perceptions, 7B, 3 seeds.

## Notes
- All eval runs `cd /datasets/omni_pretraining/gta2/ocrun` (opencompass cwd,
  chosen to survive a /project group-quota spike; tmp/ params write there).
- ±2.5–3 AnsAcc run-to-run noise (32B/tool-server nondeterminism even greedy) →
  every claim uses 3 seeds; the deterministic selector properties (8/8 poison,
  misselection rates) do not depend on it.
