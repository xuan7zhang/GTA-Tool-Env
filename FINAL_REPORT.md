# GTA-Atomic tool-environment study — final report

Scope: deploy GTA-Atomic (229-task GTA, 14 tools) on Killarney, build causal
tool-contribution probes, and test whether the tool environment E=(m,C,Φ) can be
**optimized**. All numbers from `/datasets/omni_pretraining/gta2/results/`;
methods and gotchas in `RUNBOOK.md` / `versions.lock.md`.

**One-line bottom line.** The causal probes work and give clean motivation
stats, but **mask-axis environment optimization has no robust effect on GTA** —
neither mean accuracy nor tool-call cost survives multi-seed / bootstrap
scrutiny. The only surviving positive lead is a **variance-reduction** effect
(pruning stabilizes the agent), and the real performance bottleneck is the
**harness/Φ axis** (tool presentation), not which tools are in the pool.

---

## 1. Deployment (Task 0) — PASSED

Qwen2.5-7B-Instruct via LMDeploy + AgentLego tool server + FastAPI probe proxy,
OpenCompass/Lagent ReAct (max_turn 10), on kn066 (idle L40S job; kn064 runs the
user's SDPO RL). Baseline vs public leaderboard (±5 sanity): Inst 51.99/56.38,
Tool 33.39/32.85, Arg 8.98/5.57, Summ 66.14/65.75, AnsAcc 14.35/9.06. Pass.
No Serper/Mathpix keys → GoogleSearch/MathOCR run proxy-unavailable (60/229
tasks touch them; recorded). 458 per-sample trajectories saved.

Dataset note: the brief's `gta_dataset_v2.zip` does not exist; GTA-Atomic is the
original v0.1.0 `gta_dataset`, scored by GTABenchEvaluator (the leaderboard's own
metric). Details + 14 deviations in `versions.lock.md`.

## 2. Causal probes (Task 1) — the solid contribution

A=passthrough 14.35, B=tool_free 10.85, C=corrupt_output 10.79 (`causal_report.md`).

- **ΔTool = 3.5**; tool-free keeps 76% of tool-using accuracy.
- **C ≈ B**: corrupting tool output is as damaging as removing tools.
- 2×2 (n=172): both-correct 5, tool-rescued 17, tool-hurt 9, both-wrong 141 (82%).
- **Mechanism (corrected):** the small net effect is NOT "the model ignores tool
  output". Per sample, 91.7% of correct tool-using answers break under corruption
  (100% on the tool-clean subset). When the model succeeds via tools it genuinely
  depends on them; the net effect is small because genuine tool wins are rare AND
  tools induce a comparable number of errors (tool-hurt ≈ ½ tool-rescued). C≈B
  because corruption and removal destroy the same small set of tool-dependent wins.
- **De-confound** (`causal_deconfound.md`): on the 169 tool-clean tasks the
  rescued:hurt ratio (11:5) is cleaner than the full set (17:9); the dead external
  tools inflate the "tools don't help" reading. Core findings survive.

This is the publishable core (aligns with Tool-Use-Tax / 2606.02357 / VisualNeedle).

## 3. Environment optimization — mask axis (Task 2 + optenv)

### 3a. What single-seed LOO suggested (and why it was wrong)
Leave-one-out over 14 tools (seed 0) appeared to split the pool into
contributors (removal hurts: Calculator −2.65, ImageDescription −2.37, OCR −1.47…)
and distractors (removal helps: DrawBox +3.97, ImageStylization +2.78,
GoogleSearch +2.63…). **This did not replicate.** Greedy-pruning the "distractor"
set gave +0.11 (conservative) / +0.44 (aggressive) — vs the additive prediction
of ≈+9. LOO deltas are non-additive and single-seed-noisy.

### 3b. Multi-seed verdict (3 seeds/env, 7B) — no robust effect
| environment | AnsAcc (3 runs) | mean±std | tool_call (3 runs) | mean |
|---|---|---|---|---|
| baseline (14) | 14.35 / 19.05 / 13.07 | 15.49 ± 2.57 | 131/108/119 | 119 |
| conservative (11) | 14.46/14.27/14.86 | 14.53 ± 0.24 | 116/125/103 | 115 |
| aggressive (9) | 14.79/14.21/14.71 | 14.57 ± 0.25 | 111/127/104 | 114 |

- **Mean accuracy: identical** (~14.5 all three). Supersedes the single-seed
  "+0.11/+0.44" — those were noise.
- **Cost: no robust win.** Mean tool_call 119/115/114 with heavy overlap (~4%),
  not the −15%/−32% that single seeds suggested. **Retracted.**

### 3c. 32B — bootstrap confirms the null
Serving Qwen2.5-32B (tp=2) and re-running the three environments, task-resample
bootstrap (10^4, `bootstrap_env_32b.md`) gives **all three environments AnsAcc
11.63, 95% CI [6.98,16.28], P(env>base)≈0.45** — binarized correctness identical
across environments. Mask composition changes nothing.

Notable: 32B baseline AnsAcc (11.59) is **below** 7B (14.35), because 37% of its
tool-call steps are NoAction (ReAct-format parse failures) under the default
protocol — the bottleneck is tool presentation, not tool selection. 32B ΔTool:
baseline 1.78, aggressive-pruned 3.61 (pruning the dead/distractor tools roughly
doubles the *marginal* tool contribution even though mean accuracy is flat).

### 3d. The one surviving positive lead: variance, not mean
The full pool's across-run std is **±2.57** (13.07–19.05) vs **±0.24** for both
pruned pools — a ~10× difference at equal mean. Plausible mechanism: dead/
distractor tools fire stochastically → trajectory divergence → unstable scores;
pruning removes that. **Caveat: n=3, and the spread is driven by one high run
(19.05).** This is a lead, not a result — needs 5–10 seeds/env to confirm a
genuine variance-reduction effect.

## 4. Synthesis

- **Causal instrument: solid.** ΔTool, C≈B, the corrected dependence mechanism,
  and the tool-clean de-confound are the defensible results.
- **Mask-axis optimization: null.** Across 7B (3 seeds) and 32B (bootstrap),
  which tools are in the pool does not move mean accuracy or cost on GTA. The
  earlier single-seed "contributor/distractor" magnitudes and cost savings are
  retracted as noise.
- **Where a positive result likely lives:** (i) **variance** (pruning → stability;
  confirm with more seeds); (ii) **Φ axis** — the 32B NoAction 37% shows tool
  *presentation*/protocol is the real lever, untested here; (iii) **cost only
  under a competent, format-compliant agent**.

## 5. Recommended next steps (ready, not run)
1. **Seed the variance test** (5–10 seeds × {full, pruned}, 7B; parallel across
   idle nodes) — the highest-value confirm/kill of the one positive lead.
2. **Φ-axis experiment**: fix tools, add the ReAct protocol/system-prompt; test
   whether cutting 32B's NoAction 37% raises accuracy (Φ works where m doesn't).
3. Real Serper/Mathpix keys, or keep reporting de-confounded ΔTool.
4. A/B/C at 2–3 seeds (rescued 17 vs hurt 9 differ by 8 samples).

Artifacts: `analysis/` (causal_report, causal_deconfound, sweep.csv,
deltatool_table, optenv_result, bootstrap_env_32b, bootstrap_env.py,
causal_2x2.py, aggregate_results.py, dump_trajectories.py); `envgen/variants/`
(masks, phi, compose, optenv); all service/eval scripts in `scripts/`.
