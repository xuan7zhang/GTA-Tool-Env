# Overnight run summary (2026-07-05)

All three tasks completed end-to-end on kn066 (idle L40S job `l1`; kn064 runs the
user's SDPO RL). Services still up in login-node tmux (`gta_llm/gta_tool/gta_proxy`).

## Task 0 — baseline (PASSED sanity check, ±5 vs leaderboard)

| metric | ours | leaderboard | Δ |
|---|---|---|---|
| InstAcc | 51.99 | 56.38 | −4.4 ✓ |
| ToolAcc | 33.39 | 32.85 | +0.5 ✓ |
| ArgAcc  | 8.98  | 5.57  | +3.4 ✓ |
| SummAcc | 66.14 | 65.75 | +0.4 ✓ |
| AnsAcc  | 14.35 | 9.06  | +5.3 ⚠ borderline over |
| AnsAcc+I| 12.66 | 8.95  | +3.7 ✓ |

458 trajectories saved. No Serper/Mathpix keys → GoogleSearch/MathOCR run in
proxy-unavailable (no-external-api) mode; 60/229 tasks touch them (recorded).

## Task 1 — causal probes A/B/C (`analysis/causal_report.md`)

A=passthrough 14.35, B=tool_free 10.85, C=corrupt 10.79.

- **ΔTool = 3.5** (tool-free keeps 76% of tool-using accuracy).
- **C ≈ B**: corrupting tool output is as damaging as removing tools.
- 2×2 (n=172): both-correct 5, tool-rescued 17, **tool-hurt 9**, both-wrong 141 (82%).
- Only 8.1% of tasks solvable without tools.

**Corrected mechanism** (my first read was wrong): the small net effect is NOT
"model ignores tool output". Per-sample, 91.7% of correct tool-using answers
**break under corruption** (100% on the tool-clean subset) — when the model
succeeds via tools it genuinely depends on them. Net ΔTool is small because
genuine tool wins are rare AND tools induce a comparable number of errors
(tool-hurt ≈ ½ tool-rescued). C≈B because corruption and removal destroy the
same small set of tool-dependent wins.

**De-confound** (`analysis/causal_deconfound.md`): on the 169 tool-clean tasks
(no dead GoogleSearch/MathOCR), rescued:hurt = 11:5 — cleaner than the full-set
17:9; dead tools inflate the "tools don't help" reading. Core findings survive.

## Task 2 — pilot LOO sweep (14 tools × {passthrough, tool_free}, seed 0)

`analysis/sweep.csv` (28 runs, all DONE), `analysis/deltatool_table.md`.
Leave-one-out reveals a **contributor / distractor split** in the tool pool:

- **Removal HELPS (net distractors, creativity/operation):** DrawBox +3.97,
  ImageStylization +2.78, GoogleSearch +2.63 (dead), Solver +1.07, TextToBbox +0.62.
- **Removal HURTS (genuine contributors, perception/logic):** Calculator −2.65,
  ImageDescription −2.37, OCR −1.47, CountGivenObject −0.90, TextToImage −0.76.

**Caveat — single seed, noisy.** MathOCR (also a dead tool) shows −2.38, opposite
to GoogleSearch's +2.63 — proof of single-seed variance. The contributor/distractor
*grouping* is likely real; per-tool numbers (esp. ≤20-coverage tools) are not.
Confirm with the seeded random-mask variants (generated, not yet run).

## Task 2b — environment-space optimization, mask axis (`analysis/optenv_result.md`)

Greedy-pruned two mask environments from the LOO distractor ranking:

| environment | #tools | pass | tf | Δ vs base | tool_call |
|---|---|---|---|---|---|
| baseline | 14 | 14.35 | 10.85 | +0.00 | 131 |
| conservative (−3) | 11 | 14.46 | 10.44 | **+0.11** | 116 |
| aggressive (−5) | 9 | 14.79 | 12.49 | **+0.44** | 111 |

**Negative result (important).** Naive LOO-greedy mask optimization buys ~nothing
on accuracy: DrawBox alone showed +3.97 on removal, but removing the three
"removal-helps" tools together gave +0.11 (additive prediction ≈ +9). Cause:
(1) non-additivity — pruning one distractor just shifts spurious calls elsewhere;
(2) the tf column spans 10.44–12.49, a **±1.5 single-seed noise floor**, so the
LOO per-tool magnitudes (and any optimization from them) are mostly noise. The
contributor/distractor *grouping* may hold; the magnitudes do not.

**One robust signal: cost.** tool_call drops 131→116→111 (−15%) at unchanged
accuracy. Under a tax-aware objective `Acc − λ·calls` the pruned environment
strictly dominates — the accuracy axis is not maskable on this weak model, the
cost axis is.

## Next steps (recommended, not yet run)

1. **Seed everything first** — the ±1.5 tf noise floor means any environment
   optimization below ~1.5 AnsAcc (i.e. all of Task 2b's accuracy deltas) is
   currently an artifact. This gates the whole optimization program.
2. Get real Serper/Mathpix keys OR keep reporting de-confounded (tool-clean) ΔTool.
3. Add 2–3 seeds to A/B/C — current rescued(17) vs hurt(9) differ by 8 samples.
4. Stronger model (72B / API) to lift off the 82%-both-wrong floor; test whether
   C≈B and the contributor/distractor split survive at higher accuracy.
5. Environment search: seed-denoised LOO, or direct **joint**-mask search
   (evaluate candidate environments, don't sum single-tool deltas); Φ vs m
   disambiguation (rewrite distractor descriptions instead of removing them);
   C macro-tools to cut the call count further. Generators + manifests ready.
