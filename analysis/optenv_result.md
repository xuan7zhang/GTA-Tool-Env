# Environment-space optimization (mask axis) — pilot result

**Question.** Can we optimize the tool environment E=(m,C,Φ) by pruning the
net-distractor tools that leave-one-out (LOO) ΔTool flagged (DrawBox +3.97,
ImageStylization +2.78, GoogleSearch +2.63, …)? Objective: maximize
tool-contributed capability, not raw accuracy.

**Setup.** Two greedy-pruned mask environments vs the full 14-tool baseline,
end mode, seed 0. Toolmeta pruned + proxy mask applied.
(`envgen/variants/optenv/`, `configs/manifest_optenv.jsonl`.)

| environment | #tools | pass AnsAcc | tf AnsAcc | ΔTool | Δ vs baseline | tool_call |
|---|---|---|---|---|---|---|
| baseline (all 14) | 14 | 14.35 | 10.85 | 3.50 | +0.00 | 131 |
| conservative (−GoogleSearch,DrawBox,ImageStylization) | 11 | 14.46 | 10.44 | 4.02 | +0.11 | 116 |
| aggressive (−those +Solver,TextToBbox) | 9 | 14.79 | 12.49 | 2.30 | +0.44 | 111 |

## Finding: naive LOO-greedy mask optimization buys ~nothing (accuracy)

The single-tool LOO removal gains do **not** add. Removing DrawBox alone lifted
accuracy +3.97 at seed 0; removing all three "removal-helps" tools together
lifted it +0.11. Predicted (additive) ≈ +9; observed +0.11. Two reasons, both
load-bearing:

1. **Non-additivity.** Pruning one distractor just shifts the agent's spurious
   calls to another tool; total wasted turns barely move. Tool effects interact;
   greedy single-tool ΔTool does not compound.
2. **Noise dominates.** The tf column should be constant (tool_free returns
   "unavailable" for any pool) yet spans 10.44–12.49 — a **±1.5 single-seed
   noise floor**. So the LOO Δvs-baseline values (and the ΔTool spread 2.30–4.02)
   are mostly within noise; the "biggest" LOO tool (DrawBox +3.97) was likely the
   biggest noise, not the biggest signal.

This retroactively caveats `deltatool_table.md`: the contributor/distractor
*grouping* may hold, but per-tool magnitudes and any optimization read from them
are unreliable at seed 0.

## The one robust signal: cost, not accuracy

tool_call decreases monotonically with pruning: **131 → 116 → 111 (−15%)** at
unchanged accuracy. This is not noise (consistent direction, clear mechanism:
smaller pool → fewer spurious calls). Under a **tax-aware objective
`Acc − λ·tool_calls`** the aggressive environment strictly dominates the full
pool (equal accuracy, 15% cheaper) — exactly the Tool-Use-Tax point that raw
accuracy hides.

## Lessons for optimizing over environment space

1. LOO is a poor gradient estimator here: single-seed noise swamps the signal
   and effects are non-additive, so greedy pruning does not compound.
2. Fix before any further mask search: **seed the runs to establish the noise
   floor** — any environment optimization below ~1.5 AnsAcc is currently an
   artifact.
3. Then either seed-denoise LOO, or search **joint** masks directly (evaluate
   candidate environments, don't sum single-tool deltas).
4. On this weak base model the accuracy axis is not optimizable by masking; the
   **cost axis is** — report `Acc − λ·calls`, where the robust −15% call
   reduction actually counts.
