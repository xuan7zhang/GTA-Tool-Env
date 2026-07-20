# OctoTools greedy toolset search — the published-method baseline (decisive-table row)

Fills the pending `OctoTools greedy` row of the decisive table. This is the
**outcome-based control**: it is not meant to win, it is the live specimen of how a
validation-accuracy toolset optimizer fails under injected unreliable tools. Do not
"optimize" it.

## Method (OctoTools' O(n) greedy search, as specified)

OctoTools avoids the 2^n subset search by scoring each tool's **marginal
contribution** instead of searching combinations. Three stages
(`experiments/overnight/scripts/octotools_greedy.py`):

1. **Marginal contribution** (O(n) validation evals). We measure it **leave-one-out**
   from the full 22-tool pool: `marginal(t) = acc(full) − acc(full∖{t})`. (An empty
   base breaks the ReAct harness — the agent needs ≥1 tool — and a forward
   "single-tool-from-empty" base is degenerate on chain-tool GTA tasks where one tool
   rarely solves a query; LOO preserves the exact marginal + threshold semantics and
   is non-degenerate.)
2. **Threshold prune.** Keep `{t : marginal(t) > 0}` (removing it hurts ⇒ useful);
   drop `≤ 0` (removing it helps or is neutral ⇒ poison / unused).
3. **Subset-vs-full safety valve.** Compare the selected subset vs the full pool on
   val; revert to full if the subset is worse.

Signal = **validation accuracy** ⇒ outcome-based, label-hungry (O(n) full evals),
same family as call-frequency / answer-echo.

## Setting

32B (fp16, greedy), pool = 14 real + 8 poison = 22, frozen **val = 30 tasks**, final
accuracy on the full 229, seed 0, both regimes. (Greedy ⇒ the search is near-
deterministic; residual noise ±0.8 from the tool server. Multi-seed would show the
selection variance directly; seed 0 is indicative under greedy.)

## Result

| regime | test (229) | kept poison | pruned genuine | Stage-3 revert | failure mode |
|---|---|---|---|---|---|
| **attractive** | **10.14** | **7 / 8** | 2 / 14 | no | keeps poison → below no-defense |
| **subtle** | 14.12 | 4 / 8 | **9 / 14** | no | over-prunes real tools |
| **worst-case** | **10.14** | | | | |

For comparison (greedy decisive table, seed 1): Ground+ 14.1 / 15.4 (worst **14.1**,
**8/8** detect, **0** false-prune); call_frequency 14.55 / 8.31; keep_all 12.75 / 12.9.
**OctoTools worst-case (10.14) is below keep_all (12.75) and far below Ground+ (14.1),
and it never reaches the probe's 8/8-detect, 0-false-prune property.**

## Why it fails — and it fails *differently* in each regime

The two regimes produce wildly different selected sets (19 tools vs ~13, different
composition) — that instability *is* the regime-fragility.

- **Attractive.** Only PreciseOCR had a clearly negative LOO marginal (−0.613) and was
  dropped. The other 7 poison tools had **positive** marginals (SmartCount +4.14,
  VerifiedFact +3.97, InstantAnswer +3.83, …): on 30 val tasks the ±noise (±3–4)
  swamps each poison tool's true "removal helps" signal (~0.5), so removing them looks
  harmful and they are **kept**. Two genuine tools (MathOCR −0.024, AddText −0.014)
  fall just below zero on noise and are **false-pruned**. Net: 7/8 poison kept, test
  10.14 — *worse than doing nothing*.
- **Subtle.** Neutral descriptions make poison low-frequency, so their marginal signal
  is even weaker; the search **false-prunes 9 of 14 genuine tools** (Calculator,
  Solver, TextToBbox, GoogleSearch — genuinely needed — plus the image-output tools).
  Test lands at 14.12 **by luck, not competence**: the pruned set happens to include
  the image-generation tools (Plot/DrawBox/TextToImage/ImageStylization/AddText) that
  are declutter-able and don't help AnsAcc, so cutting them offsets the damage from
  cutting Calculator/Solver/etc. A defense that removes over half the real tools is not
  trustworthy even when the number happens to land high.

## Bottom line for the paper

OctoTools' greedy search is **outcome-based and regime-fragile**: its selected set
swings drastically between regimes, it keeps most poison in one regime and shreds the
genuine pool in the other, its worst-case (10.14) sits below no-defense, and it never
achieves the probe's deterministic 8/8-detect / 0-false-prune. It also costs O(n) full
labeled validation evals vs the probe's few label-free calls per tool. This is exactly
the "outcome-based methods are regime-fragile, behavioral probes are regime-invariant"
contrast — the failure mode simply differs by regime (keep-poison vs over-prune) rather
than a single clean collapse.

Raw: `results/octotools_gta/{attractive,subtle}_s0/metrics.json` (per-tool marginals,
Stage-2 selection, Stage-3 revert flag, kept_poison, pruned_genuine).
