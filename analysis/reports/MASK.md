# Mask axis (m) — results report

**Question.** In E = (m, C, Φ), can the tool **mask** be optimized to raise agent
accuracy — and can that be done *without ground truth*? Testbed: GTA-Atomic (229
tasks, 14 tools), Qwen2.5 3B/7B/14B, proxy-in-front-of-toolserver, AnsAcc (end).

> Status: 7B complete (3 seeds); 3B in progress; 14B and the online signal-ablation
> queued (running as of this writing). Numbers marked *(nX)* give the seed count.
> A ±2.5–3 AnsAcc run-to-run noise floor applies throughout — the load-bearing
> evidence is the deterministic mechanism metrics (selection accuracy, all-kept),
> not the noisy AnsAcc means.

---

## 1. The mask mechanism (what "mask" means here)

Each task's agent menu is the dict `action_executor.actions`; masking = which tools
are in it. Three levers set it (`models/lagent.py`):
`menu = (per-task resources ∩ deployed) + GTA_EXTRA_TOOLS − GTA_HIDE_TOOLS`,
optionally restricted per-task by `GTA_PREDICTED_MASK`. GTA ships a **per-task
relevance annotation** (each task lists its 1–4 needed tools; avg **2.3 of 14**), so
`per_task` = the **oracle** mask and `full_everywhere` (force all 14) = mask OFF.

## 2. Oracle mask helps — and it is a capability effect

`per_task` (oracle) vs `full_everywhere` (mask off), full 229, AnsAcc:

| model | full (mask off) | oracle | Δ (oracle gain) |
|---|---|---|---|
| 7B | 11.5 *(n3)* | 15.9 *(n3)* | **+4.4** |
| 3B | 10.6 *(n3)* | 11.8 *(n2)* | +1.2 (noisy) |
| 14B | *(queued)* | | |

Direction matches the earlier multi-model sweep (weak models gain most; 32B ≈ 0):
strong models ignore honest clutter, weak models don't. **Existence result: the
per-task mask carries real, capability-dependent signal.** Note the oracle gain is
noisy (7B seed-1 was +9.5, 3-seed mean +4.4) — see the noise caveat.

## 3. Mechanism — mask raises tool-**selection accuracy**

Why does a smaller menu help? The 7B/3B are *bad at picking tools*, so clutter
causes misselection. Per tool call, does the agent hit a GT-relevant tool?

| model | full (14 tools) | oracle mask |
|---|---|---|
| 3B | **44%** (263/593 calls) | 54% |
| 7B | **59%** (69/116) | **82%** |

Mask off → the 7B misselects 41% of calls; oracle mask → 82% hit rate. 3B is worse
on both (44%) and calls tools **5× more often** (593 vs 116) — weak models are both
trigger-happy and inaccurate. For reference, GTA's own step-mode **ToolAcc reproduced
at 33.4** (leaderboard 32.85) — the 7B strictly selects the GT tool only ~⅓ of steps.
Low tool-selection accuracy is *why* the mask axis exists.

## 4. Global (pool-wide) mask optimization is NULL

Earlier arc: leave-one-out over the *global* 14-tool pool (same mask for every task)
did **not** move accuracy — apparent single-seed effects didn't replicate across
seeds/bootstrap (`FINAL_REPORT.md`). Which tools are in the *pool* is inert; only the
*per-task* mask carries signal. So the real question is per-task selection.

## 5. No-GT per-task selector — can we predict the oracle mask?

Honest-clutter relevance is **query-dependent** (unlike poison, which is tool-
intrinsic input-degeneracy — the grounding selector's signal does **not** apply
here). We combine three no-GT signals; GT is used only to *score* them.

**Offline coverage vs GT** (all-relevant-kept = fraction of tasks whose every GT tool
is kept; pool = avg menu size, oracle 2.3 / full 14):

| selector | all-kept | pool |
|---|---|---|
| embedding (mpnet query↔tool, top-4) | 21% | 4.0 |
| 7B LLM router | 22% | 1.9 |
| 14B LLM router | 40% | 3.1 |
| frequency prior {OCR, ImageDescription} | 11% | 2.0 |
| **hybridA** = 7B-router ∪ emb ∪ freq | **80%** | 6.0 |
| **hybridStrong** = 14B-router ∪ emb ∪ freq | **93%** | 6.7 |

Single signals are weak (all-kept ~20%); **the frequency prior is the highest-leverage
recall booster** (router 22%→64% when unioned with it) because {OCR, ImageDescription}
covers the dominant perception tools the others miss. Stronger router → better routing
(7B 22% → 14B 40%). The hybrid reaches 80–93% all-kept at pool 6.

## 6. Online recovery — how much oracle gain does the no-GT mask recover?

`recovery = (predicted − full) / (oracle − full)`. Predicted mask applied per-task via
`GTA_PREDICTED_MASK`; same session as full/oracle (fair).

| model | full | oracle | predicted (hybridA) | recovery |
|---|---|---|---|---|
| 7B | 11.5 | 15.9 | 13.5 *(n3)* | **~45%** |
| 3B | 10.6 | 11.8 | 14.6 *(n2)* | predicted **> oracle** (noisy) |
| 14B | *(queued)* | | | |

- **7B: the no-GT mask recovers ~45% of the oracle gain** — a real but partial lever.
  It lifts selection accuracy 59% → 67% (vs oracle's 82%), consistent with recovering
  part, not all, of the gain (all-kept 80% and pool 6 ≫ oracle 2.3 → less decluttering).
- **3B: predicted (14.6) exceeds both full (10.6) and oracle (11.8)** — the 6-tool
  hybrid helps the 3B *more* than its 2.3-tool GT mask. Intriguing (the extra common
  tools may be genuinely useful to a very weak model) but oracle gain is tiny (+1.2,
  n=2) so treat as provisional.
- **Counterintuitive:** the higher-coverage `hybridStrong` (93%) recovered *less* than
  `hybridA` (80%) on 7B seed-averaged — coverage doesn't monotonically buy recovery
  (bigger pool = less decluttering). The signal ablation (running) isolates this.

## 7. Signal ablation (running)

Leave-one-out over {router, emb, freq} on 7B, online, to see which signal drives
*recovery* (not just coverage) — in particular whether router/emb add anything beyond
the trivial frequency prior + pruning. **Offline** coverage already shows freq is the
recall backbone; the online marginal-contribution table lands when the run completes.

## 8. Honest limitations / open

- AnsAcc noise floor ±2.5–3 → per-model oracle gains are noisy (n=2–3); the robust
  signals are selection accuracy and all-kept.
- The no-GT selector **partially** recovers the oracle gain (~45% on 7B); it does not
  approach oracle, because predicting per-task relevance without GT is hard (all-kept
  caps at 80–93%, pool stays at 6 vs oracle 2.3).
- These runs are under the JsonParser tool-fidelity fix; earlier oracle-gain numbers
  (pre-fix) may differ.
- Open: does recovery scale with capability (14B pending)? Which signal drives online
  recovery (ablation pending)? Can a stronger router (32B) or a learned selector close
  the gap to oracle?

## Bottom line

The per-task mask carries real, capability-dependent signal (oracle gain, driven by a
measurable jump in tool-selection accuracy 59%→82%), while the *global* pool mask is
inert. A no-GT selector built from query↔tool routing + embedding + a frequency prior
**recovers ~45% of the oracle gain on the 7B** — a genuine but partial method, the
first no-GT lever on the honest-clutter mask (the grounding selector only worked on
injected poison). Closing the remaining gap is the open problem.

Artifacts: `experiments/{relevance_selector,router_selector,build_hybrid,pertask_score}.py`,
`predmask_curve.sh`, `mask_overnight.sh`, `abl_exp.sh`; masks in
`results/inject_opt/{mask_hybridA,mask_hybridStrong,abl_*}.json`; runs `results/pm{3,7,14}b_*`.
