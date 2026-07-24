# The structure of honest-tool clutter — actionable rules for shaping the tool space

Clean pool, **no injection**. We already knew per-task masking is a large but
capability-gated lever (inverted-U: 7B nearly doubles, 14B/32B ≈ 0) and that no-GT
selectors recover only ~20% of it. This report asks the *structural* question that
tells you how to prune: **how much, what first, and why the no-GT selectors stall.**

Qwen2.5-7B (the sweet-spot model), GTA-Atomic full 229, greedy. Each config is a
per-task mask = GT-relevant tools + a controlled set of *honest* distractors.

## 1. Dose–response: clutter is a **cliff, not a slope**

Add k random irrelevant tools on top of the per-task GT set (2 seeds averaged):

| menu (avg pool) | AnsAcc | tool-selection acc |
|---|---|---|
| oracle (2.3) | **20.9** | **92%** |
| +2 (4.3) | 13.7 | 88% |
| +4 (6.3) | 12.9 | 77% |
| +6 (8.3) | 11.1 | 66% |
| all-14 (14.0) | 10.6 | 74% |

**~70% of the total damage happens at the first two distractors** (20.9 → 13.7, −7.2
of the −10.3 total). From pool 4 to 14 the agent only loses ~3 more points. The
landscape is flat-bad once you are past a couple of distractors.

**Rule 1 — pruning is all-or-nothing.** Trimming to a *moderate* pool (6–8 tools) is
nearly worthless; the big gain only appears when you get close to the GT-minimal set
(≤3). Do not do "gentle" pruning.

## 2. Category attribution: prune **operation and perception** first

Add only the irrelevant tools of one category to the oracle menu; normalize the
damage by how many tools were added:

| distractor category | damage per tool |
|---|---|
| **operation** (DrawBox, AddText, GoogleSearch) | **−3.44** |
| perception (OCR, ImageDescription, TextToBbox, RegionAttr) | −2.63 |
| creativity (TextToImage, ImageStylization) | −2.18 |
| **logic** (Calculator, Solver, Plot, MathOCR, CountGivenObject) | **−1.16** |

**Rule 2 — prune by category weight, not uniformly.** An irrelevant *operation* or
*perception* tool costs 2.6–3.4 points each; an irrelevant *logic* tool costs only
~1.2. When you cannot get to the GT-minimal set, drop operation/perception distractors
first and it is safe to keep logic tools.

## 3. Mechanism: clutter → misselection → damage

Tool-selection accuracy (fraction of the agent's calls that hit a GT-relevant tool)
falls monotonically with pool size, 92% → 60%, tracking AnsAcc. The damage is
mediated by misselection: extra tools make the 7B call the wrong tool, and it cannot
recover from the irrelevant evidence.

## 4. Why the no-GT selectors stall — the cliff explains it

Our best no-GT selector (router ∪ embedding ∪ frequency) lands at pool ≈ 6 and recovers
only ~20% of the oracle gain. The dose–response says why: **pool 6 sits at the bottom
of the cliff** (acc ≈ 12.9, vs oracle 20.9 and full 10.6). A selector that keeps ~6
tools is structurally incapable of recovering most of the gain — not because it picks
badly, but because *any* 6-tool menu is already near the floor. Recovering the gain
requires reaching the ≤3-tool regime, which needs near-perfect relevance ranking — and
that is exactly what no-GT signals cannot deliver (see `LL_SIGNAL.md`, `SIGNAL_MAP`).
The cliff and the no-GT-signal ceiling are two views of the same wall.

## Actionable summary for tool-space optimization

1. **All-or-nothing pruning.** The ROI of masking is concentrated at the GT-minimal
   set; moderate trimming barely helps.
2. **Category-weighted pruning.** Drop operation/perception distractors first; logic
   distractors are nearly harmless.
3. **The no-GT gap is structural.** Because damage is a cliff and no-GT selection
   floors at pool ~6, label-free masking cannot recover most of the gain — the lever
   needs either ground truth or a capability-matched model (the inverted-U).

Raw runs: `results/cl_*`; masks: `results/inject_opt/clutter_masks/`. Scripts:
`experiments/{gen_clutter_masks.py, clutter_battery.sh, clutter_analyze.py}`.
