# GTA, read_arith: does the leave-one-out score measure necessity or text volume?

## Why this exists

LOTS ties a size-matched random mask on GTA. Aggregated over all 229 tasks the
selector looks useless, but the aggregate mixes three populations that should
never have been averaged, and taking them apart changes the conclusion.

Per task type, hand-labelled from the question text alone (no gold annotation
read), Qwen2.5-7B, 3 greedy replicates, scored on the 156 tasks that carry an
exact textual reference:

| type | n | no tools | full menu | oracle | oracle − full |
|---|---|---|---|---|---|
| **read_arith** | **65** | 3.1 | 4.6 | **24.1** | **+19.5** |
| web_fact | 35 | 1.9 | 4.8 | 13.3 | +8.6 |
| count_arith | 30 | 7.8 | 6.7 | 10.0 | +3.3 |
| math_eq | 17 | 19.6 | 23.5 | 15.7 | −7.8 |
| visual_qa | 6 | 22.2 | 22.2 | 16.7 | −5.6 |

Three things were being averaged together:

1. 56 of 229 tasks (image generation, chart conversion, localization) need
   image-output tools, which a text-likelihood proxy cannot score at all.
2. On `math_eq` and `visual_qa` the oracle mask is *worse* than the full menu,
   so any pruning there subtracts.
3. `read_arith`, the largest type, hides a 19.5-point gap that the other two
   groups cancel out.

`read_arith` is the one place on GTA where the method's preconditions hold: the
gap is large, and the tools that close it (OCR, Calculator) are both scoreable
from text.

## The defect this experiment targets

Cluster-mean marginal on `read_arith` (n=66), highest first:

```
1. RegionAttributeDescription  +0.8652   gold in 0/66 tasks
2. OCR                         +0.6839   gold in 65/66
3. ImageDescription            +0.1373
4. TextToBbox                  +0.0596
5. Calculator                  +0.0500   gold in 44/66
```

The tool this type never needs ranks first; the tool it needs in two thirds of
tasks ranks fifth. top-2 recovers 0.523 of the gold tools, and only top-6
reaches 0.903 — by which point the mask is back in the size regime where random
matches it.

The hypothesis for why: `I(t) = L(y_S|S) − L(y_S|S\{t})` is a mean over all
tokens of the answer. `Calculator` contributes a short number; deleting it
barely moves a per-token mean. `RegionAttributeDescription` contributes a
paragraph; deleting it changes the surrounding language statistics everywhere.
**The score is dominated by how much text a tool injects, not by whether the
answer depends on it.** The same mechanism explains two earlier results: on a
22-tool pool the deceptive tools, which emit confident answer-shaped strings,
were preferentially kept (AUROC gold vs poison only 0.561), and across eight
task types the top-6 masks were near-identical (mean pairwise Jaccard 0.648,
with OCR / ImageDescription / RegionAttributeDescription in all eight).

## The experiment

Restrict the scored span to the tokens of the answer that carry the content,
rather than averaging over the whole answer.

* **A. span-restricted score.** Recompute `I(t)` over only the numeric and
  entity spans of `y_S` (for `read_arith` the answer is a price or a count).
  A tool that determines those tokens should now outrank a tool that only
  changes the prose around them.
* **B. deploy it.** Build a top-2 mask on `read_arith` from the new ranking and
  run it against the anchors already measured on the same stack: full menu 4.6,
  oracle 24.1, and a size-matched random 2-tool mask.

Read on the 65 `read_arith` tasks only, since that is the population the
preconditions were checked on.

## What would kill it

* If `Calculator` does not rise into the top 2 under the span-restricted score,
  the text-volume story is wrong and the ranking fails for another reason;
  stop and report the null rather than sweeping the span definition.
* If it does rise but the top-2 mask does not beat the size-matched random
  2-tool mask, then tool recovery still does not convert on GTA, which is the
  same conclusion the aggregate gave and should be reported as such.
* One seed decides nothing here. The differences that matter (19.5 points on
  65 tasks) are large, but a 2-3 point result needs replicates: per-arm spread
  across 3 greedy replicates on this benchmark has been 1-2 points.

## Provenance

* `manual_task_types.json` — the hand assignment of all 229 tasks to the eight
  types, made from the question text only.
* `gta_recount.py` — scores an arm on the 156 exactly-scoreable tasks. The
  shipped `answer_acc` divides by all 229 while crediting only the 172 that
  carry a textual reference, which flatters any method that drops the five
  image-output tools; this one states its denominator.
* `cache_poison_outputs.py`, `build_loo_mask_p22.py` — the 22-tool padded-pool
  study that established that identity pays on GTA only once the menu contains
  actively misleading tools and the mask is small (oracle 16.50 vs a
  size-matched random 11.76 vs full 11.73).

---

## Outcome (run on a second machine, 3 greedy replicates, 65 read_arith tasks)

**Gate 1 passed.** Under the span-restricted score `Calculator` moves from rank
5 (+0.05, whole-answer mean) to rank 2 (+1.128). The text-volume diagnosis was
correct: the whole-answer mean was distorted by how much text a tool injects,
and restricting the span removes that specific distortion.

**Gate 2 failed, so the result is null.**

| arm | replicates | mean |
|---|---|---|
| span-LOTS top-2 | 4.62 / 3.08 / 3.08 | 3.59% |
| full menu | 6.15 / 4.62 / 7.69 | 6.15% |
| size-matched random top-2 | 4.62 / 6.15 / 3.08 | 4.62% |
| oracle | 18.46 / 23.08 / 16.92 | **19.49%** |

Gold recall of the new ranking: top-2 35.51% micro / 33.85% macro; top-4 81.88%
micro / 86.15% macro.

Read the effect sizes before the signs: one point is 0.65 tasks here, so the
three non-oracle arms sit within a task of each other and none of them is
distinguishable. What is not in doubt is the gap they all fail to close, the
oracle's 19.49% against a 6.15% full menu.

### A protocol error this exposed, present from the start

The span top-2 is `GoogleSearch + Calculator`, and `OCR` — gold in 65 of 66
tasks — falls outside it, which is why top-2 recall is only 35.5%. The machine
has no SERPER key, so the arm attempted GoogleSearch 44 times across three runs
and got errors; the arm effectively ran one working tool and one error source.

This is not a bad run to be discarded. It is a design fault in this study on
both machines: **the candidate set contained a tool the deployment cannot
execute.** The original protocol masked `GoogleSearch` and `MathOCR` through the
proxy for every arm, while the cached outputs used to compute the marginals
still contained real GoogleSearch results — whose text is full of prices, which
a span-restricted score weights heavily. The selector was free to spend a slot
on a tool it could never use.

Restricting the candidate set to executable tools is a correction, not a knob:
it removes an option that was never deployable. It must be declared before the
rerun and reported whichever way it comes out, and it does not retract this
null unless it changes the outcome. Given top-4 recall is already 81.88%, the
informative rerun is top-4 against a size-matched random 4, not another attempt
at top-2.
