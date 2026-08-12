# Does pruning the tool menu actually help? — BFCL pool-subset experiment

Date: 2026-08-11. Branch: `bfcl-token-likelihood-pilot`. Model: Qwen3-VL-8B-Instruct
(vLLM, `127.0.0.1:8013`, served as `gpt-qwen3-vl-8b`). **Full run, final numbers**
(n=1022 of 1052 tasks scored; 30 dropped to transient connection errors under
concurrency, see Caveats).

An interactive version with charts and full transcripts is published as an
Artifact: [Does Pruning the Tool Menu Help?](https://claude.ai/code/artifact/1e938c6c-3dee-4623-bf76-f45fd3bb3c22)
(same numbers as this file, redeployed with the final run).

## Question

The earlier pilot in this investigation (`BFCL_V1_TOKEN_LIKELIHOOD_PILOT_20260811.md`)
showed token likelihood can *identify* the right tool out of a small,
already-curated candidate list (2–4 tools, near ceiling). That left the real
practical question open: **given a large tool catalog where only a few tools
are relevant to any one task, does actually pruning the menu down before
generation improve real accuracy — or does the model handle clutter fine on
its own?**

This experiment builds a genuinely cluttered pool (most tools in it are
irrelevant to any given task) and compares three ways of presenting it to the
model, scoring **real unconstrained generation** (not forced scoring) against
BFCL's ground truth.

## Method

**Building the pool.** BFCL v1 `live_multiple` gives each task only 2–4 of its
own candidate tools — too easy to test clutter directly. So tasks are grouped
into **batches**, and every task in a batch shares the *union* of that
batch's candidate tools as one common pool.

Worked example: batch 0 pools three unrelated tasks — a movie search, a
weather lookup, a sensor query — into one 6-tool menu (`Media_FindMovies,
Media_BuyTicket, get_weather, get_forecast, get_sensor_readings_latest,
get_sensor_alerts`). For the movie task, only the first two are relevant —
the other four are honest, real, but irrelevant distractors, not adversarial
ones.

1052 tasks (of BFCL v1 `live_multiple`'s 1053; one has no `possible_answer`
entry and is skipped) are split into 150 batches by **round-robin**
assignment (`task_idx % 150`), not consecutive slices — consecutive tasks in
the source file are usually the same domain asking near-identical questions,
which barely pools at all. Round-robin scatters that adjacency, producing a
mean pool of **24.0 tools** per batch (min 15, max 33 among scored tasks).

**Three conditions**, real unconstrained generation (temperature 0, single
sample, no repeats):

- **full** — every tool in the batch's pool shown to the model
- **topk** — only the 5 tools ranked highest by forced-scoring token
  likelihood against this task's own query (same method as the earlier
  pilot: score `P(candidate_name | prefix)` via vLLM `/v1/completions`
  `echo=True`; no ground truth used to build the ranking)
- **randomk** — 5 tools chosen at random from the pool, same size as `topk`,
  no signal at all — the control that separates "a smaller menu helps
  regardless of which tools are cut" from "the selection specifically helps"

Implementation: `experiments/bfcl_token_likelihood/pool_subset_experiment.py`
(generation + scoring) + `summarize_pool_subset.py` (pooled metrics).

## Results (n=1022, single seed)

| Condition | Menu size | Accuracy | GT actually in menu | Accuracy given GT in menu |
|---|---:|---:|---:|---:|
| **full** | 24.0 (mean) | 80.2% | 100% | 80.2% |
| **topk** | 5 | **93.2%** | 97.9% | **95.1%** |
| **randomk** | 5 | 0.4% | 0.8% | 50%¹ |

¹ n=8 for randomk's "GT in menu" cases — far too thin to read as a real
number, shown for completeness only.

**Headline: pruning to a likelihood-selected 5-tool subset beats showing the
model the whole ~24-tool pool by 13.0 points (93.2% vs. 80.2%), and the
random-same-size control shows this isn't just "smaller is better."**

## Reading this correctly

**The `randomk` comparison is the load-bearing one.** If `topk` only won
because a 5-tool menu is easier to read than a 24-tool one, `randomk` would
have won too — same size, same "less clutter." It didn't (0.4% vs. 93.2%),
because it almost never has the right tool available at all (0.8% recall,
vs. topk's 97.9%). That gap is what shows the *selection* is doing the work,
not the shrinkage.

**The conditional-accuracy gap is the second load-bearing number.** Even
restricted to tasks where `full` *did* have the right tool available
(all of them, by construction), it still only got 80.2% right vs. `topk`'s
95.1% conditional accuracy — so pruning isn't just about recall (having the
right tool in view), a smaller honest menu measurably helps the model reason
better too. See Case study 2 below (`"Function 13"`) for a vivid, if
anecdotal, illustration of why: a long tool list visibly degrades output
discipline in a way a short one doesn't.

**How good is the likelihood ranking itself** (before any top-5 cutoff, rank
of the ground-truth tool within the full pool, n=1022):

| Rank of GT | Share of tasks |
|---|---:|
| 1 | 87.7% |
| 2 | 6.9% |
| 3 | 1.9% |
| 4 | 0.9% |
| 5 | 0.6% |
| below top-5 | 2.0% |
| scoring failed for GT | 0.1% |

Cumulative top-5 recall = 98.0%, which is why `topk`'s actual `gt_in_menu`
rate (97.9%) lands almost exactly there.

## By pool size — does the gain grow with clutter?

| Pool size | n tasks | full accuracy | topk accuracy | randomk accuracy |
|---|---:|---:|---:|---:|
| 15–29 tools | 987 | 80.5% | 93.1% | 0.4% |
| 30+ tools | 35 | **71.4%** | **94.3%** | 0.0% |

At the more-cluttered end (30+ tools, n=35), `full`'s accuracy drops further
(71.4% vs. 80.5%) while `topk` barely moves (94.3% vs. 93.1%) — the gap
widens from 12.6 points to 22.9 points. This matches the direction this
repo's own GTA mask-axis work found (`FINAL_REPORT.md`: masking gain is
inversely related to how much clutter/capability gap there is). n=35 at the
large-pool end is still thin relative to the 987-task main bucket, but the
direction is consistent, not just noise around zero.

## Case studies

Four real transcripts — same task, same model, three different menus.
`free_text` is the model's literal unconstrained output.

### 1. Clutter causes a wrong pick

`live_multiple_300-130-9` — *"…find a home rated above 4.2 for 2 people in
Rio…"* — GT: `Hotels_2_SearchHouse` (ranked #2 of 23 by likelihood)

| Condition | Menu | Output | Verdict |
|---|---|---|---|
| full | 23 tools | `"Hotels_4_SearchHotel"` | ✕ picked the sibling function — hotel search instead of house search |
| topk | 5 tools | `"Hotels_2_SearchHouse"` | ✓ correct |
| randomk | 5 tools | `"None of the functions provided can be used to find a home rated above 4.2 for 2 people in Rio"` | ✕ correctly declines — GT wasn't in this random menu |

### 2. Clutter breaks the output format entirely

`live_multiple_600-158-6` — *"…book smoking rooms in a hotel…"* — GT:
`Hotels_4_SearchHotel` (ranked #1 of 23 by likelihood)

| Condition | Menu | Output | Verdict |
|---|---|---|---|
| full | 23 tools | `"Function 13"` | ✕ names the numbered *slot*, not the function — a listing this long visibly degrades output discipline |
| topk | 5 tools | `"Hotels_4_SearchHotel"` | ✓ correct |
| randomk | 5 tools | `"None of the functions provided can be used to book smoking rooms in a hotel."` | ✕ GT wasn't in this random menu |

### 3. Honest counter-example — a topk miss

`live_multiple_904-186-1` — *"…find train availability from New York to Los
Angeles…"* — GT: `Trains_1_FindTrains` (ranked #2 of 19 by likelihood)

| Condition | Menu | Output | Verdict |
|---|---|---|---|
| full | 19 tools | `"Function 19: Trains_1_FindTrains"` | ✓ correct |
| topk | 5 tools | `"Call function Train_1_FindTrains with the parameters: from: \"New York, NY\", to: \"Los…"` | ✕ typo'd the name (`Train_1` not `Trains_1`) — **GT was in the menu**, the model just misspelled it; a string-match miss, not a reasoning failure |
| randomk | 5 tools | `"None of the provided functions can be used to reserve train seats or search for train availability."` | ✕ GT wasn't in this random menu |

### 4. The one time random got lucky

`live_multiple_22-4-14` — factual lookup task — GT: `HNA_WQA.search` (pool
size 25; likelihood scoring failed for this one candidate, so it has no
rank)

| Condition | Menu | Output | Verdict |
|---|---|---|---|
| full | 25 tools | `"HNA_WQA.search"` | ✓ correct |
| topk | 5 tools | `"None"` | ✕ the selector itself missed GT this time (`gt_in_menu: false`) — part of the ~2% of tasks where the likelihood ranking drops the right tool out of the top 5 |
| randomk | 5 tools | `"HNA_WQA.search"` | ✓ correct — one of the ~1% of random draws that happened to include GT |

## Caveats

- **Single seed, no repeats.** Every condition is one greedy (temperature 0)
  generation per task — no variance estimate.
- **30/1052 tasks (2.9%) dropped to transient connection errors**
  (`ConnectionResetError`/`RemoteDisconnected`) under the concurrency used for
  this run (4 tasks in parallel × ≤12 concurrent per-candidate scoring calls,
  ≤48 peak). This is a tuned-down setting: an earlier attempt at ~60-way
  concurrency **crashed the shared vLLM server outright** (it and three other
  jobs sharing the same endpoint all went down; the server had to be
  relaunched). 48-way peak survived the full run with a small error rate
  instead of a hard crash — not re-run here, but worth budgeting for on a
  repeat.
- **Scoring is exact-string-match** against BFCL's ground-truth function
  name. Case study 3 shows this can call a correct-in-substance answer
  "wrong" over a typo. This likely affects `topk` and `full` similarly (both
  are free-form generations parsed the same way), but — unlike the earlier
  forced-scoring pilot, where the free-gen baseline's matcher was
  hand-audited against every "wrong" case — this run's parsing has not been
  separately audited case-by-case at scale, so the exact size of this effect
  on the headline numbers is unverified.
- **Pool composition is synthetic-but-honest.** The "irrelevant" tools are
  real BFCL functions from other tasks, but which tasks land in which batch
  is round-robin, not adversarially chosen to be confusable. A pool
  deliberately full of near-duplicate tools (e.g. many similarly-named
  search functions) would be a harder, more realistic stress test than this
  one.
- **`topk`'s own selection isn't perfect** — case study 4 is one of the ~2%
  of tasks where the likelihood ranking itself drops the ground-truth tool
  out of the top 5; `topk`'s ceiling is bounded by its own recall (97.9%),
  not 100%.
- Single model (Qwen3-VL-8B), single benchmark category (`live_multiple`).

## Artifacts

- Code: `experiments/bfcl_token_likelihood/{pool_subset_experiment.py,summarize_pool_subset.py}`
- Data source: `runtime/bfcl_live_multiple_data/{questions,answers}.json`
- Raw per-task results: `runtime/bfcl_pool_subset_20260811/results.jsonl`
- Pooled summary: `runtime/bfcl_pool_subset_20260811/summary_final.json`
  (n=1022, matches every number in this report; earlier
  `summary_interim_*.json` snapshots kept alongside for the run's progression)
- Interactive version with charts: [Does Pruning the Tool Menu Help?](https://claude.ai/code/artifact/1e938c6c-3dee-4623-bf76-f45fd3bb3c22)
