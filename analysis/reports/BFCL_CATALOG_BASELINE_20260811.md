# Catalog-scale baseline: is picking a tool out of the full 457-tool library actually hard?

Date: 2026-08-11. Branch: `bfcl-token-likelihood-pilot`. Model: Qwen3-VL-8B-Instruct
(vLLM, `http://127.0.0.1:8013/v1`, served-model-name `gpt-qwen3-vl-8b` — same
model as every other experiment in this investigation, unchanged).

## Question

Every earlier experiment in this investigation pruned a **small, artificially
constructed** pool (≤24–37 tools) — either a round-robin batch of a handful
of tasks' candidates (pool-subset experiment) or one DBSCAN cluster's own
tool set (cluster-subspace experiment). Neither tests the thing actually
motivating "optimize the tool space": **given the model access to the real,
full tool catalog (457 distinct tools across all of BFCL v1 `live_multiple`),
is finding the right one actually hard, and does likelihood-based filtering
help?**

## Exact settings

| Setting | Value |
|---|---|
| Model | Qwen3-VL-8B-Instruct |
| Serving | vLLM, OpenAI-compatible, `http://127.0.0.1:8013/v1`, served-model-name `gpt-qwen3-vl-8b` |
| Data source | BFCL v1 `live_multiple` (HF `gorilla-llm/Berkeley-Function-Calling-Leaderboard`), all 1052 answerable tasks used to build the catalog |
| Full catalog size | **457** distinct function names (union of every candidate tool ever shown across all 1052 tasks) |
| Pool size per task (`N`) | **40** tools — see "Why 40, not 50" below |
| Pool construction | The task's own ground-truth tool **forced into the draw**, plus `N-1=39` tools sampled uniformly at random (`random.Random(seed).sample`, no replacement) from the other 456 catalog tools, order shuffled |
| Seed | 1 seed per task (`seeds_per_task=1`), seed value = `0` |
| Task sample | 30 tasks, taken by fixed stride across the 1052-task list (`all_tasks[::stride][:30]`, `stride = 1052 // 30 = 35`) — spread across the dataset, not from one domain/cluster |
| Selector (`likelihood_topk`) | Forced-scoring token likelihood (identical method to every other experiment: `POST /v1/completions`, `echo=True, max_tokens=0, logprobs=20`, mean per-token logprob over the candidate-name span), ranked over the same 40-tool draw, top-`K=5` kept — **no ground truth used to build it** |
| Generation params | `POST /v1/completions`, `temperature=0.0, stop=["\n"], max_tokens=24` |
| Concurrency | Two flat `ThreadPoolExecutor` phases (scoring, then generation) — no nested per-item executors (an earlier version of this script nested pools and both under- and over-subscribed the server; see Caveats) — `scoring_workers=10, gen_workers=6` for this run |
| Runtime | 10m48s for 30 tasks (1200 scoring calls + 60 generation calls) — much slower than earlier same-scale experiments; see Caveats |
| Code | `experiments/bfcl_token_likelihood/catalog_baseline_experiment.py` |

### Why 40, not 50

The user's original request was N=50. Tested directly against the live
server: a 50-tool prompt (real tool schemas, name+description+parameters)
already **exceeds this model's 8192-token context window** —
`POST /v1/completions` returns a 400 (`"prompt contains at least 8192 input
tokens"`). Measured sizes for several `N`:

| N tools | Prompt tokens |
|---:|---:|
| 50 | ≥8193 (**overflow, rejected**) |
| 45 | 7587 |
| 40 | **6473** |
| 35 | 6051 |
| 30 | 5048 |

N=40 was picked as the largest size with a comfortable buffer (~1700 tokens)
for longer-than-average queries and the generation itself, given tool
description lengths vary across the catalog and a different random draw
could land on longer-than-average descriptions.

**457 tools cannot be shown to this model in one prompt at all** — even 50
already overflows, and 457 would need roughly 8x that budget. This is a hard
constraint of the model/serving config, not a design choice; see the
`analysis/reports/BFCL_TOOL_SPACE_CLUSTERING_20260811.md` and prior
conversation turns for the fuller discussion of why "just show it
everything" is not literally executable at this model's context length —
which is itself evidence for why tool-space optimization has to exist.

## Results (n=30, single seed)

| Condition | Accuracy | GT recall |
|---|---:|---:|
| **catalog_baseline** (all 40 drawn tools shown) | **93.3%** (28/30) | 100% (by construction) |
| **likelihood_topk** (top-5 of those 40, by forced-scoring) | **96.7%** (29/30) | 96.7% (29/30) |

Rank of the ground-truth tool within the 40-tool draw (forced-scoring, no
GT used to build the ranking): 26/30 ranked #1, 3/30 ranked #2, 1/30 ranked
outside the top 5.

## K sweep: bigger K is not monotonically better

Same 30 tasks, same 40-tool draws (identical scoring, only the top-K cutoff
and the resulting generation menu differ), `--topk` swept over 5, 10, 20:

| K | `likelihood_topk` accuracy | GT-in-topk (recall) | Failures |
|---:|---:|---:|---|
| 5 | **96.7%** (29/30) | 96.7% (29/30) | `live_multiple_105-43-3` (GT not recalled) |
| 10 | **96.7%** (29/30) | 96.7% (29/30) | `live_multiple_105-43-3` (GT not recalled, same task) |
| 20 | **90.0%** (27/30) | **100%** (30/30) | `live_multiple_105-43-3`, `live_multiple_280-128-3`, `live_multiple_910-189-0` (all 3 had GT recalled — the model chose wrong despite the correct tool being visible) |
| 40 (= `catalog_baseline`, no filtering) | 93.3% (28/30) | 100% | `live_multiple_105-43-3`, `live_multiple_280-128-3` |

**K=10 recovers nothing over K=5** — the one miss at K=5
(`live_multiple_105-43-3`, GT=`dartfx_help`) is not a recall problem at any
K: the model answers `"help"` instead of `"dartfx_help"` even in
`catalog_baseline` with all 40 tools visible, so it is a naming/abbreviation
mismatch (compare the `Trains_1`/`Train_1` typo case study in
`BFCL_POOL_SUBSET_20260811.md`), not something a larger candidate set can
fix.

**K=20 is worse than both K=5 and K=10**, despite perfect GT recall — going
from 10 to 20 candidates adds two new failures on tasks the smaller K values
answered correctly (`live_multiple_280-128-3`,
`live_multiple_910-189-0`), both with the correct tool visibly present in
the menu. This is the same non-monotonic pattern found in the
cluster-subspace K sweep (`BFCL_TOOL_SPACE_CLUSTERING_20260811.md`: K=5→49%,
K=14→77%, still below `native_full`'s 100%) — recall and discriminability
trade off in opposite directions as K grows: too small loses candidates
outright, too large re-admits enough marginal noise to start confusing the
model again. In this specific 40-tool/30-task run, K=5–10 was the sweet
spot; K=20 (half the pool) had already crossed into the "too cluttered"
regime.

## The headline finding: this is much easier than the earlier pool experiments, and that gap is itself informative

A **random** draw of 39 distractors from the full 457-tool catalog, plus the
correct tool, is answered correctly 93.3% of the time — a +13-point
improvement from filtering (pool-subset experiment, 80.2%→93.2%) does **not**
reproduce here; filtering only adds +3.4 points (93.3%→96.7%). Compare
across all three "large pool" experiments run in this investigation:

| Experiment | Pool composition | Unfiltered accuracy |
|---|---|---:|
| Pool-subset (`BFCL_POOL_SUBSET_20260811.md`) | Round-robin batch of ~7 tasks' candidates (mean 24 tools, moderately cross-domain) | 80.2% |
| Cluster-subspace, K=14 (`BFCL_TOOL_SPACE_CLUSTERING_20260811.md`) | One DBSCAN cluster's own tools (24 tools, all same broad domain — travel/leisure booking) | *(shared-subspace condition, not directly comparable — see that report)* |
| **This experiment** | **40 tools, 39 random draws from the full 457-tool catalog (maximally cross-domain)** | **93.3%** |

**The pattern: task difficulty tracks how *similar* the distractors are to
the correct answer, not how *many* there are.** A random draw from a
457-tool catalog spanning many unrelated domains (coffee ordering, hotel
booking, dev-ops tooling, portfolio APIs, weather, …) is mostly composed of
distractors that are obviously irrelevant to any single query — easy for the
model to rule out. The genuinely hard cases found in this investigation were
always **domain-confusable** distractors — the cluster-subspace experiment's
same-domain tool families (`Hotels_2_BookHouse` vs. `Hotels_2_SearchHouse`
vs. `Hotels_4_ReserveHotel` vs. `Hotels_4_SearchHotel`) — not sheer catalog
size.

## The 2 baseline failures

| Task | GT | Model's choice | Note |
|---|---|---|---|
| `live_multiple_105-43-3` | `dartfx_help` | `"help"` | Truncated/near-miss — plausibly a different `*_help`-named tool in the random draw, or an abbreviation; not independently diagnosed further |
| `live_multiple_280-128-3` | `Restaurants_2_FindRestaurants` | `None` (declined) | **Same task appeared as a failure case in the pool-subset experiment's case studies** (`BFCL_POOL_SUBSET_20260811.md`, case study 1) — this task appears to be intrinsically hard for this model regardless of which large pool it's embedded in |

## Caveats

- **n=30, single seed, single draw per task.** The original plan was
  n=150; that run was killed after 23 minutes (still mid-scoring, no output
  written yet) because it was taking far longer than expected for this
  server's current load — see the timing note below. n=30 is enough to see
  the *direction* (catalog_baseline is far higher than the earlier pool
  experiments' unfiltered accuracy) but not enough to trust the exact
  percentages to better than a few points.
- **This run was noticeably slower than earlier same-scale experiments in
  this investigation** (10m48s for 1200 scoring + 60 generation calls, vs.
  e.g. the pool-subset experiment's 1052-task, ~25,000-call run finishing in
  under a minute). Diagnosed as genuine GPU-side load (this is a shared
  server; `nvidia-smi` showed 2 of 4 data-parallel GPU shards at 97–99%
  utilization during the run, and other processes were active on the host),
  not a bug — confirmed by killing an earlier, badly-concurrent version of
  this script (see next point) and finding the server recovered instantly
  each time, ruling out a hung/deadlocked client.
- **An earlier version of `catalog_baseline_experiment.py` nested a
  ThreadPoolExecutor per task inside an outer pool** (same mistake made
  once before in `cluster_subspace_experiment.py`) — this was fixed to a
  flat two-phase design (score everything, then generate everything, each
  its own single pool) before any of the numbers in this report were
  produced, but cost real wall-clock time to discover and fix mid-session.
- **Only 1 random draw per task.** A single draw's 39 distractors could
  happen to be unusually easy or hard to distinguish from the GT tool;
  `--seeds-per-task` supports redrawing with multiple seeds and averaging,
  not done here due to the time cost already incurred at n=30/1-seed.
- **N=40 caps how large a "full-catalog" test can get** — this is not
  literally "show it all 457," it's the largest random sample this model's
  context window supports. A genuinely different model/serving config with
  a larger context window would be needed to test closer to the true
  catalog size; not attempted here (this session already caused one
  shared-server outage earlier by pushing concurrency too hard, and is
  deliberately conservative about further infrastructure changes to this
  shared vLLM instance).
- **Random distractor composition may be too easy to be the representative
  "hard" test.** As the headline finding argues, this experiment's 93.3%
  baseline is easy specifically *because* random draws from a diverse
  catalog are mostly obviously-irrelevant distractors. A deliberately
  adversarial draw (distractors chosen for name/domain similarity to the
  GT tool, not pure random) was proposed as a natural follow-up but not
  run in this report.

## Artifacts

- Code: `experiments/bfcl_token_likelihood/catalog_baseline_experiment.py`
- Raw results, K=5 (n=30): `runtime/bfcl_catalog_baseline_20260811/results.jsonl`
- Raw results, K=10 (n=30, same 40-tool draws): `runtime/bfcl_catalog_baseline_20260811/results_k10.jsonl`
- Raw results, K=20 (n=30, same 40-tool draws): `runtime/bfcl_catalog_baseline_20260811/results_k20.jsonl`
- Smoke-test results (n=15, same method, kept for reference): `runtime/bfcl_catalog_baseline_smoke/results.jsonl`
