# BFCL v1 pilot: can per-candidate token likelihood pick out the ground-truth tool?

Date: 2026-08-11. Branch: `bfcl-token-likelihood-pilot`. Model: Qwen3-VL-8B-Instruct
(already running on this host, vLLM, `127.0.0.1:8013`, served as `gpt-qwen3-vl-8b`).

## Question

Given a task with several candidate tools and one ground-truth (GT) tool, does
ranking the candidates by their token likelihood (or, inversely, entropy)
recover the GT tool — i.e. can this signal be used to prune a task's tool
space down to a working subset, with no ground truth used at selection time?

This is the BFCL analogue of this repo's `experiments/relevance_selector.py`
(which ranks GTA's 14 tools by query↔tool embedding similarity and scores
against GT) — same "rank all candidates, evaluate against GT" evaluation
shape, swapping the signal from embedding similarity to token likelihood.

## Data

[BFCL v1](https://gorilla.cs.berkeley.edu/leaderboard.html) `multiple`
category (single-turn, non-executable, AST-scored — one of the original v1
categories; the HF dataset has renamed the files in place to `BFCL_v3_*` but
`multiple`/`simple`/`parallel`/`parallel_multiple` are still the same v1
category semantics). Source:
`gorilla-llm/Berkeley-Function-Calling-Leaderboard` on Hugging Face,
`BFCL_v3_multiple.json` + `possible_answer/BFCL_v3_multiple.json`.
All 200 tasks in the category; 2–4 candidate functions per task (mean 2.79),
exactly one GT function per task (no multi-GT cases in this category).
Raw data: `runtime/bfcl_v1_pilot_data/{questions,answers}.json`.

## Method

**Forced-scoring (the actual signal being tested).** For each task, build one
shared ReAct-style prefix — numbered function list (name, description,
JSON-schema params) + user request + `Thought: ... \nAction:` — mirroring the
exact convention this repo's GTA/lagent agent already uses (see
`GTA/opencompass/opencompass/models/token_metrics.py`'s phase classifier,
which parses this same `Action:` cue). For every candidate function in the
task, score `P(candidate_name | prefix)` by calling the live vLLM
OpenAI-compatible `/v1/completions` endpoint with `echo=True, max_tokens=0,
logprobs=20` — this returns per-token logprobs for the whole echoed prompt,
including the appended candidate name, so we slice out just the candidate's
own tokens via character offset and take the mean per-token logprob (ranking
metric, matches `token_metrics.py`'s own mean-logprob convention) and the
mean top-k entropy at those positions. Rank all candidates per task by mean
logprob, descending.

**Free-generation baseline (context only, not the object of study).** One
unconstrained completion of `Action:` per task (temperature 0), string-matched
against candidate names — answers "does the model already solve this without
any forced scoring," to contextualize whether the forced-scoring result is
doing real work or riding a ceiling.

Implementation: `experiments/bfcl_token_likelihood/score_candidates.py`
(scoring) + `summarize.py` (pooled metrics). Fetch script:
`experiments/bfcl_token_likelihood/fetch_data.sh`.

## Results (all 200 tasks, single seed)

| Metric | Value |
|---|---:|
| Top-1 accuracy, forced-scoring (highest-likelihood candidate = GT) | **98.5%** |
| MRR, forced-scoring | 0.9925 |
| Pooled AUROC (mean logprob vs. is-GT, 557 candidate-task pairs) | **0.9977** |
| Random-guess baseline (mean of 1/n_candidates per task) | 38.4% |
| Free-generation baseline (unconstrained, no forced scoring) | 94.5% |
| Mean candidate-set entropy (softmax over per-candidate scores) when top-1 correct | 0.204 nats |
| Mean candidate-set entropy when top-1 **wrong** | 0.630 nats |
| Pearson r, normalized entropy vs. top-1 correctness | −0.247 |

An 80-task subset (first 80 tasks, which happened to skew toward 2–3
candidates rather than the full 2–4 range) gave consistent numbers: top-1
97.5%, AUROC 0.995, MRR 0.9875, free-gen 92.5% — included for reference in
`runtime/bfcl_token_likelihood_pilot_20260811/summary.json`.

**Headline: on this category, token likelihood recovers the GT tool almost
perfectly (AUROC 0.998) and beats the free-generation baseline (98.5% vs.
94.5%).** The 3 failures (of 200) all have small likelihood gaps between the
top pick and GT (e.g. −0.36 vs. −0.985 nats, −0.094 vs. −1.872 nats) and sit
in the high-entropy tail — the signal is not just accurate, it is also
reasonably well-calibrated about when it's likely to be wrong.

## This is the opposite conclusion from the GTA/Shapley pilot — read the gap, don't average it away

`analysis/reports/TOKEN_LIKELIHOOD_SHAPLEY_QWEN3VL_20260804.md` found
essentially **no relationship** (r ≈ −0.06 to −0.16) between token likelihood
and a tool's causal contribution (Shapley value) on GTA. This pilot finds a
**strong** relationship (AUROC 0.998) between token likelihood and whether a
candidate is the GT tool on BFCL. Both are real, single-seed, small-n
findings from the same model family — the likely reason they diverge is a
genuine task-structure difference, not noise:

- **GTA/Shapley** measures confidence in *proposing a JSON tool call* inside a
  multi-turn ReAct trajectory, and correlates it against *how much that tool's
  presence causally changed the final answer* (Shapley) — two different
  things. A tool can be called with high syntactic confidence and still not
  be the one that mattered for the answer.
- **This BFCL pilot** measures confidence in *naming the one correct function*
  out of 2–4 explicitly enumerated candidates in a single shot, scored
  against whether that name literally *is* the GT label — likelihood and
  correctness are much more directly the same question here.

So: token likelihood looks like a strong signal for **"which of these
explicitly-listed candidates is the right one to call"** (a retrieval/
selection framing), and a weak-to-null signal for **"which called tool
actually mattered for getting the task right"** (a causal-attribution
framing). Both are reasonable readings of "optimize the tool space" — this
pilot only speaks to the first one.

## Caveats (single pilot, read narrowly)

- **Ceiling effect.** Both the forced-scoring method (98.5%) and the naive
  free-generation baseline (94.5%) are near-ceiling on this category with an
  8B instruct model — the category may simply be easy (2–4 candidates,
  semantically distinct names like `walmart.vegan_products` vs.
  `safeway.vegan_products`). This limits how much the pilot can say about
  whether likelihood adds value over just asking the model directly; the gap
  (98.5% vs. 94.5%, i.e. forced-scoring recovers 4 of 200 cases free-gen
  missed) is the real but modest signal, not the headline AUROC.
- **Single category, single model, single seed.** Only BFCL v1 `multiple`
  (2–4 candidates) was tested. `parallel_multiple` (more candidates, multiple
  correct) and `simple` (only 1 candidate, no selection problem) would be
  informative extensions — `parallel_multiple` especially, since it's the
  closest BFCL analogue to "prune a larger tool pool," unlike `multiple`
  which already hands the model a small pre-filtered menu.
  `possible_answer/` ground truth was used only to score the ranking, never
  to build it, matching `relevance_selector.py`'s convention.
  No repeats, no other model tested (`Qwen2.5-7B` is running elsewhere on
  this host and would be a cheap second data point, same category).
- **Task format is single-shot naming, not GTA's live multi-turn execution
  loop.** This pilot never actually calls a tool or checks argument
  correctness — it only tests whether likelihood identifies the right
  *function name* among candidates, the first half of BFCL's own AST-match
  scoring (which also checks arguments).
- Entropy here is entropy over the *candidate set* (softmax of per-candidate
  mean logprobs), not the per-token entropy `token_metrics.py` logs during
  natural generation — a different quantity from the GTA reports, chosen
  because it directly answers "how uncertain is the ranking," not "how
  uncertain was any single token."

## Suggested next step

Re-run on `parallel_multiple` (larger, noisier candidate pools — the harder
and more representative case for "optimize the tool space") before treating
this as validated; if it holds up there, the natural extension is a
`predmask_curve.sh`-style recovery-curve experiment (build a likelihood-based
predicted mask, measure accuracy recovery vs. the oracle-GT-mask and
full-pool baselines, same metric already used for the embedding/router
selectors in `experiments/relevance_selector.py` and
`experiments/router_selector.py`).

## Artifacts

- Code: `experiments/bfcl_token_likelihood/{fetch_data.sh,score_candidates.py,summarize.py}`
- Raw data: `runtime/bfcl_v1_pilot_data/{questions,answers}.json`
- Per-task scored output: `runtime/bfcl_token_likelihood_pilot_20260811/{scored_full200,scored}.jsonl`
- Pooled summaries: `runtime/bfcl_token_likelihood_pilot_20260811/{summary_full200,summary}.json`
