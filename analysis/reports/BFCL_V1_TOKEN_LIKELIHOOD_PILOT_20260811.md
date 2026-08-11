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
metric) and the mean top-k entropy at those positions. Rank all candidates
per task by mean logprob, descending.

**Name-blinding ablation.** Same method, but every candidate's literal name
in both the function list and the scored continuation is replaced with
`func_1..func_N` (description and parameter schema untouched). Tests whether
the signal is genuine reasoning over the task/description match, or just
lexical pattern-matching between the query and a suggestively-named function
(e.g. a query about weather scoring `get_weather` high just because the
words overlap, independent of whether it's actually the right tool).

**Two baselines, for cost/benefit context, not the object of study:**
- *Free-generation*: one unconstrained completion of `Action:` per task
  (temperature 0), matched against candidate names (exact string match,
  substring in either direction, or unambiguous short-form match e.g.
  `get_rate` for `currency_conversion.get_rate`) — does the model already
  solve this without any forced scoring?
- *Token overlap*: zero-inference baseline — rank candidates by
  `|words(query) ∩ words(name + description)|`, ties split credit. Cheapest
  possible no-GT selector, analogous in spirit to `relevance_selector.py`'s
  embedding signal but requiring no model at all.

Implementation: `experiments/bfcl_token_likelihood/score_candidates.py`
(scoring, `--anonymize` flag for the ablation) + `summarize.py` (pooled
metrics). Fetch script: `experiments/bfcl_token_likelihood/fetch_data.sh`.

## Exact settings

| Setting | Value |
|---|---|
| Model | Qwen3-VL-8B-Instruct |
| Serving | vLLM, OpenAI-compatible, `http://127.0.0.1:8013/v1`, served-model-name `gpt-qwen3-vl-8b` (already running on this host for other work; not started by this pilot) |
| Endpoint used for scoring | `POST /v1/completions` |
| Scoring request params | `echo=True, max_tokens=0, logprobs=20` |
| Free-gen baseline params | `POST /v1/completions`, `max_tokens=24, temperature=0.0, stop=["\n"]` |
| Data | `gorilla-llm/Berkeley-Function-Calling-Leaderboard` (HF), `BFCL_v3_multiple.json` + `possible_answer/BFCL_v3_multiple.json` — all 200 tasks, no sampling/filtering |
| Concurrency | `ThreadPoolExecutor`, 8 workers, sequential HTTP calls |
| Runtime | ~13s for 80 tasks, ~30s for 200 tasks (this server, shared with other jobs) |

**Where exactly the scored tokens sit — before the tool call, only the name span, nothing else.**
Every candidate is scored against the *same* shared prefix (function list +
user request + `Thought: ...\nAction:`); only the text appended after
`Action:` differs per candidate, and only that appended span's tokens are
read out of the response. Concretely, for `multiple_1`:

```
[... 3-function list, user request, Thought: ... ]
Action:                              <- prefix ends here (identical for all 3 candidates)
        math.triangle_area_heron     <- scored tokens for candidate 1 (≈5 sub-word tokens)
        math.circle_area             <- scored tokens for candidate 2 (≈3 sub-word tokens)
        math.triangle_area_base_height  <- scored tokens for candidate 3 (≈6 sub-word tokens)
```

The split is done by character offset (`text_offset >= len(prefix)` in the
vLLM completions response), not by re-tokenizing separately, so it's exact
regardless of how the tokenizer merges the leading space into the first
sub-word. This is **only the function-name span** — no arguments, no tool
execution, no observation, no final answer. It is the direct analogue of
`token_metrics.py`'s `before_tool` phase in the GTA infra, narrowed further
to just the name (GTA's `before_tool` phase covers the whole `Action: name\n
Action Input: {...}` block). There is no `after_tool` / `final_answer` phase
in this pilot at all — BFCL's `multiple` category is single-shot,
non-executable, AST-scored; no tool is ever actually called, so that phase
doesn't exist here (unlike the GTA/`token_metrics.py` setup, which does log
both phases across a live multi-turn ReAct loop).

## Results (all 200 tasks, single seed)

| Signal | Top-1 accuracy | Mean within-task AUROC¹ | MRR |
|---|---:|---:|---:|
| **Forced-scoring, real names** | 98.5% (197/200) | 0.9900 | 0.9925 |
| **Forced-scoring, names anonymized (`func_1..func_N`)** | **100.0% (200/200)** | 1.0000 | 1.0000 |
| Free-generation (unconstrained, context only) | 95.0% (190/200) | — | — |
| Token-overlap, no model, context only | 92.7%² | — | — |
| Random guessing | 38.4% | 0.500 | — |

¹ Mean over tasks of the within-task pairwise AUROC (GT's score vs. every
distractor *in the same task*). Reported instead of a naive pooled AUROC
across all 557 candidate-task pairs — pooling scores GT candidates against
distractors drawn from *other* tasks too, which are separable on absolute
likelihood scale for reasons that have nothing to do with tool selection
(that inflated number was 0.9977; it's noted here only to flag it as the
wrong metric, not to lead with it).
² With split credit on the 21/200 tasks with tied top overlap scores.

**Headline: token likelihood recovers the GT tool almost perfectly** on this
category (98.5–100% top-1, depending on whether names are shown), clearing
both the free-generation baseline (95.0%) and a zero-inference token-overlap
baseline (92.7%). The forced-scoring/free-generation gap is modest but real
(7–8 tasks) — this is not primarily a parsing artifact: the free-gen matcher
was checked against all "wrong" cases by hand, and most really are the model
outputting a wrong tool or a non-answer (`"function_call"`, `"Function 1"`),
not a name-matching failure on the scorer's side.

**Name-blinding ablation result (the important check): the signal survives,
and in fact improves, when literal tool names are hidden.** This rules out
the most obvious failure mode — that "high likelihood" is just detecting
superficial lexical overlap between the query and a suggestively-named tool
(e.g. picking `get_weather` for a weather query independent of whether it's
actually correct). With names replaced by opaque `func_i` labels, the model
still has to route based on the *description text* to name the right index,
and it does that with zero errors across all 200 tasks. (The improvement
from 98.5%→100% is plausibly partly a measurement-noise effect too — `func_i`
labels are uniform-length across candidates, removing the variable
subword-tokenization-length confound that real names of different lengths
introduce into the mean-logprob estimate — but either way, the finding that
motivated running this check is confirmed: it is not lexical-name-matching.)

A position-bias check (does the signal work by favoring whichever position
GT happens to sit in, rather than reading the task) also came back clean:
GT position in the candidate list is close to uniform (73/71/42/14 across
positions 0–3) and top-1 accuracy is flat across positions (0.986 / 0.986 /
0.976 / 1.000) — no exploitable positional prior. A ranking-normalization
check (`mean_logprob` vs `sum_logprob` per candidate) also came back clean:
0/200 tasks flip their top-1 pick between the two normalizations.

## This is the opposite conclusion from the GTA/Shapley pilot — read the gap, don't average it away

`analysis/reports/TOKEN_LIKELIHOOD_SHAPLEY_QWEN3VL_20260804.md` found
essentially **no relationship** (r ≈ −0.06 to −0.16) between token likelihood
and a tool's causal contribution (Shapley value) on GTA. This pilot finds a
**strong** relationship (top-1 98.5–100%, within-task AUROC 0.99–1.00)
between token likelihood and whether a candidate is the GT tool on BFCL. Both
are real, single-seed, small-n findings from the same model family — the
likely reason they diverge is a genuine task-structure difference, not
noise:

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

## Per-candidate token likelihood: correct tool vs. incorrect tool

The AUROC/top-1 numbers above are aggregates; this is the group-level
statistic underneath them — pooling every scored candidate across all 200
tasks into two buckets, "is the ground-truth tool" vs. "is a distractor",
and reporting `mean_logprob` for each bucket separately. Reproduce with
`experiments/bfcl_token_likelihood/gt_vs_distractor_breakdown.py`; raw
numbers in `runtime/bfcl_token_likelihood_pilot_20260811/gt_vs_distractor_breakdown.json`.

**Real tool names:**

| Group | n | mean logprob | std | min | max |
|---|---:|---:|---:|---:|---:|
| Correct tool (GT) | 200 | **−0.144** | 0.297 | −1.872 | −0.0002 |
| Incorrect tool (distractor) | 357 | **−5.211** | 2.835 | −26.505 | −0.095 |
| Gap (correct − incorrect) | | **+5.07 nats** | | | |

**Names anonymized (`func_1..func_N`):**

| Group | n | mean logprob | std | min | max |
|---|---:|---:|---:|---:|---:|
| Correct tool (GT) | 200 | **−0.201** | 0.142 | −0.814 | −0.015 |
| Incorrect tool (distractor) | 357 | **−4.267** | 0.619 | −6.505 | −0.856 |
| Gap (correct − incorrect) | | **+4.07 nats** | | | |

In both settings the two groups are almost non-overlapping: in nats, −0.14
vs. −5.21 is roughly the difference between the model assigning the correct
tool ~87% probability on average and a distractor ~0.5–1% — a two-order-of-
magnitude gap, which is why a simple threshold on this one number separates
the groups almost perfectly (this is the same fact the AUROC/top-1 numbers
above are reporting, just as raw probabilities instead of a ranking metric).

**Per-token entropy along that same span does *not* separate the groups**
(0.168 vs. 0.187 nats real-names; 0.447 vs. 0.425 nats anonymized — both
pairs within noise of each other). Entropy at a token position measures "how
many other tokens could plausibly continue from here," and a tool name is
close to lexically unique in the vocabulary either way (correct or wrong),
so local per-token entropy along the forced path carries almost no signal
about whether the *whole name* is the right choice. The signal lives in the
**joint probability of the whole name span** (`mean_logprob`, i.e.
likelihood), not in per-token local uncertainty (`entropy`) — this is the
practical reason the pilot ranks candidates by mean logprob and only uses
entropy as a secondary/calibration check (candidate-set-level entropy, see
Caveats below, not this per-token entropy).

## Caveats (single pilot, read narrowly)

- **Ceiling effect.** Both the forced-scoring method (98.5–100%) and the
  free-generation baseline (95.0%) are near-ceiling on this category with an
  8B instruct model — the category may simply be easy (2–4 candidates,
  usually semantically distinct). This limits how much the pilot can say
  about incremental value over just asking the model directly; the gap is
  real (7–8 of 200 cases) but small in absolute terms.
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
- Entropy reported (`candidate_entropy_nats`) is entropy over the *candidate
  set* (softmax of per-candidate mean logprobs), not the per-token entropy
  `token_metrics.py` logs during natural generation — a different quantity
  from the GTA reports, chosen because it directly answers "how uncertain is
  the ranking," not "how uncertain was any single token." It behaves
  sensibly: 0.20 nats mean when top-1 is correct vs. 0.63 nats when wrong
  (r = −0.25 between normalized entropy and top-1 correctness).

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

- Code: `experiments/bfcl_token_likelihood/{fetch_data.sh,score_candidates.py,summarize.py,gt_vs_distractor_breakdown.py}`
- Raw data: `runtime/bfcl_v1_pilot_data/{questions,answers}.json`
- Per-task scored output: `runtime/bfcl_token_likelihood_pilot_20260811/scored_full200{,_anonymized}.jsonl`
- Pooled summaries: `runtime/bfcl_token_likelihood_pilot_20260811/summary_full200{,_anonymized}.json`
- GT-vs-distractor breakdown: `runtime/bfcl_token_likelihood_pilot_20260811/gt_vs_distractor_breakdown.json`
