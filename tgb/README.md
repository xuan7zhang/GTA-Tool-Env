# TGB — a tool-grounded benchmark for likelihood-based tool valuation

The 8,500-task controlled set (`dataset_full.json`) shows that gold likelihood
tracks **evidence** utility. It contains no tools: the evidence arrives already
extracted, and `tool_name` is nowhere in a record. That is fine as a positive
control and wrong as the main support for a claim about *tool* utility.

TGB is the layer above it. Every task has a raw input, a menu of tools, several
**candidate coalitions** that are actually executed, and a closed-loop
selection test that ends in task accuracy rather than in a ranking metric.

```
raw input  ->  tool execution  ->  tool outputs  ->  reasoning  ->  answer
  PNG           deterministic       GTA-format        model          scored
                executors           strings           side         vs gold
```

## The one structural invariant

    gold = easy_final( hard_intermediate , k )

The chain computes `hard_intermediate`; `k` (cash paid, a voucher, a coupon)
lives in the question. Everything the benchmark needs follows from this and is
asserted at generation time, not asserted in prose:

| property | why it holds | enforced by |
|---|---|---|
| non-echo | every tool output stops one easy step short of gold | invariant A |
| upstream necessary | the intermediate is only obtainable by reading/counting/retrieving | question carries no figures |
| downstream has no marginal alone | its argument is *parsed from* the upstream output | invariant D |
| useful coalition sufficient | replaying the easy step on the real Calculator output reproduces gold | invariant C |
| corruption ≠ absence | garbled digits stay parseable, so the chain returns a *wrong* number rather than an error | invariant E |

## Families

| family | n | raw input | chain |
|---|---:|---|---|
| `f1_extract_compute` | 800 | receipt / price-table PNG | OCR → Calculator |
| `f2_visual_reason` | 800 | shelf PNG (objects + price tag) | CountGivenObject + OCR → Calculator |
| `f3_retrieve_reason` | 800 | order-form PNG + fixed corpus | OCR + GoogleSearch → Calculator |

F3's chain is the same 3-tool shape (`Calculator + GoogleSearch + OCR`) that
appears in the real GTA read-compute subset.

## Conditions scored per task

| name | tools | what it isolates |
|---|---|---|
| `none` | ∅ | baseline for ΔL |
| `useful` | the GT chain | complete + clean |
| `partial_1of2` | one upstream dropped | coalition completeness |
| `no_downstream` | upstream only | raw evidence, hard final step |
| `no_upstream` | Calculator only | orphaned downstream (errors) |
| `wrong` | ImageDescription + Calculator | right modality, no content |
| `corrupt` | GT chain, upstream garbled | output quality |
| `full` | all 12 tools | the all-tools environment |

`useful` and `corrupt` have the *same* tools and near-identical context length
(mean 379 chars each), so a ΔL gap between them cannot be a length or a
tool-count artifact. Likewise `corrupt` keeps a `Calculator: <number>` line, so
the surface form of a Calculator answer is controlled for.

## Running it

```bash
python -m tgb.generate --n 800 --out $GTA_BIG/results/taco/tgb   # ~25 s, CPU
python -m tgb.tests.test_tgb                                     # 18 invariants
./tgb/run_all.sh                                                 # score the fleet
python -m tgb.closed_loop --tag 7b --tag 14b ...                 # the tables
```

Validation of the simulator against a real engine:

```bash
$GTA_BIG/envs/agentlego/bin/python -m tgb.validate_real_ocr --n 120
```

runs EasyOCR (what AgentLego serves GTA) over the same PNGs, formats it as the
GTA OCR tool does, and pushes it through the *same* downstream parser — so the
reported number is whether a real perception tool lands the chain on the same
intermediate.

## Files

    tgb/scenes.py     raw inputs + PNG rendering (records glyph boxes)
    tgb/tools.py      7 deterministic executors, GTA output formats, noise, chain runner
    tgb/families.py   the three task families
    tgb/coalitions.py the candidate menu per task
    tgb/generate.py   build + validate; rejects are counted, not hidden
    tgb/score_dl.py   per-model gold likelihood + greedy accuracy per coalition
    tgb/selfcons.py   zero-label self-consistency proxy
    tgb/closed_loop.py ranking AUROC + closed-loop selector table
    tgb/stats.py      dataset card
