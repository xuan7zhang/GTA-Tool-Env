# C-axis (composition) — NULL / negative result

**Question.** In E=(m, C, Φ), does the composition operator C help? Concretely:
take two primitive tools an agent must otherwise orchestrate itself and fuse them
into one macro-tool. Hypothesis: a weak model that orchestrates poorly should
benefit from the collapsed call.

**Macro tested.** `PerceiveAll` = `ImageDescription ⊕ OCR` in a single tool call
(chains both primitives through the proxy, returns
`"Image description: …\nRecognized text: …"`). Replaces the two primitives in the
composed condition. Code: `experiments/macro_tools.py`, registered via
`start_toolserver.sh GTA_MACRO=1`; driver `experiments/compose_exp.sh`.

**Setup.** Qwen2.5-7B-Instruct, full GTA-Atomic (229), 3 seeds/condition, on the
kn080 stack (LLM GPU0 / tool-server+macro GPU2 / proxy 16281). Analysis focuses on
the **42 tasks that genuinely need both perceptions** (`compose_42ids.json`), where
the macro is supposed to pay off. AnsAcc = end-to-end answer accuracy.

## Result — composing HURTS

| condition | s1 | s2 | s3 | mean ± sd | w_imggen mean |
|---|---|---|---|---|---|
| baseline (2 primitives) | 16.51 | 11.71 | 15.52 | **14.58 ± 2.53** | 13.06 |
| composed (PerceiveAll)  | 11.02 |  9.81 | 10.98 | **10.60 ± 0.69** |  9.01 |

**Δ = −3.98 AnsAcc.** This survives the noise floor: the composed condition's best
seed (11.02) is *below* the baseline's worst seed (11.71) — all three composed runs
sit under all three baseline runs. 42-subset seed-1 spot check agreed (baseline
16.13 vs composed 12.90). The direction is consistent, not a seed artifact.

## Why composition hurts here

1. **Forced redundancy.** `PerceiveAll` always returns *both* the description and
   the OCR text, even when the task needs only one. That injects an irrelevant
   modality's output into context on every call — the same mechanism that makes the
   **mask** axis positive (removing irrelevant tool output raises accuracy) makes
   this negative in reverse: a macro that packages extra output is anti-mask.

2. **Orchestration wasn't the 7B bottleneck.** The premise was "weak model
   orchestrates two tools poorly." But the baseline already calls the two
   primitives fine; the real bottleneck is perception/reasoning *quality*, which
   the macro doesn't touch. Collapsing two steps into one saves no real cost and
   adds context noise.

3. **No conditional path.** A fused tool can't skip the branch a task doesn't need.
   The agent using primitives *can* (call only OCR when only text matters); the
   macro can't.

## Consistency with the rest of the project

Fits the mask finding cleanly: **on GTA, reducing irrelevant tool output is the
lever that moves accuracy (mask gain ∝ 1/capability); adding output — even
pre-packaged "composed" output — moves it the wrong way.** Composition as *eager
fusion* is anti-mask and therefore negative for the same reason the mask axis is
positive.

## Status: PerceiveAll closed negative — but the axis is NOT dead (see RegionRead)

PerceiveAll fails because it is *parallel* fusion (concatenate two independent
perceptions). The next macro, RegionRead, fuses along a *data dependency* and wins.

Raw runs: `results/compose_{baseline,macro}_s{1,2,3}/`. IDs: `compose_42ids.json`.
Selector/mask config: `results/inject_opt/selectors_compose.json`.

---

# RegionRead — sequential-dependency composition is POSITIVE (+28.8 AnsAcc)

**Macro.** `RegionRead(image, text, attribute)` = `TextToBbox → RegionAttributeDescription`
fused into one call: locate the object, hide the bbox coordinates, VQA the region for
the asked attribute, return only the answer. Three design principles, all the inverse
of PerceiveAll: (1) fuse along the dependency (A's bbox feeds B), not in parallel;
(2) hide the error-prone glue (the LLM otherwise parses `(x1,y1,x2,y2), score N` and
threads it back — and botches it); (3) shrink the output (one attribute, no coords) so
it is pro-mask. Code: `experiments/macro_tools.py::RegionRead`.

**Setup.** The 8 GTA tasks whose gt uses this chain (ids 9,10,11,40,84,164,165,166;
`regionread_ids.json`). Qwen2.5-7B, 5 seeds. baseline = agent chains the two
primitives itself; composed = primitives hidden from the menu (`GTA_HIDE_TOOLS`),
RegionRead forced (`GTA_EXTRA_TOOLS`). Driver `experiments/regionread_exp.sh`,
config `selectors_regionread.json`. Runs on a second stack (proxy 16282 → tool
server 16182 with RegionRead) so the macro's internal calls still forward while the
menu hides the primitives.

**REQUIRED harness fix (else the result is an artifact).** First pass showed composed
+10 but with **0/40 clean tool executions in either arm** — a mirage. Cause: the ReAct
LLM appends a hallucinated ``` ```\nResponse: ... ``` after a valid JSON action, and
lagent's `JsonParser` json.loads the whole string, fails, and drops the entire tool
call (ARGS_ERROR). Fixed by monkey-patching `JsonParser.parse_inputs` to recover the
leading balanced `{...}` object (`models/lagent.py::_patch_lagent_json_parser_trailing`,
applied symmetrically to both arms). This lifted clean tool execution from ~5% to a
usable rate and made the measurement real. NB: this bug also depressed tool fidelity
in the earlier mask / PerceiveAll runs (~21% clean on full-229) — those conclusions
were drawn under low fidelity and are worth re-checking.

**Result (patched, clean).**

| | baseline (2 primitives) | composed (RegionRead) |
|---|---|---|
| clean tool executions | **2/40 = 5%** | **22/40 = 55%** (11×) |
| AnsAcc (5-seed mean ± sd) | 34.2 ± 16.2 | **63.0 ± 8.2** |
| per-seed Δ | — | +55.5 / +30.1 / +25.7 / +32.2 / +0.4 (**5/5 win**) |

**Mechanism — composition removes an orchestration barrier the weak model can't
cross.** The 7B almost never cleanly invokes the 2-step locate→describe chain (5%):
the bbox-threading is hard enough that it gives up and hallucinates an answer. The
1-step fused tool it *will* invoke (55%), and being tool-grounded it scores +28.8 with
*lower* variance (±8.2 vs ±16.2 — no hallucination lottery). The win is not "one fewer
call"; it is "the fused unit is below the invocation barrier, the chain is above it."

**When composition helps (the rule this axis establishes).** Fuse along a data
dependency, hide the error-prone intermediate, shrink the output → wins (RegionRead).
Concatenate independent outputs in parallel → loses, because it is anti-mask
(PerceiveAll). Composition buys performance by removing orchestration the model can't
do, not by bundling perception it already gets.

**Honest bounds.** n=8, so AnsAcc is coarse (each task 12.5%); the robust evidence is
the deterministic tool-execution rate (2/40 vs 22/40) plus 5/5 sign consistency, not
the noisy means. Benchmark-wide reach is capped by the chain's rarity (8/229 = 3.5%).
Still top1 (mechanical) — the 3 multi-object tasks (9/40/84) that need spatial
selection ("middle/right") are a known soft spot; RegionRead executes but can pick the
wrong box.

Raw runs: `results/rr_{baseline,composed}_s{1..5}/`. Macro/patches:
`experiments/macro_tools.py`, `patches/GTA/.../models/lagent.py`,
`patches/GTA/.../datasets/gta_bench.py` (GTA_TASK_IDS). Config:
`selectors_regionread.json`.
