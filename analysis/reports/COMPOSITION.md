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

## Status: closed (negative), one live rescue path

The only version that could plausibly turn positive is a **lazy/conditional macro**
— `PerceiveAll(image, want='desc'|'ocr'|'both')` that returns only the requested
branch. But that is just parameterized orchestration; it abandons the "fuse two
tools into one" claim the C-axis was meant to test. Recorded as the open lever, not
pursued. C-axis is archived negative.

Raw runs: `results/compose_{baseline,macro}_s{1,2,3}/`. IDs: `compose_42ids.json`.
Selector/mask config: `results/inject_opt/selectors_compose.json`.
