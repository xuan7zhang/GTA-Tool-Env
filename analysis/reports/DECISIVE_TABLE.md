# Decisive table — trust axis, greedy (deterministic) decoding

The head-to-head that carries the paper: with 8 unreliable ("poison") tools injected
into the pool, which selection method survives **both** attack regimes? Evaluated
under the minimax objective `max_E min_regime Acc(E)` — worst case matters because the
regime is unknown at deployment.

## Setting

Qwen2.5-32B (fp16, **greedy/deterministic decoding**), GTA-Atomic full 229, AnsAcc.
Pool = 14 real + 8 injected = 22 tools. Two regimes isolate the two factors of
`E[harm] = P(call) · E[damage | call]`:

- **attractive** — authoritative descriptions inflate `P(call)`; harm flows through Φ.
- **subtle** — descriptions rewritten neutral; call rates normal, harm flows only
  through behaviour `C`.

Greedy decoding drops the run-to-run noise floor to ~±0.8 (tool-server nondeterminism
only); the sampled harness default (temperature 0.7) had been inflating it to ±2.5–3.
Seed 1 unless noted.

## Results

| method | signal | attractive | subtle | **worst-case** | detect | false-prune |
|---|---|---|---|---|---|---|
| **Ground+** (ours) | behavioural probe + declutter | 14.10 | **15.40** | **14.10** | **8/8** | **0** |
| Ground (ours) | behavioural probe | 13.66 | 13.83 | 13.66 | **8/8** | **0** |
| Oracle (cheats) | ground-truth poison set | 13.86 | 13.35 | 13.35 | 8/8 | 0 |
| KeepAll | — (no defence) | 12.75 | 12.90 | 12.75 | — | — |
| CallFreq | call frequency | 14.55 | **8.31** | 8.31 | ✗ | ✗ |
| AnswerEcho | answer agreement | 12.00 | **8.13** | 8.13 | ✗ | ✗ |
| **OctoTools greedy** | validation accuracy | **10.14** | 14.12 | **10.14** | 1–4/8 | 2–9 |

## Reading

- **Ground+ wins the minimax** (worst-case 14.10), and both probe variants return
  *identical verdicts in both regimes* — a probe asks whether a tool's output depends
  on its input, which is a property of the tool alone, so it cannot be moved by the
  description channel the attacker manipulates.
- **Every outcome-based method is regime-fragile.** CallFreq and AnswerEcho look fine
  or good on attractive (14.55 / 12.00) and then **collapse to ~8 on subtle** — below
  doing nothing (12.75) — because they prune genuine OCR/captioning tools once the
  poison stops being over-called.
- **OctoTools fails in both regimes, differently** (see `OCTOTOOLS.md`): on attractive
  its leave-one-out marginals are swamped by val noise so it **keeps 7/8 poison** and
  lands at 10.14, *below no-defence*; on subtle it **false-prunes 9/14 genuine tools**
  and only reaches 14.12 by luck (the pruned set happens to include the declutter-able
  image-output tools). It never achieves the probe's 8/8-detect / 0-false-prune, and
  its selected set swings wildly between regimes — that instability *is* the fragility.
- Ground+ exceeds Oracle on subtle because Oracle only removes the injected tools while
  Ground+ additionally declutters image-output tools; see the ablation in `SUMMARY.md`.

## Cost

The probe needs a handful of **label-free** calls per tool and is agent-independent.
OctoTools needs **O(n) full labelled validation evaluations**; CallFreq/AnswerEcho need
full agent rollouts. This asymmetry belongs in the table's discussion.

## Clean-pool control — does the defence harm when there is nothing to defend against?

A defence that damages a clean environment is not deployable. We separate the two
components:

| condition (clean pool, **no injection**, 32B greedy) | AnsAcc |
|---|---|
| natural 14-tool menu (`gm32b_oracle`) | 14.3 |
| forced all-14 (`gm32b_full`) | 14.5 *(n=2)* |
| **Ground (probe only)** | **≡ natural, by construction** |
| **Ground+ (probe + declutter, 9 tools)** | **12.28** |

- **Ground is a provable no-op on a clean pool.** The probe prunes only tools it
  *positively* proves input-degenerate; anything untestable or errored is kept
  (fail-safe). Empirically **0/14 genuine tools are flagged INVARIANT** (5 verified
  `grounded`, 3 `untestable`, 6 `errored`), so its keep-set is byte-identical to
  keep_all — it cannot self-harm. This is the property no outcome-based method has:
  CallFreq/AnswerEcho/OctoTools all false-prune genuine tools and fall below
  no-defence in at least one regime.
- **Ground+ is not free.** Its declutter component fires regardless of whether poison
  is present, so on a clean pool it still removes the 5 image-output tools and
  **costs ≈2 points** (12.28 vs 14.3). The extra subtle-regime gain of Ground+ over
  Ground (15.40 vs 13.83) is bought with this clean-environment cost.
- **Deployment reading:** **Ground is the safe default** — use it when you do not know
  whether the pool is compromised. **Ground+ only when injection is suspected**, and
  report the declutter trade-off (it also depresses image-generation queries).

*(Caveat: the clean-pool Ground+ run is n=1 and its paired same-stack baseline did not
finish before the node expired, so the ≈2-point cost is measured across stacks
(±0.8 noise). It is outside the noise floor but deserves a paired re-run.)*

## Caveats

Seed 1 (greedy ⇒ ±0.8 residual, so single-seed separations of ≥2 points are meaningful;
the 6-point gaps here are far outside it). Multi-seed means would tighten the estimates
and expose the selection variance of the outcome-based methods, which is expected to be
large. Raw runs: `results/gt1_{attractive,subtle}_*_s1/`, `results/octotools_gta/`.
