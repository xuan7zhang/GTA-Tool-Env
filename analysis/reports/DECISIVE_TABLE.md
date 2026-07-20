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

## Caveats

Seed 1 (greedy ⇒ ±0.8 residual, so single-seed separations of ≥2 points are meaningful;
the 6-point gaps here are far outside it). Multi-seed means would tighten the estimates
and expose the selection variance of the outcome-based methods, which is expected to be
large. Raw runs: `results/gt1_{attractive,subtle}_*_s1/`, `results/octotools_gta/`.
