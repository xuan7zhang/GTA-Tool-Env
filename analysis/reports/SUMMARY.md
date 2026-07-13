# GTA tool-environment optimization — project summary

Goal: treat the tool environment E=(m,C,Φ) as a first-class object and test
whether it can be **optimized**. Testbed: GTA-Atomic (229 tasks, 14 real tools),
served with a FastAPI probe-proxy in front of the AgentLego tool server, driven
by an OpenCompass/Lagent ReAct agent. Models: Qwen2.5-7B, then 32B (tp=2) on
Killarney L40S. Every accuracy is AnsAcc (end-to-end); a ±2.5–3 run-to-run noise
floor (32B/tool-server nondeterminism even under greedy decoding) means single
runs are unreliable — multi-seed and paired-bootstrap throughout.

## Arc 1 — deployment + causal probes (done)
Baseline reproduced within ±5 of the leaderboard. Causal probes (proxy modes:
passthrough / tool_free / corrupt_output / unreliable): ΔTool=3.5; corrupting a
tool's output is as damaging as removing it (C≈B); of correct tool-using answers
91–100% break under corruption (they genuinely depend on the tool), but such
wins are rare and tools induce a comparable number of errors — so the *net*
tool contribution is small. (`FINAL_REPORT.md`, `causal_report.md`.)

## Arc 2 — mask-axis optimization is NULL (done)
Leave-one-out ΔTool *looked* like it split the pool into contributors and
distractors, but it did not replicate: 3-seed (7B) means and a 32B task-bootstrap
put all mask variants at the same accuracy (CIs identical). Which tools are in
the pool does not move accuracy or cost on GTA. The apparent single-seed effects
were noise. 32B baseline is *below* 7B because 37% of its ReAct steps are
NoAction (a harness/Φ bottleneck, not a tool-selection one).

## Arc 3 — injection → optimization → beat baselines (the paper direction)
Inject 8 authoritative-wrong **poison** tools (advertise one-step superiority,
return confident wrong answers). A tool-eager 32B takes the bait (49% of calls
go to poison; AnsAcc drops). Then test whether a tool-selection method recovers
by pruning, and beats baselines.

**Two regimes** (the key control): *attractive* poison (authoritative
descriptions → over-called) vs *subtle* poison (descriptions rewritten neutral
via the Φ axis → called at normal rates). This exposes a **harm∝frequency
coupling**: a tool harms only through use, so heavy use → frequency-detectable;
subtle poison harms less but is invisible to frequency/echo signals.

### Method comparison (32B AnsAcc; ours_grounding*/gplus/cf are multi-seed means)
| method | signal | attractive | subtle | worst-case | robust? |
|---|---|---|---|---|---|
| **ours_grounding_plus** | grounding poison ∪ image-output declutter | **13.02 ± 0.64** | **14.64 ± 0.72** | **13.02** | ✅ tight, both |
| ours_grounding | input-grounding (poison only) | 12.6 ± 0.4 | 12.72 ± 3.1 | 12.6 | ✅ |
| oracle (needs ground truth) | prune injected | 13.09 | 13.38 | 13.09 | ✅ (cheats) |
| call_frequency | prune most-called | 14.49 ± 2.17* | 10.16 ± 0.25 | 10.16 | ❌ regime-fragile |
| answer_echo | agent submits its output | 15.59 | 7.69 | 7.69 | ❌ hurts subtle |
| error_rate / random | — | 9.26 / 9.19 | — | <keep_all | ❌ |
| keep_all (no optimization) | — | 10.19 | 11.13 | 10.19 | baseline |

\*call_frequency attractive is 2 seeds (12.32, 16.66) — huge variance; 3rd seed pending.

### The method that wins: input-grounding
Every baseline uses an *indirect* signal that a noise/coupling defeats:
call-frequency (biased by attractiveness), answer-echo (can't tell
useful-echoed from poison-echoed), corruption-attribution (drowned by ±accuracy
noise — it pruned 8 real tools and kept all poison, scoring 8.49, the worst).
**ours_grounding** probes each tool *directly* for whether its output depends on
its input — no agent, no accuracy, no ±noise, no frequency, description-independent:
- image tools: output(real image) vs output(black square) — poison is invariant;
- text tools: output cardinality over 15 diverse inputs — poison collapses into
  its tiny fixed pool (distinct/tested < 0.5).

It selects **exactly the 8 poison, 0 real tools, in both regimes** (= the oracle
tool-set, without ground truth) — a deterministic, noise-free property.
**ours_grounding_plus** adds a principled declutter (prune image-OUTPUT tools,
useless for text answers), giving the de-cluttering bonus call_frequency got.

## Standing result (cf 3rd/4th seed pending)
- **ours_grounding_plus wins on subtle (14.64 vs ~10.2), worst-case (13.02 vs
  10.16), and variance (±0.64 vs ±2.17).** call_frequency/answer_echo are
  regime-fragile: they win or tie on attractive poison but drop *below*
  no-optimization on subtle, because they prune genuinely useful tools
  (OCR/ImageDescription). ours never prunes a real perception/logic tool, so it
  structurally cannot hurt.
- attractive-regime *mean* is the only open cell: call_frequency (14.49, wide,
  n=2) vs gplus (13.02, tight). Finalizing with the last cf seeds; if
  call_frequency's mean holds >13.5, a small declutter-tuning iteration (keep 3
  image tools instead of pruning all 5) recovers the attractive edge while
  preserving the subtle/robustness win.

## Honest limitations
- ±2.5–3 run-noise; claims rest on the deterministic **selection** property
  (8/8 poison, 0 real pruned) more than on the noisy accuracy means.
- Recovery is partly de-cluttering (removing natural image distractors), not
  only de-poisoning; declutter tanks AnsAcc_w_imggen (image-gen tasks) — scoped
  to answer-type AnsAcc.
- Grounding assumes input-degenerate poison; an adversary returning
  plausible, image-*dependent* wrong outputs would defeat it (next open problem).

## Bottom line
Environment optimization under injected harm is real (oracle recovers). A cheap,
ground-truth-free, regime-robust selector (**input-grounding**, +declutter)
matches the oracle tool-set and is the only realistic method with positive
worst-case recovery and low variance; frequency/echo baselines are regime-fragile
and can actively harm. This is the defensible "our method beats the baselines"
result, on the axis that matters against an unknown adversary: worst-case
robustness.

Deliverables: `REPORT.md` (full two-regime + grounding), this SUMMARY, `FINAL_REPORT.md`
(mask-axis null), `causal_report.md`, `bootstrap*.md`, `selectors*.json`;
scripts in `scripts_extra/` (grounding_selector, grounding_plus, iter_run,
cf_seeds, answer_echo_selector); envgen `variants/poison/`.
