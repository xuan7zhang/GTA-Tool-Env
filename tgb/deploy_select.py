"""Gold-free selection signals on TGB-v2.

ΔL needs the answer, which makes it an offline valuation tool rather than a
deployable selector. The obvious replacement is to score the model's *own*
output instead of the gold:

    conf(S) = (1/|y_hat|) * sum_t log p(y_hat_t | q, o(S), y_hat_<t)

where y_hat is what greedy decoding actually produced under S. It is strictly
cheaper than ΔL -- one generation pass with logprobs, no teacher-forced
forward over the gold at all -- and needs no labels.

The reason to expect it to be weaker is definitional, not incidental. Greedy
output sits near the mode by construction, so its likelihood is high almost
regardless of whether the context was any good; a model that reads a
contaminated number and states it confidently scores well. And across
coalitions the quantity compares *different strings*, so it is not a
comparison of the same event.

Whether that objection actually bites in v2's conflicting environment is an
empirical question -- contamination might genuinely reduce confidence -- so it
is measured here rather than assumed.

Signals computed per (task, candidate):
    conf      mean logprob of the greedy output
    margin    conf minus the same under the no-tool context
    ent       mean per-token entropy proxy (1 - exp(conf))

    GD_TAG=7b GD_MODEL=... python -m tgb.deploy_select
"""
import argparse
import collections
import json
import math
import os
import random
import statistics as st
import tempfile
import zlib

from vllm import LLM, SamplingParams

from . import scenes, tools_v2
from .add_chains_v2 import CHAINS
from .families_v2 import UNION_TOOLS
from .generate import TPL, context
from .score_dl import correct
from .tools import run_chain

tools_v2.register()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb2")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")
    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    if a.n:
        tasks = tasks[::max(1, len(tasks) // a.n)][:a.n]
    tmp = tempfile.mkdtemp(prefix="tgb_dep_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    cand = ["none", "union"] + [f"chain_{f}" for f in CHAINS]
    tools_of = dict({"none": [], "union": UNION_TOOLS},
                    **{f"chain_{f}": c for f, c in CHAINS.items()})

    def ctx_of(t, mask):
        rng = random.Random(zlib.crc32(
            f"{t['id']}/{'+'.join(sorted(mask))}".encode()))
        outs, _ = run_chain(t, t["scene"], t["boxes"], set(mask), rng,
                            corrupt_tools=())
        return context(outs, [s["tool"] for s in t["plan"]])

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    # logprobs=0 returns the logprob of each sampled token -- one pass, no
    # teacher forcing, no gold
    sp = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                        temperature=0, logprobs=0)

    prompts, meta = [], []
    for t in tasks:
        for c in cand:
            prompts.append(TPL.format(q=t["question"], o=ctx_of(t, tools_of[c])))
            meta.append((t["id"], c, t["gold"]))
    outs = llm.generate([{"prompt": p} for p in prompts], sp)

    V = collections.defaultdict(dict)
    for (tid, c, g), o in zip(meta, outs):
        out = o.outputs[0]
        lps = []
        for step in (out.logprobs or []):
            if not step:
                continue
            lp = next(iter(step.values()))
            lps.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
        conf = sum(lps) / len(lps) if lps else None
        V[tid][c] = dict(conf=conf, acc=correct(out.text, g),
                         gen=out.text.strip()[:60])
    p = os.path.join(a.dir, f"tgb_deploy_{tag}.json")
    json.dump({k: v for k, v in V.items()}, open(p, "w"), indent=1)

    T = {t["id"]: t for t in tasks}
    nt = {c: len(tools_of[c]) for c in cand}

    def ev(pick, pool):
        acc, cost = [], []
        for tid in V:
            ch = pick(tid, pool)
            acc.append(V[tid][ch]["acc"])
            cost.append(nt[ch])
        return st.mean(acc), st.mean(cost)

    def conf(tid, c):
        v = V[tid][c]["conf"]
        return v if v is not None else -99

    pool_all = cand
    pool_nonone = [c for c in cand if c != "none"]
    print(f"\n[{tag}] gold-free selection on {len(V)} tasks "
          f"({len(cand)} candidates)")
    for label, pool in [("with `none`", pool_all), ("without `none`", pool_nonone)]:
        a1, k1 = ev(lambda tid, p: max(p, key=lambda c: conf(tid, c)), pool)
        # margin against the no-tool context, the confidence analogue of dL
        a2, k2 = ev(lambda tid, p: max(
            p, key=lambda c: conf(tid, c) - conf(tid, "none")), pool)
        a3, k3 = ev(lambda tid, p: "union", pool)
        a4, k4 = ev(lambda tid, p: f"chain_{T[tid]['family']}", pool)
        print(f"  --- {label}")
        print(f"    own-confidence argmax        {a1:.3f}   tools {k1:.2f}")
        print(f"    confidence margin vs no-tool {a2:.3f}   tools {k2:.2f}")
        print(f"    fixed union                  {a3:.3f}   tools {k3:.2f}")
        print(f"    oracle routing               {a4:.3f}   tools {k4:.2f}")
    # how often is the model confident and wrong -- the reason the signal fails
    hi = [(V[tid][c]["conf"], V[tid][c]["acc"]) for tid in V for c in cand
          if c != "none" and V[tid][c]["conf"] is not None]
    hi.sort(reverse=True)
    top = hi[:len(hi) // 4]
    print(f"  confident-and-wrong: in the top quartile of confidence, "
          f"{1 - st.mean([x[1] for x in top]):.0%} of answers are wrong")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
