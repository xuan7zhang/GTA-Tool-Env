"""Likelihood-based vs accuracy-based Shapley value, computed exactly.

    phi_i^A = sum_{S subset T\\{i}} w(S) [ A(S+i) - A(S) ]
    phi_i^L = sum_{S subset T\\{i}} w(S) [ L(S+i) - L(S) ],   w(S)=|S|!(n-|S|-1)!/n!

The tool menu here is TGB-v1's four-tool union, so all 2^4 = 16 subsets are
enumerated -- these are exact Shapley values, not sampled estimates.

The claim under test is *not* "likelihood-Shapley is good". It is whether
phi^L agrees with phi^A well enough to stand in for it, and whether a subset
chosen by phi^L actually raises held-out accuracy. The project's own §3 already
measured corr(phi^L, phi^A) ~= 0.29 on real GTA with three of seven tools
flipping sign, so the prior is not encouraging; this asks whether that was a
property of the signal or of the extraction-dominated environment it was
measured in.

Held-out protocol: Shapley values are estimated on a *train* half and the
selected subset is evaluated on the other half. The split is by task index
within family, which shares templates across halves -- a leak worth naming,
and the reason the held-out number should be read as an upper bound.

    GD_TAG=7b GD_MODEL=... python -m tgb.shapley_ll
"""
import argparse
import itertools
import json
import math
import os
import random
import statistics as st
import tempfile
import zlib

from vllm import LLM, SamplingParams

from . import scenes
from .generate import TPL, context
from .score_dl import correct
from .tools import run_chain

MENU = ["OCR", "Calculator", "CountGivenObject", "GoogleSearch"]


def spearman(x, y):
    def rank(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
                j += 1
            av = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[o[k]] = av
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else 0.0


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** .5
    return num / den if den else 0.0


def main():
    global MENU
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--tgb4", action="store_true",
                    help="TGB-v4: exact Shapley over the 7-tool real union "
                         "(2^7 subsets), fixed-boundary span scoring")
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")
    if a.tgb4:
        from . import tools_v2, tools_v4
        from .families_v4 import UNION_TOOLS_V4
        tools_v2.register()
        tools_v4.register()
        MENU = list(UNION_TOOLS_V4)
        if a.dir.rstrip("/").endswith("/tgb"):
            a.dir = a.dir.rstrip("/")[:-4] + "/tgb4"
    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    tasks = tasks[::max(1, len(tasks) // a.n)][:a.n]
    tmp = tempfile.mkdtemp(prefix="tgb_shap_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    subsets = [frozenset(c) for r in range(len(MENU) + 1)
               for c in itertools.combinations(MENU, r)]

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
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sp_gen = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                            temperature=0)

    prompts, meta = [], []
    for t in tasks:
        for S in subsets:
            prompts.append(TPL.format(q=t["question"], o=ctx_of(t, S)))
            meta.append((t["id"], S, t["gold"]))
    jobs = []
    if a.tgb4:
        from .score_dl import span
        for p, (_, _, g) in zip(prompts, meta):
            jobs.append(span(tok, p, g, 4096))
    else:
        for p, (_, _, g) in zip(prompts, meta):
            gi = tok(g, add_special_tokens=False)["input_ids"]
            pi = tok(p, add_special_tokens=False)["input_ids"][:4096 - len(gi) - 1]
            jobs.append((pi + gi, len(pi), len(gi)))
    outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
    LLv = []
    for (full, plen, ng), o in zip(jobs, outs):
        lps = []
        for i in range(plen, plen + ng):
            d = o.prompt_logprobs[i]
            if d is None:
                continue
            lp = d.get(full[i])
            if lp is not None:
                lps.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
        LLv.append(sum(lps) / len(lps) if lps else None)
    gouts = llm.generate([{"prompt": p} for p in prompts], sp_gen)

    L, A = {}, {}
    for (tid, S, g), ll, go in zip(meta, LLv, gouts):
        L[(tid, S)] = ll
        A[(tid, S)] = correct(go.outputs[0].text, g)

    ids = [t["id"] for t in tasks]
    half = len(ids) // 2
    train, test = ids[:half], ids[half:]

    def shapley(fn, pool):
        n = len(MENU)
        phi = {}
        for i in MENU:
            tot = 0.0
            for r in range(n):
                for S in itertools.combinations([x for x in MENU if x != i], r):
                    S = frozenset(S)
                    w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                    tot += w * (fn(S | {i}, pool) - fn(S, pool))
            phi[i] = tot
        return phi

    def meanA(S, pool):
        return st.mean([A[(t, S)] for t in pool])

    def meanL(S, pool):
        v = [L[(t, S)] for t in pool if L[(t, S)] is not None]
        return st.mean(v) if v else 0.0

    phiA = shapley(meanA, train)
    phiL = shapley(meanL, train)
    print(f"\n[{tag}] exact Shapley over {len(MENU)} tools, "
          f"{len(subsets)} subsets, train n={len(train)}")
    print(f"    {'tool':<20}{'phi^A':>10}{'phi^L':>10}")
    for i in MENU:
        print(f"    {i:<20}{phiA[i]:>10.4f}{phiL[i]:>10.4f}")
    xs = [phiA[i] for i in MENU]
    ys = [phiL[i] for i in MENU]
    print(f"    pearson (phi^A, phi^L)  {pearson(xs, ys):+.3f}")
    print(f"    spearman(phi^A, phi^L)  {spearman(xs, ys):+.3f}")
    sign = sum(1 for i in MENU if (phiA[i] > 0) != (phiL[i] > 0))
    print(f"    tools whose sign disagrees: {sign}/{len(MENU)}")

    # does a subset chosen by phi^L raise held-out accuracy
    def pick(phi):
        return frozenset(i for i in MENU if phi[i] > 0) or frozenset(MENU)
    for nm, S in [("phi^L subset", pick(phiL)), ("phi^A subset", pick(phiA)),
                  ("all four", frozenset(MENU)), ("empty", frozenset())]:
        print(f"    held-out acc, {nm:<16} {meanA(S, test):.3f}   {sorted(S)}")
    best = max(subsets, key=lambda S: meanA(S, test))
    print(f"    held-out acc, best subset    {meanA(best, test):.3f}   {sorted(best)}")
    json.dump(dict(tag=tag, phiA=phiA, phiL=phiL, menu=MENU,
                   heldout={",".join(sorted(S)): meanA(S, test) for S in subsets}),
              open(os.path.join(a.dir, f"tgb_shapley_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
