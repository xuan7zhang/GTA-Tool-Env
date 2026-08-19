"""Gold-free likelihood metrics for coalition selection.

No method here touches the answer. Gold is used only to *evaluate* -- to say
whether the coalition a metric picked actually produced the right answer.

Four likelihood metrics, differing only in **which string is scored**, which is
the whole story of what works:

    L_self(S)   = log p(y_S | x, o(S))            the model's own answer under S
    D_y0(S)     = log p(y0 | x, o(S)) - log p(y0 | x)      y0 = the no-tool answer
    D_cons(S)   = log p(y~ | x, o(S)) - log p(y~ | x)      y~ = cross-coalition mode
    conf-margin                                    (reported, and degenerate)

L_self is the natural thing to try and it fails, for a reason that is not
incidental: across candidates it scores *different strings*, so the comparison
is not of one event, and subtracting a per-task baseline cancels inside the
argmax. D_y0 and D_cons keep the string fixed, which restores the subtraction.

They differ in the *direction* that means "useful". y0 is almost always wrong,
so the informative coalition is the one that **refutes** it -- argmin. y~ is
more often right, so the rule is argmax. Both directions are measured rather
than assumed, because getting the sign backwards costs an order of magnitude.

Two non-likelihood gold-free baselines are included for context: majority vote
across coalitions, and self-consistency over K samples.

    GD_TAG=7b GD_MODEL=... python -m tgb.goldfree --dir .../tgb4 --v4
"""
import argparse
import bisect
import collections
import json
import os
import random
import re
import statistics as st
import tempfile
import zlib

from vllm import LLM, SamplingParams

from . import scenes
from .score_dl import span
from .generate import TPL, context
from .score_dl import correct
from .tools import run_chain


def norm(t):
    m = re.search(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    return m.group() if m else t.strip().lower()[:24]


def auroc(pos, neg):
    if not pos or not neg:
        return None
    pos = sorted(pos)
    tot = 0
    for b in neg:
        lo = bisect.bisect_left(pos, b)
        hi = bisect.bisect_right(pos, b)
        tot += lo + 0.5 * (hi - lo)
    return 1.0 - tot / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb4")
    ap.add_argument("--v4", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")

    from . import tools_v2
    tools_v2.register()
    if a.v4 or os.environ.get("TGB_V4"):
        from . import tools_v4
        from .families_v4 import FAMILIES_V4 as FAMS, UNION_TOOLS_V4 as UNION
        tools_v4.register()
    else:
        from . import tools_v3
        from .families_v3 import FAMILIES_V3 as FAMS, UNION_TOOLS_V3 as UNION
        tools_v3.register()

    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    if a.n:
        tasks = tasks[::max(1, len(tasks) // a.n)][:a.n]
    tmp = tempfile.mkdtemp(prefix="tgb_gf_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    CH = {}
    for f in FAMS:
        ex = next((t for t in tasks if t["family"] == f), None)
        if ex is not None:
            CH[f] = ex["gt_tools"]
    cand = ["none", "union"] + [f"chain_{f}" for f in CH]
    tools_of = dict({"none": [], "union": UNION},
                    **{f"chain_{f}": c for f, c in CH.items()})

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
    GEN = int(os.environ.get("GD_GENTOK", "64"))
    sp_gen = SamplingParams(max_tokens=GEN, temperature=0, logprobs=0)
    sp_k = SamplingParams(n=a.k, max_tokens=GEN, temperature=0.7, top_p=0.95,
                          seed=7)

    PR = {(t["id"], c): TPL.format(q=t["question"], o=ctx_of(t, tools_of[c]))
          for t in tasks for c in cand}
    flat = [(t["id"], c) for t in tasks for c in cand]
    prompts = [PR[k] for k in flat]
    gold_by = {t["id"]: t["gold"] for t in tasks}

    # ---- one greedy pass: the answer, its own logprob, and whether it is right
    g = llm.generate([{"prompt": p} for p in prompts], sp_gen)
    ANS, SELF, ACC = ({}, {}, {})
    for (tid, c), o in zip(flat, g):
        out = o.outputs[0]
        lps = [next(iter(s.values())) for s in (out.logprobs or []) if s]
        lps = [x.logprob if hasattr(x, "logprob") else float(x) for x in lps]
        ANS[(tid, c)] = norm(out.text)
        SELF[(tid, c)] = (sum(lps) / len(lps)) if lps else -99
        ACC[(tid, c)] = correct(out.text, gold_by[tid])

    # ---- the two fixed hypotheses
    y0 = {t["id"]: ANS[(t["id"], "none")] or "0" for t in tasks}
    cons = {}
    for t in tasks:
        v = collections.Counter(ANS[(t["id"], c)] for c in cand if c != "none")
        cons[t["id"]] = v.most_common(1)[0][0] or "0"

    def score_fixed(strings):
        # same boundary rule as score_dl: the continuation is tokenized in
        # place, because scoring a space-less first token inverts dL
        jobs = [span(tok, PR[(tid, c)], s or "0", 4096, cap=32)
                for (tid, c), s in zip(flat, strings)]
        outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
        res = {}
        for (tid, c), (full, plen, ng), o in zip(flat, jobs, outs):
            lps = []
            for k in range(plen, plen + ng):
                d = o.prompt_logprobs[k]
                if d is None:
                    continue
                lp = d.get(full[k])
                if lp is not None:
                    lps.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
            res[(tid, c)] = (sum(lps) / len(lps)) if lps else None
        return res

    Ly0 = score_fixed([y0[tid] for tid, _ in flat])
    Lcs = score_fixed([cons[tid] for tid, _ in flat])
    D_y0 = {k: (Ly0[k] - Ly0[(k[0], "none")]) if Ly0[k] is not None else 0.0
            for k in flat}
    D_cs = {k: (Lcs[k] - Lcs[(k[0], "none")]) if Lcs[k] is not None else 0.0
            for k in flat}

    # ---- self-consistency baseline
    gk = llm.generate([{"prompt": p} for p in prompts], sp_k)
    SC = {}
    for (tid, c), o in zip(flat, gk):
        v = [norm(x.text) for x in o.outputs]
        SC[(tid, c)] = collections.Counter(v).most_common(1)[0][1] / len(v)
    AGREE = {(t["id"], c): sum(1 for d in cand
                               if d != c and ANS[(t["id"], d)] == ANS[(t["id"], c)])
             for t in tasks for c in cand}

    T = {t["id"]: t for t in tasks}
    nt = {c: len(tools_of[c]) for c in cand}
    ids = [t["id"] for t in tasks]

    def ev(pick):
        return (st.mean([ACC[(i, pick(i))] for i in ids]),
                st.mean([nt[pick(i)] for i in ids]))

    def au(metric):
        p = [metric[(i, c)] for i in ids for c in cand if ACC[(i, c)] == 1]
        n = [metric[(i, c)] for i in ids for c in cand if ACC[(i, c)] == 0]
        return auroc(p, n)

    print(f"\n[{tag}] gold-free metrics, {len(ids)} tasks, {len(cand)} candidates")
    print(f"    {'metric':<34}{'AUROC':>8}{'sel.acc':>9}{'tools':>7}  rule")
    rows = [
        ("L_self (own answer)", SELF,
         lambda i: max(cand, key=lambda c: SELF[(i, c)]), "argmax"),
        ("D_y0 (fixed no-tool answer)", {k: -v for k, v in D_y0.items()},
         lambda i: min(cand, key=lambda c: D_y0[(i, c)]), "argMIN"),
        ("D_cons (fixed consensus answer)", D_cs,
         lambda i: max(cand, key=lambda c: D_cs[(i, c)]), "argmax"),
    ]
    for nm, metric, pick, rule in rows:
        acc, k = ev(pick)
        v = au(metric)
        print(f"    {nm:<34}{'n/a' if v is None else format(v, '.3f'):>8}"
              f"{acc:>9.3f}{k:>7.2f}  {rule}")
    for nm, pick in [("cross-coalition majority",
                      lambda i: max(cand, key=lambda c: (AGREE[(i, c)], -nt[c]))),
                     (f"self-consistency K={a.k}",
                      lambda i: max(cand, key=lambda c: SC[(i, c)])),
                     ("fixed union (do nothing)", lambda i: "union"),
                     ("no tools", lambda i: "none")]:
        acc, k = ev(pick)
        print(f"    {nm:<34}{'':>8}{acc:>9.3f}{k:>7.2f}")
    orc, ork = ev(lambda i: max(cand, key=lambda c: (ACC[(i, c)], -nt[c])))
    gt, _ = ev(lambda i: f"chain_{T[i]['family']}"
               if f"chain_{T[i]['family']}" in cand else "union")
    print(f"    {'ORACLE routing (uses gold)':<34}{'':>8}{gt:>9.3f}")
    print(f"    {'per-task oracle (uses gold)':<34}{'':>8}{orc:>9.3f}{ork:>7.2f}")
    print(f"\n    hypothesis quality: y0 correct {st.mean([correct(y0[i], gold_by[i]) for i in ids]):.3f}"
          f"   consensus correct {st.mean([correct(cons[i], gold_by[i]) for i in ids]):.3f}")

    json.dump({f"{i}|{c}": dict(self=SELF[(i, c)], dy0=D_y0[(i, c)],
                                dcons=D_cs[(i, c)], sc=SC[(i, c)],
                                agree=AGREE[(i, c)], acc=ACC[(i, c)])
               for i in ids for c in cand},
              open(os.path.join(a.dir, f"tgb_goldfree_{tag}.json"), "w"))


if __name__ == "__main__":
    main()
