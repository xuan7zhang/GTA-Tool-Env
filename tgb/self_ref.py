"""Gold-free selection with a *fixed* hypothesis string.

Scoring the model's own answer per coalition fails (tgb/deploy_select.py:
7B 0.420, below the do-nothing union at 0.447), and the reason is structural --
across coalitions it compares different strings, so subtracting a baseline
cancels in the argmax and cannot fix it.

The minimal repair keeps the string fixed. Take the answer the model gives with
*no tools*, y0, and ask each coalition how it moves the likelihood of that one
string:

    D(S) = log p(y0 | x, o(S)) - log p(y0 | x)

Now every candidate is scored on the same event, so the subtraction means
something again, and no label is needed anywhere. y0 is almost always wrong
here (the no-tool condition scores 0.000), so the coalition that *refutes* it
hardest is the one carrying real information -- the decision rule is argmin,
not argmax. All three rules are measured rather than assumed, along with the
cheapest signal of all: whether the answer changed at all.

    GD_TAG=7b GD_MODEL=... python -m tgb.self_ref
"""
import argparse
import collections
import json
import os
import random
import re
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


def norm(t):
    m = re.search(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    return m.group() if m else t.strip().lower()[:24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb2")
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")
    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    tmp = tempfile.mkdtemp(prefix="tgb_sr_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    cand = ["union"] + [f"chain_{f}" for f in CHAINS]
    tools_of = dict({"union": UNION_TOOLS},
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
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sp_gen = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                            temperature=0)

    # ---- y0: the model's answer with no tools at all
    p0 = [TPL.format(q=t["question"], o="") for t in tasks]
    g0 = llm.generate([{"prompt": p} for p in p0], sp_gen)
    y0 = [o.outputs[0].text.strip() or "0" for o in g0]

    def score_fixed(prompts, strings):
        jobs = []
        for p, s in zip(prompts, strings):
            gi = tok(s, add_special_tokens=False)["input_ids"][:32] or \
                tok("0", add_special_tokens=False)["input_ids"]
            pi = tok(p, add_special_tokens=False)["input_ids"][:4096 - len(gi) - 1]
            jobs.append((pi + gi, len(pi), len(gi)))
        outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
        res = []
        for (full, plen, ng), o in zip(jobs, outs):
            lps = []
            for k in range(plen, plen + ng):
                d = o.prompt_logprobs[k]
                if d is None:
                    continue
                lp = d.get(full[k])
                if lp is not None:
                    lps.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
            res.append(sum(lps) / len(lps) if lps else None)
        return res

    base = score_fixed(p0, y0)

    D, ACC, CHANGED = (collections.defaultdict(dict),
                       collections.defaultdict(dict),
                       collections.defaultdict(dict))
    y0by = {t["id"]: y0[i] for i, t in enumerate(tasks)}
    for c in cand:
        ctxs = [ctx_of(t, tools_of[c]) for t in tasks]
        pr = [TPL.format(q=t["question"], o=x) for t, x in zip(tasks, ctxs)]
        Ls = score_fixed(pr, y0)
        gs = llm.generate([{"prompt": p} for p in pr], sp_gen)
        for t, b, l, o in zip(tasks, base, Ls, gs):
            D[t["id"]][c] = (l - b) if (l is not None and b is not None) else 0.0
            ACC[t["id"]][c] = correct(o.outputs[0].text, t["gold"])
            CHANGED[t["id"]][c] = int(norm(o.outputs[0].text)
                                      != norm(y0by[t["id"]]))
    T = {t["id"]: t for t in tasks}
    nt = {c: len(tools_of[c]) for c in cand}

    def ev(pick):
        acc = [ACC[i][pick(i)] for i in D]
        cost = [nt[pick(i)] for i in D]
        hit = sum(pick(i) == f"chain_{T[i]['family']}" for i in D)
        return st.mean(acc), st.mean(cost), hit / len(D)

    print(f"\n[{tag}] fixed-hypothesis selection on {len(D)} tasks, no gold")
    rules = {
        "argmin D  (most refuting)": lambda i: min(cand, key=lambda c: D[i][c]),
        "argmax |D| (most moving)": lambda i: max(cand, key=lambda c: abs(D[i][c])),
        "argmax D  (most supporting)": lambda i: max(cand, key=lambda c: D[i][c]),
    }
    for nm, fn in rules.items():
        acc, cost, hit = ev(fn)
        print(f"    {nm:<30} {acc:.3f}   tools {cost:.2f}   routing {hit:.0%}")
    # cheapest signal of all: did the tool output change the answer at all
    acc, cost, hit = ev(lambda i: max(cand, key=lambda c: (CHANGED[i][c],
                                                           -D[i][c])))
    print(f"    {'answer changed, tie-break D':<30} {acc:.3f}   "
          f"tools {cost:.2f}   routing {hit:.0%}")
    unacc, uncost, _ = ev(lambda i: "union")
    gtacc, _, _ = ev(lambda i: f"chain_{T[i]['family']}")
    print(f"    {'fixed union (do nothing)':<30} {unacc:.3f}   tools {uncost:.2f}")
    print(f"    {'oracle routing':<30} {gtacc:.3f}")
    json.dump({i: D[i] for i in D},
              open(os.path.join(a.dir, f"tgb_selfref_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
