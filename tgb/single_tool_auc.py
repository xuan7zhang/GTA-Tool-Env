"""Condition (3), tested with the AUROC criterion: does dL work per *tool*?

Everything measured so far scores tool *sets*. The paper argues sets are
necessary because a chained task gives a downstream tool no standalone
marginal, but it argues that from Shapley disagreement and GT-coverage, not
from the discrimination criterion the rest of the analysis uses. This closes
that gap directly: score each tool **alone** and ask the same question that
was asked of coalitions --

    score = dL({t}) = L(y* | x, o({t})) - L(y* | x)
    label = did the model answer correctly with only tool t

pooled over (task, tool) pairs, and broken out per tool. If condition (3) is
right the pooled number should fall well below the set-level AUROC, and the
per-tool breakdown should show upstream perception tools retaining signal
while downstream compute tools collapse.

Runs on the real GTA tasks (cached outputs) by default; --tgb2 runs the same
on the synthetic benchmark for contrast.

    GD_TAG=7b GD_MODEL=... python -m tgb.single_tool_auc
"""
import argparse
import bisect
import collections
import json
import os
import random
import statistics as st
import tempfile
import zlib

from vllm import LLM, SamplingParams

from .real_minimal import (BIG, TEXT_TOOLS, TPL, correct, gold_of,
                           question_of)


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
    ap.add_argument("--tgb2", action="store_true")
    ap.add_argument("--tgb4", action="store_true")
    ap.add_argument("--out", default=f"{BIG}/results/taco/real_minimal")
    ap.add_argument("--cap", type=int, default=400)
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")

    if a.tgb4:
        from . import scenes, tools_v2, tools_v4
        from .coalitions_v4 import FULL_MENU_V4
        from .generate import context
        from .tools import run_chain
        tools_v2.register()
        tools_v4.register()
        D = f"{BIG}/results/taco/tgb4"
        tasks = json.load(open(f"{D}/tgb_tasks.json"))[::4]
        tmp = tempfile.mkdtemp(prefix="tgb_st_")
        for t in tasks:
            t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))
        TOOLS_L = list(FULL_MENU_V4)

        def build(t, tools):
            rng = random.Random(zlib.crc32(
                f"{t['id']}/{'+'.join(sorted(tools))}".encode()))
            outs, _ = run_chain(t, t["scene"], t["boxes"], set(tools), rng,
                                corrupt_tools=())
            return context(outs, [s["tool"] for s in t["plan"]])
        items = [(t["id"], t["question"], t["gold"], t) for t in tasks]
        label = "TGB-v4"
    elif a.tgb2:
        from . import scenes, tools_v2
        from .families_v2 import UNION_TOOLS
        from .generate import context
        from .tools import run_chain
        tools_v2.register()
        D = f"{BIG}/results/taco/tgb2"
        tasks = json.load(open(f"{D}/tgb_tasks.json"))[::4]
        tmp = tempfile.mkdtemp(prefix="tgb_st_")
        for t in tasks:
            t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))
        TOOLS_L = UNION_TOOLS

        def build(t, tools):
            rng = random.Random(zlib.crc32(
                f"{t['id']}/{'+'.join(sorted(tools))}".encode()))
            outs, _ = run_chain(t, t["scene"], t["boxes"], set(tools), rng,
                                corrupt_tools=())
            return context(outs, [s["tool"] for s in t["plan"]])
        items = [(t["id"], t["question"], t["gold"], t) for t in tasks]
        label = "TGB-v2"
    else:
        TO = json.load(open(f"{BIG}/results/inject_opt/tool_outputs.json"))
        DS = json.load(open(f"{BIG}/data/gta_dataset/dataset.json"))
        TOOLS_L = TEXT_TOOLS
        items = [(k, question_of(DS, k), gold_of(DS, k), k)
                 for k in TO if gold_of(DS, k)]

        def build(k, tools):
            return "\n".join(f"{t}: {str(TO[k][t])[:a.cap]}" for t in tools
                             if TO[k].get(t))
        label = "real GTA"

    print(f"[{tag}] {label}: {len(items)} tasks x "
          f"({len(TOOLS_L)} single tools + none + all)")

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=8192)
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sp_gen = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                            temperature=0)

    envs = [("none", [])] + [(t, [t]) for t in TOOLS_L] + [("__all", TOOLS_L)]
    prompts, meta = [], []
    for tid, q, g, handle in items:
        for name, tools in envs:
            prompts.append(TPL.format(q=q, o=build(handle, tools)))
            meta.append((tid, name, g))

    jobs = []
    if a.tgb4:
        # v4 uses the fixed boundary handling: score exactly the gold span of
        # tok(prompt + " " + gold), never gold tokenized on its own.
        from .score_dl import span
        for p, (_, _, g) in zip(prompts, meta):
            jobs.append(span(tok, p, g, 8192))
    else:
        for p, (_, _, g) in zip(prompts, meta):
            gi = tok(g, add_special_tokens=False)["input_ids"]
            pi = tok(p, add_special_tokens=False)["input_ids"][:8192 - len(gi) - 1]
            jobs.append((pi + gi, len(pi), len(gi)))
    outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
    LL = []
    for (full, plen, ng), o in zip(jobs, outs):
        lps = []
        for i in range(plen, plen + ng):
            d = o.prompt_logprobs[i]
            if d is None:
                continue
            lp = d.get(full[i])
            if lp is not None:
                lps.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
        LL.append(sum(lps) / len(lps) if lps else None)
    gouts = llm.generate([{"prompt": p} for p in prompts], sp_gen)

    V = collections.defaultdict(dict)
    for (tid, name, g), ll, go in zip(meta, LL, gouts):
        V[tid][name] = dict(L=ll, acc=correct(go.outputs[0].text, g))

    dls, accs, per = [], [], collections.defaultdict(lambda: ([], []))
    for tid in V:
        base = V[tid]["none"]["L"]
        if base is None:
            continue
        for t in TOOLS_L:
            d = V[tid].get(t)
            if not d or d["L"] is None:
                continue
            dl = d["L"] - base
            dls.append(dl)
            accs.append(d["acc"])
            per[t][0].append(dl)
            per[t][1].append(d["acc"])
    au_single = auroc([x for x, y in zip(dls, accs) if y == 1],
                      [x for x, y in zip(dls, accs) if y == 0])
    adl, aac = [], []
    for tid in V:
        base = V[tid]["none"]["L"]
        d = V[tid].get("__all")
        if base is None or not d or d["L"] is None:
            continue
        adl.append(d["L"] - base)
        aac.append(d["acc"])
    au_all = auroc([x for x, y in zip(adl, aac) if y == 1],
                   [x for x, y in zip(adl, aac) if y == 0])

    print(f"\n[{tag}] {label}")
    print(f"  AUROC(dL -> correct), SINGLE tools pooled  {au_single:.3f}"
          f"   n={len(dls)}")
    print(f"  AUROC(dL -> correct), the all-tools env    "
          f"{au_all if au_all is None else round(au_all, 3)}   n={len(adl)}")
    print(f"  accuracy with one tool only: "
          f"{st.mean(accs):.3f}   with all tools: {st.mean(aac):.3f}")
    print(f"\n  per tool ({'tool':<28}{'AUROC':>8}{'acc':>8}{'mean dL':>10})")
    for t in TOOLS_L:
        d, ac = per[t]
        if not d:
            continue
        au = auroc([x for x, y in zip(d, ac) if y == 1],
                   [x for x, y in zip(d, ac) if y == 0])
        print(f"    {t:<28}{'n/a' if au is None else format(au, '.3f'):>8}"
              f"{st.mean(ac):>8.3f}{st.mean(d):>10.2f}")
    os.makedirs(a.out, exist_ok=True)
    suffix = "tgb4" if a.tgb4 else ("tgb2" if a.tgb2 else "real")
    json.dump({t: dict(dL=per[t][0], acc=per[t][1]) for t in TOOLS_L},
              open(os.path.join(
                  a.out, f"single_tool_{suffix}_{tag}.json"),
                  "w"), indent=1)


if __name__ == "__main__":
    main()
