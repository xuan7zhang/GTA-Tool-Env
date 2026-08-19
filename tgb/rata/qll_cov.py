r"""QLL-Cov+ -- query-likelihood coverage tool-set selection on TGB-v2.

One mask per task *type*, fixed ahead of time from a handful of probe requests,
then applied unchanged to every later request of that type. No labels, no
generation, no tool execution anywhere in the selection path.

    a(q,d)      = [log p(q | doc(d)) - log p(q | -)] / |q|
    prefilter   greedy on U(S) = mean_j max_{d in S} a(q_j,d), keep M = 2B
    a_loo(q,d)  = [log p(q | M) - log p(q | M \ {d})] / |q|
    X+          = alpha * a + (1-alpha) * a_loo
    S*          = argmax_{S subset M, |S|=B} mean_j max_{d in S} X+(q_j,d)

The reading is inverted on purpose: the tool document is the context and the
*request* is what gets scored, so a tool wins by making the user's phrasing
more predictable rather than by being topically similar to it. That distinction
is load-bearing here -- Stage 1 measured cosine retrieval at 0.135/0.121 end-
task accuracy against a 0.447 do-nothing union, because in this environment the
most topically related tool is usually the most harmful one.

Granularity note. S* is per family, not per question: 8 masks, not 720. That
places the method between the global greedy mask (1 mask, 0.438) and per-
question selection, and it caps the method at oracle_chain (0.571 on 7B) rather
than at the per-task oracle (0.661). Both ceilings are reported.

alpha=0.5, M=2B and 3 probe requests per task are fixed in advance. alpha=1.0
(solo only) and alpha=0.0 (LOO only) are reported as diagnostics because an
earlier run of this idea on BFCL found the LOO term to be a negative
contribution; they reuse the same cached scores and cost nothing.

    GD_TAG=7b GD_MODEL=... python -m tgb.rata.qll_cov
"""
import argparse
import collections
import itertools
import json
import os
import random
import statistics as st
import tempfile
import zlib

from vllm import LLM, SamplingParams

from .. import scenes, tools_v2
from ..add_chains_v2 import CHAINS
from ..families_v2 import NUISANCE, UNION_TOOLS
from ..generate import TPL, context
from ..score_dl import correct
from ..tools import IMAGE_TOOLS, run_chain
from .tool_specs import SPECS, doc_text

tools_v2.register()

MENU = UNION_TOOLS + NUISANCE + IMAGE_TOOLS
OUT = "/datasets/omni_pretraining/gta2/results/taco/tgb2/rata_set_ports"
TGB2 = "/datasets/omni_pretraining/gta2/results/taco/tgb2"
FAM = ["f1_extract_compute", "f2_visual_reason", "f3_retrieve_reason",
       "gauge_over", "temp_over", "equation_short", "timetable_over",
       "invoice_currency"]

HDR1 = "Available tool:\n{d}\n\n"
HDRN = "Available tools:\n{d}\n\n"
QPRE = "User request: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--probes", type=int, default=3)
    ap.add_argument("--budgets", default="2,3,4,5")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--anon", action="store_true")
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b") + ("_anon" if a.anon else "")
    BUDGETS = [int(x) for x in a.budgets.split(",")]

    # Split is a property of the benchmark, not of the model, so it is
    # recomputed from the task list with the rule Stage 0 and Stage 1 used
    # rather than read back out of a model-tagged artefact.
    tasks_all = json.load(open(f"{TGB2}/tgb_tasks.json"))
    by_fam = collections.defaultdict(list)
    for t in tasks_all:
        by_fam[t["family"]].append(t)
    per = a.n // len(by_fam)
    tasks = [t for f in sorted(by_fam) for t in by_fam[f][:per]]
    split = {}
    for k, t in enumerate(tasks):
        split[t["id"]] = ("test" if k % 5 >= 2 else
                          ("dev" if k % 5 == 1 else "train"))
    T = {t["id"]: t for t in tasks}
    test = [i for i in split if split[i] == "test"]
    dev = [i for i in split if split[i] == "dev"]

    # probe requests: the first `--probes` TRAIN questions of each family
    probes = {}
    for f in FAM:
        cand = [i for i in split if split[i] == "train" and T[i]["family"] == f]
        probes[f] = sorted(cand)[:a.probes]
    print(f"[{tag}] {len(FAM)} task types, {a.probes} probes each, "
          f"pool P={len(MENU)}, test n={len(test)}")

    names = {t: (f"T{k:02d}" if a.anon else t) for k, t in enumerate(MENU)}
    SPEC = {s["tool_name"]: s for s in SPECS}

    def doc(d):
        s = dict(SPEC[d])
        s["tool_name"] = names[d]
        return doc_text(s)

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)

    NFWD = collections.Counter()
    NPTOK = collections.Counter()

    def score_q(prefixes, qtexts, stage):
        """mean log p(q | prefix), q scored token-by-token."""
        jobs = []
        for pre, q in zip(prefixes, qtexts):
            qi = tok(QPRE + q, add_special_tokens=False)["input_ids"]
            pi = tok(pre, add_special_tokens=False)["input_ids"][:4096 - len(qi) - 1]
            jobs.append((pi + qi, len(pi), len(qi)))
            NFWD[stage] += 1
            NPTOK[stage] += len(pi) + len(qi)
        outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
        res = []
        for (full, plen, nq), o in zip(jobs, outs):
            s, n = 0.0, 0
            for k in range(plen, plen + nq):
                d = o.prompt_logprobs[k]
                lp = d.get(full[k]) if d else None
                if lp is not None:
                    s += lp.logprob if hasattr(lp, "logprob") else float(lp)
                    n += 1
            res.append(s / max(1, n))
        return res

    # ------------------------------------------------- step 1: solo scores
    allq = [(f, i) for f in FAM for i in probes[f]]
    qtext = {i: T[i]["question"] for _, i in allq}
    base = dict(zip([i for _, i in allq],
                    score_q(["" for _ in allq], [qtext[i] for _, i in allq],
                            "step1_base")))
    A = collections.defaultdict(dict)          # A[i][tool] = a(q,d)
    for d in MENU:
        pre = [HDR1.format(d=doc(d)) for _ in allq]
        v = score_q(pre, [qtext[i] for _, i in allq], "step1_solo")
        for (_, i), x in zip(allq, v):
            A[i][d] = x - base[i]

    def cover(S, qs, X):
        return st.mean(max(X[i][d] for d in S) for i in qs)

    # -------------------------------------- steps 2-4, per family, per budget
    SEL, DIAG = collections.defaultdict(dict), {}
    for f in FAM:
        qs = probes[f]
        for B in BUDGETS:
            M = min(2 * B, len(MENU))
            S = []
            while len(S) < M:                                    # step 2
                best = max((d for d in MENU if d not in S),
                           key=lambda d: (cover(S + [d], qs, A), -MENU.index(d)))
                S.append(best)
            Mset = sorted(S, key=MENU.index)
            # step 3: leave-one-out inside the shortlist only
            full_pre = HDRN.format(d="\n".join(doc(d) for d in Mset))
            Lfull = dict(zip(qs, score_q([full_pre] * len(qs),
                                         [qtext[i] for i in qs], "step3_full")))
            LOO = collections.defaultdict(dict)
            for d in Mset:
                pre = HDRN.format(d="\n".join(doc(x) for x in Mset if x != d))
                v = score_q([pre] * len(qs), [qtext[i] for i in qs], "step3_loo")
                for i, x in zip(qs, v):
                    LOO[i][d] = Lfull[i] - x
            for alpha, nm in [(a.alpha, "mix"), (1.0, "solo"), (0.0, "loo")]:
                X = {i: {d: alpha * A[i][d] + (1 - alpha) * LOO[i][d]
                         for d in Mset} for i in qs}
                best = max(itertools.combinations(Mset, B),
                           key=lambda c: (cover(list(c), qs, X),
                                          tuple(-MENU.index(x) for x in c)))
                SEL[(nm, B)][f] = sorted(best)
            DIAG[f"{f}|B{B}"] = {"shortlist": Mset,
                                 "solo_rank": sorted(MENU, key=lambda d: -st.mean(
                                     A[i][d] for i in qs))[:6]}

    # ------------------------------------------------------ evaluate on test
    tmp = tempfile.mkdtemp(prefix="qll_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))
    REF = {"union9": {f: sorted(UNION_TOOLS) for f in FAM},
           "oracle_chain": {f: sorted(CHAINS[f]) for f in FAM}}
    ALL = dict(SEL)
    ALL.update({(k, 0): v for k, v in REF.items()})
    need = sorted({(i, tuple(m[T[i]["family"]]))
                   for m in ALL.values() for i in test + dev})
    print(f"  distinct (task, mask) to execute: {len(need)}")
    sp_gen = SamplingParams(max_tokens=64, temperature=0)
    prompts, ntok = [], {}
    for i, S in need:
        t = T[i]
        rng = random.Random(zlib.crc32(f"{i}/{'+'.join(S)}".encode()))
        outs, _ = run_chain(t, t["scene"], t["boxes"], set(S), rng,
                            corrupt_tools=())
        ctx = context(outs, [x["tool"] for x in t["plan"]])
        ntok[(i, S)] = len(tok(ctx, add_special_tokens=False)["input_ids"])
        prompts.append(TPL.format(q=t["question"], o=ctx))
    gens = llm.generate([{"prompt": p} for p in prompts], sp_gen)
    ACC = {k: correct(g.outputs[0].text, T[k[0]]["gold"])
           for k, g in zip(need, gens)}

    def ev(m, ids):
        acc = st.mean(ACC[(i, tuple(m[T[i]["family"]]))] for i in ids)
        S = st.mean(len(m[T[i]["family"]]) for i in ids)
        tk = st.mean(ntok[(i, tuple(m[T[i]["family"]]))] for i in ids)
        tr = st.mean(int(set(CHAINS[T[i]["family"]]) <= set(m[T[i]["family"]]))
                     for i in ids)
        f1 = []
        for i in ids:
            g, s = set(CHAINS[T[i]["family"]]), set(m[T[i]["family"]])
            r, p = len(g & s) / len(g), len(g & s) / max(1, len(s))
            f1.append(0 if r + p == 0 else 2 * r * p / (r + p))
        return dict(acc=acc, mask=S, tok=tk, tracc=tr, set_f1=st.mean(f1))

    res = {f"{k[0]}_B{k[1]}" if k[1] else k[0]: {
        "masks": {f: v[f] for f in FAM}, "test": ev(v, test), "dev": ev(v, dev)}
        for k, v in ALL.items()}
    res["_cost"] = {"forward_passes": dict(NFWD), "prefill_tokens": dict(NPTOK),
                    "generations_in_selection": 0, "tool_executions": 0,
                    "test_time_selection_cost": 0}
    json.dump(res, open(f"{OUT}/eval/qll_cov_{tag}.json", "w"), indent=1)
    json.dump(DIAG, open(f"{OUT}/set_scorer/qll_shortlists_{tag}.json", "w"),
              indent=1)

    print(f"\n[{tag}] QLL-Cov+  test n={len(test)}  (per-family masks)")
    print(f"  {'method':<16} {'testAcc':>8} {'devAcc':>7} {'|S|':>5} {'TRACC':>6} "
          f"{'setF1':>6} {'tok':>6}")
    for k in sorted(res):
        if k.startswith("_"):
            continue
        v = res[k]
        print(f"  {k:<16} {v['test']['acc']:8.3f} {v['dev']['acc']:7.3f} "
              f"{v['test']['mask']:5.1f} {v['test']['tracc']:6.3f} "
              f"{v['test']['set_f1']:6.3f} {v['test']['tok']:6.1f}")
    print(f"\n  selection cost: {sum(NFWD.values())} forward passes, "
          f"{sum(NPTOK.values())/1e3:.1f}k prefill tokens, "
          f"0 generations, 0 tool executions, 0 test-time cost")
    print(f"  by stage: {dict(NFWD)}")
    print("\n  chosen masks (alpha=%.2f):" % a.alpha)
    for f in FAM:
        for B in BUDGETS:
            if B == 3:
                print(f"    {f:<22} B=3 {SEL[('mix', 3)][f]}   GT={sorted(CHAINS[f])}")


if __name__ == "__main__":
    main()
