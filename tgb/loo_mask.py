"""Coalition-aware one-shot masking: leave-one-out inside a joint coalition.

The selector in `closed_loop.py` scores whole candidate coalitions and takes an
argmax. This is the other one-shot design: hand the model the *joint* coalition
S, then measure each tool's value *in that context*

    I_i(t; S) = L(y_i | q_i, o_i(S)) - L(y_i | q_i, o_i(S \\ {t}))

and keep the tools that carry positive value,

    M_i = { t in S : I_i(t; S) > tau }.

Two reasons this can beat argmax-over-coalitions. It never asks the model to
judge a tool in isolation, so a downstream tool with no standalone marginal is
still measured correctly (condition 3). And it produces a genuinely per-task
mask in one parallel scoring stage -- |S|+1 teacher-forced passes, batched, no
search loop -- rather than picking from a fixed menu.

    GD_TAG=7b GD_MODEL=... python -m tgb.loo_mask --tools OCR,Calculator,CountGivenObject,GoogleSearch

Reports the accuracy of the resulting per-task masks against the fixed
coalition S (what a global greedy search returns) on the same tasks.
"""
import argparse
import collections
import json
import os
import random
import statistics as st
import tempfile
import zlib

from vllm import LLM, SamplingParams

from . import scenes
from .generate import TPL, context
from .score_dl import correct, span
from .tools import run_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb")
    ap.add_argument("--tools", default="OCR,Calculator,CountGivenObject,"
                                       "GoogleSearch")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--v4", action="store_true")
    ap.add_argument("--train", type=int, default=400,
                    help="greedy's train split size; those tasks are held out "
                         "here too so both methods are read on the same test "
                         "set -- greedy saw labels for them, this must not "
                         "score itself on them")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, keep only the first N test tasks (the train "
                         "split is unaffected); for cheaper transfer studies")
    ap.add_argument("--tau", default="0.0",
                    help="comma-separated thresholds to sweep")
    ap.add_argument("--hyp", choices=["gold", "self_S"], default="gold",
                    help="which string I_i scores. gold is an upper bound -- "
                         "it reads the answer of the task being answered. "
                         "self_S fixes on the model's own greedy answer under "
                         "the full menu, which needs no label and is right on "
                         "0.70-0.90 of tasks against y0's 0.10.")
    a = ap.parse_args()
    if a.v4 or os.environ.get("TGB_V4"):
        from . import tools_v2, tools_v4
        tools_v2.register()
        tools_v4.register()
    elif a.v2 or os.environ.get("TGB_V2"):
        from . import tools_v2
        tools_v2.register()
    tag = os.environ.get("GD_TAG", "7b")
    if a.v4 or os.environ.get("TGB_V4"):
        # match greedy's action space exactly. Comparing a per-task selector
        # over 7 tools against a global search over 15 would hand the selector
        # a smaller problem and call the difference a method win.
        from .coalitions_v4 import FULL_MENU_V4
        S = list(FULL_MENU_V4) if a.tools == ap.get_default("tools") \
            else a.tools.split(",")
    else:
        S = a.tools.split(",")
    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    # Identical split to greedy_env: `tasks[::step][:train]` is the split whose
    # labels greedy consumed. Both splits are kept and scored, because tau is a
    # free parameter and picking it by looking at test accuracy would be the
    # same overfitting greedy is being measured for -- worse, it would be
    # invisible in the table. tau is chosen on train, and the number that
    # counts is test accuracy at that tau.
    step = max(1, len(tasks) // a.train)
    tr_ids = {t["id"] for t in tasks[::step][:a.train]}
    print(f"[{os.environ.get('GD_TAG', '7b')}] "
          f"train {len(tr_ids)} / test {len(tasks) - len(tr_ids)}, |S|={len(S)}")
    if a.limit:
        # stride, never a prefix: tgb_tasks.json is ordered by family, so the
        # first N test tasks cover only the first few families and silently
        # drop the controls (no_tool, compute_only) and the hardest chains.
        te = [t for t in tasks if t["id"] not in tr_ids]
        stride = max(1, len(te) // a.limit)
        tasks = [t for t in tasks if t["id"] in tr_ids] + te[::stride][:a.limit]

    tmp = tempfile.mkdtemp(prefix="tgb_loo_")

    def ctx_of(t, mask):
        rng = random.Random(zlib.crc32(
            f"{t['id']}/{'+'.join(sorted(mask))}".encode()))
        outs, _ = run_chain(t, t["scene"], t["boxes"], set(mask), rng,
                            corrupt_tools=())
        return context(outs, [s["tool"] for s in t["plan"]])

    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sp_gen = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                            temperature=0)

    def gold_ll(pairs):
        # span(), not a standalone gold tokenization: scoring the space-less
        # first token inverts the very quantity this file differences. I_i is a
        # difference of two L's, so a per-condition artefact does not cancel --
        # it is exactly what the artefact varies with.
        jobs = [span(tok, prompt, gold, 4096) for prompt, gold in pairs]
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

    # ---- the string I_i scores.
    # gold is an upper bound: it reads the answer of the task being answered,
    # so the label budget is one per test task, not the 400 greedy spends.
    # self_S is label-free. It carries a bias worth naming: y_S was produced
    # greedily *under S*, so L(y_S|S) sits near its own maximum and removing
    # any tool can mostly only lower it -- I_i skews positive and the selector
    # drifts toward keeping everything. A positive tau absorbs that, but if
    # this run lands exactly on the fixed-S row, that is the degeneracy and
    # not a verdict on the signal.
    hyp = {t["id"]: t["gold"] for t in tasks}
    if a.hyp == "self_S":
        base = llm.generate(
            [{"prompt": TPL.format(q=t["question"], o=ctx_of(t, S))}
             for t in tasks], sp_gen)
        hyp = {t["id"]: (o.outputs[0].text.strip().split("\n")[0][:64] or "0")
               for t, o in zip(tasks, base)}
        n_ok = st.mean([correct(hyp[t["id"]], t["gold"]) for t in tasks])
        print(f"    hypothesis = own answer under S, correct on {n_ok:.3f}")

    # ---- one parallel scoring stage: S and every leave-one-out variant
    variants = [("S", S)] + [(t, [x for x in S if x != t]) for t in S]
    pairs, meta = [], []
    for t in tasks:
        for name, m in variants:
            pairs.append((TPL.format(q=t["question"], o=ctx_of(t, m)),
                          hyp[t["id"]]))
            meta.append((t["id"], name))
    LL = gold_ll(pairs)
    V = collections.defaultdict(dict)
    for (tid, name), v in zip(meta, LL):
        V[tid][name] = v

    # ---- sweep tau. Removing a needed tool costs everything (accuracy -> 0);
    # keeping an unneeded one costs almost nothing, so the threshold should be
    # conservative (tau < 0 keeps a tool unless dropping it actively *helps*).
    taus = [float(x) for x in str(a.tau).split(",")]
    all_masks, gen_prompts, gmeta = {}, [], []
    # The fixed full menu, answered on the same tasks. Without it the sweep is
    # unreadable: every tau row is only meaningful as a difference from "keep
    # everything and never choose", which is what a deployment does today.
    for t in tasks:
        all_masks[("S", t["id"])] = list(S)
        gen_prompts.append(TPL.format(q=t["question"], o=ctx_of(t, S)))
        gmeta.append(("S", t["id"]))
    for tau in taus:
        for t in tasks:
            d = V[t["id"]]
            keep = [x for x in S
                    if d.get("S") is not None and d.get(x) is not None
                    and (d["S"] - d[x]) > tau]
            if not keep:
                keep = list(S)
            all_masks[(tau, t["id"])] = keep
            gen_prompts.append(TPL.format(q=t["question"], o=ctx_of(t, keep)))
            gmeta.append((tau, t["id"]))
    gouts = llm.generate([{"prompt": p} for p in gen_prompts], sp_gen)
    gold_by = {t["id"]: t["gold"] for t in tasks}
    fam_by = {t["id"]: t["family"] for t in tasks}
    gt_by = {t["id"]: set(t["gt_tools"]) for t in tasks}

    rows = collections.defaultdict(lambda: dict(acc=[], cost=[], exact=[],
                                                rec=[], tr=[], te=[],
                                                fam=collections.defaultdict(list)))
    for (tau, tid), o in zip(gmeta, gouts):
        m = all_masks[(tau, tid)]
        r = rows[tau]
        c = correct(o.outputs[0].text, gold_by[tid])
        r["acc"].append(c)
        (r["tr"] if tid in tr_ids else r["te"]).append(c)
        r["cost"].append(len(m))
        r["exact"].append(int(set(m) == gt_by[tid]))
        # v4 added `no_tool`, whose gt_tools is empty -- recall is undefined
        # there, not zero, and dividing would crash the whole run at report
        # time after every forward pass had already been paid for.
        if gt_by[tid]:
            r["rec"].append(len(set(m) & gt_by[tid]) / len(gt_by[tid]))
        r["fam"][fam_by[tid]].append(c)

    out = dict(tag=tag, S=S, n_test=len(tasks),
               passes_per_task=len(S) + 1, sweep={})
    print(f"\n[{tag}] leave-one-out mask, |S|={len(S)}, n={len(tasks)}, "
          f"{len(S) + 1} scoring passes per task")
    print(f"    {'tau':>7}{'train':>8}{'test':>8}{'tools':>8}{'==GT':>8}"
          f"{'GTrecall':>10}   size hist")
    for tau in ["S"] + taus:
        r = rows[tau]
        lab = "fixed S" if tau == "S" else f"{tau:.2f}"
        out["sweep"][str(tau)] = dict(
            acc=st.mean(r["acc"]), train=st.mean(r["tr"]), test=st.mean(r["te"]),
            tools=st.mean(r["cost"]), exact_gt_match=st.mean(r["exact"]),
            gt_recall=st.mean(r["rec"]) if r["rec"] else None,
            by_family={f: st.mean(v) for f, v in r["fam"].items()},
            masks={tid: all_masks[(tau, tid)] for tid in gold_by})
        print(f"    {lab:>7}{st.mean(r['tr']):>8.3f}{st.mean(r['te']):>8.3f}"
              f"{st.mean(r['cost']):>8.2f}{st.mean(r['exact']):>8.1%}"
              f"{st.mean(r['rec']) if r['rec'] else float('nan'):>10.2f}   "
              f"{dict(sorted(collections.Counter(r['cost']).items()))}")

    # tau selected on train, read on test -- the only number comparable to
    # greedy's test accuracy, which was also produced by fitting on these 400.
    pick = max(taus, key=lambda x: st.mean(rows[x]["tr"]))
    out["selected"] = dict(tau=pick, train=st.mean(rows[pick]["tr"]),
                           test=st.mean(rows[pick]["te"]),
                           tools=st.mean(rows[pick]["cost"]),
                           fixed_S_test=st.mean(rows["S"]["te"]))
    print(f"\n    tau={pick:+.2f} chosen on train -> TEST {st.mean(rows[pick]['te']):.3f}"
          f"   (fixed S test {st.mean(rows['S']['te']):.3f},"
          f" {st.mean(rows[pick]['cost']):.2f} tools/task)")
    # keep the gold-hypothesis results: they are the upper bound this run is
    # measured against, and overwriting them would cost four model-hours
    suffix = "" if a.hyp == "gold" else f"_{a.hyp}"
    p = os.path.join(a.dir, f"tgb_loo_{tag}{suffix}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
