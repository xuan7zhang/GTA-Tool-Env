"""Label-free coalition selection: no gold anywhere in the method.

Gold is still used to *evaluate* -- to say whether the coalition a method chose
produced the right answer -- but never inside a method.

What is already known, from v2, and is carried here as controls:

    own-answer likelihood   fails (7B 0.420 against 0.447 for doing nothing);
                            across coalitions it scores *different strings*, so
                            subtracting a baseline cancels in the argmax
    fixed-y0 likelihood     mixed (7B 0.409, Llama 0.555); y0 is the no-tool
                            answer, almost always wrong, so the rule is argmin
    self-consistency K=8    works (7B 0.550) but costs ~70 generations/task

Two new methods, both motivated by exactly those failures:

  CROSS-COALITION CONSENSUS. Self-consistency draws its diversity from
  sampling temperature at K generations per candidate. Draw it from the
  *environments* instead: generate once under each candidate and pick the
  candidate whose answer the most other candidates agree with. The generations
  were needed anyway, so the marginal cost is nil.

  PSEUDO-GOLD dL. Take the consensus answer as a pseudo-label and run the dL
  machinery against it. This keeps the one property that made gold-dL work and
  L_self fail -- the scored string is the *same* across all candidates -- while
  using a string that is far more often right than y0 is.

    GD_TAG=7b GD_MODEL=... python -m tgb.labelfree --dir .../tgb3
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

from . import scenes, tools_v2, tools_v3
from .generate import TPL, context
from .score_dl import correct
from .tools import run_chain

tools_v2.register()
tools_v3.register()


def norm(t):
    m = re.search(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    return m.group() if m else t.strip().lower()[:24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb3")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--k", type=int, default=4,
                    help="samples for the self-consistency baseline")
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")
    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    if a.n:
        tasks = tasks[::max(1, len(tasks) // a.n)][:a.n]
    tmp = tempfile.mkdtemp(prefix="tgb_lf_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    from .families_v3 import FAMILIES_V3, UNION_TOOLS_V3
    CHAINS = {}
    for f in FAMILIES_V3:
        ex = next((t for t in tasks if t["family"] == f), None)
        if ex is not None:
            CHAINS[f] = ex["gt_tools"]
    # gold-free candidate menu: the chains the environment supports, the union,
    # and the empty mask. Identical for every task.
    cand = ["none", "union"] + [f"chain_{f}" for f in CHAINS]
    tools_of = dict({"none": [], "union": UNION_TOOLS_V3},
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
    GEN = int(os.environ.get("GD_GENTOK", "64"))
    sp_gen = SamplingParams(max_tokens=GEN, temperature=0, logprobs=0)
    sp_k = SamplingParams(n=a.k, max_tokens=GEN, temperature=0.7, top_p=0.95,
                          seed=7)

    CTX = {(t["id"], c): ctx_of(t, tools_of[c]) for t in tasks for c in cand}
    prompts, meta = [], []
    for t in tasks:
        for c in cand:
            prompts.append(TPL.format(q=t["question"], o=CTX[(t["id"], c)]))
            meta.append((t["id"], c))

    # ---- one greedy generation per (task, candidate): answers + own-confidence
    g = llm.generate([{"prompt": p} for p in prompts], sp_gen)
    ANS, CONF, ACC = (collections.defaultdict(dict),
                      collections.defaultdict(dict),
                      collections.defaultdict(dict))
    gold_by = {t["id"]: t["gold"] for t in tasks}
    for (tid, c), o in zip(meta, g):
        out = o.outputs[0]
        lps = [next(iter(s.values())) for s in (out.logprobs or []) if s]
        lps = [x.logprob if hasattr(x, "logprob") else float(x) for x in lps]
        ANS[tid][c] = norm(out.text)
        CONF[tid][c] = (sum(lps) / len(lps)) if lps else -99
        ACC[tid][c] = correct(out.text, gold_by[tid])

    # ---- consensus answer per task, and each candidate's agreement with it
    CONS, AGREE = {}, collections.defaultdict(dict)
    for t in tasks:
        tid = t["id"]
        votes = collections.Counter(ANS[tid][c] for c in cand if c != "none")
        top, n = votes.most_common(1)[0]
        CONS[tid] = top
        for c in cand:
            AGREE[tid][c] = sum(1 for d in cand
                                if d != c and ANS[tid][d] == ANS[tid][c])

    # ---- pseudo-gold dL: score the consensus string under every candidate
    def score_fixed(pairs):
        jobs = []
        for p, s in pairs:
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

    Lp = score_fixed([(p, CONS[tid]) for p, (tid, c) in zip(prompts, meta)])
    PL = collections.defaultdict(dict)
    for (tid, c), v in zip(meta, Lp):
        PL[tid][c] = v if v is not None else -99
    DPL = {tid: {c: PL[tid][c] - PL[tid]["none"] for c in cand} for tid in PL}

    # ---- self-consistency baseline, K samples per candidate
    gk = llm.generate([{"prompt": p} for p in prompts], sp_k)
    SC = collections.defaultdict(dict)
    for (tid, c), o in zip(meta, gk):
        v = [norm(x.text) for x in o.outputs]
        SC[tid][c] = collections.Counter(v).most_common(1)[0][1] / len(v)

    T = {t["id"]: t for t in tasks}
    nt = {c: len(tools_of[c]) for c in cand}

    def ev(pick):
        acc = [ACC[i][pick(i)] for i in ACC]
        cost = [nt[pick(i)] for i in ACC]
        return st.mean(acc), st.mean(cost)

    print(f"\n[{tag}] label-free selection, {len(ACC)} tasks, "
          f"{len(cand)} candidates (no gold in any method)")
    print(f"    {'method':<34}{'acc':>8}{'tools':>8}  cost")
    rows = [
        ("cross-coalition consensus", lambda i: max(
            cand, key=lambda c: (AGREE[i][c], -nt[c])), f"{len(cand)} gen (free)"),
        ("pseudo-gold dL (consensus)", lambda i: max(
            cand, key=lambda c: DPL[i][c]), f"{len(cand)} gen + {len(cand)} prefill"),
        ("pseudo-gold dL, cost-aware", lambda i: min(
            [c for c in cand if DPL[i][c] >= max(DPL[i].values()) - 0.10],
            key=lambda c: nt[c]), "same"),
        (f"self-consistency K={a.k}", lambda i: max(
            cand, key=lambda c: SC[i][c]), f"{len(cand)*a.k} gen"),
        ("own-answer confidence", lambda i: max(
            cand, key=lambda c: CONF[i][c]), f"{len(cand)} gen"),
        ("fixed union (do nothing)", lambda i: "union", "0"),
        ("no tools at all", lambda i: "none", "0"),
    ]
    for nm, fn, cost in rows:
        acc, k = ev(fn)
        print(f"    {nm:<34}{acc:>8.3f}{k:>8.2f}  {cost}")
    orc, ork = ev(lambda i: max(cand, key=lambda c: (ACC[i][c], -nt[c])))
    gt, _ = ev(lambda i: f"chain_{T[i]['family']}"
               if f"chain_{T[i]['family']}" in cand else "union")
    print(f"    {'ORACLE routing (gold)':<34}{gt:>8.3f}")
    print(f"    {'per-task oracle (gold)':<34}{orc:>8.3f}{ork:>8.2f}")
    print(f"\n    consensus answer is correct on "
          f"{st.mean([correct(CONS[i], gold_by[i]) for i in CONS]):.3f} of tasks"
          f"  (y_none is correct on "
          f"{st.mean([ACC[i]['none'] for i in ACC]):.3f})")

    # per family, so a method that only works on the OCR-style tasks is visible
    print(f"\n    {'family':<18}{'consensus':>11}{'pseudo-dL':>11}"
          f"{'union':>8}{'oracle':>8}")
    for f in sorted({t["family"] for t in tasks}):
        ids = [t["id"] for t in tasks if t["family"] == f]
        def e(pick):
            return st.mean([ACC[i][pick(i)] for i in ids])
        print(f"    {f:<18}"
              f"{e(lambda i: max(cand, key=lambda c: (AGREE[i][c], -nt[c]))):>11.3f}"
              f"{e(lambda i: max(cand, key=lambda c: DPL[i][c])):>11.3f}"
              f"{e(lambda i: 'union'):>8.3f}"
              f"{e(lambda i: max(cand, key=lambda c: ACC[i][c])):>8.3f}")

    json.dump({i: dict(consensus=CONS[i], agree=AGREE[i], dpl=DPL[i],
                       sc=SC[i], conf=CONF[i], acc=ACC[i]) for i in ACC},
              open(os.path.join(a.dir, f"tgb_labelfree_{tag}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
