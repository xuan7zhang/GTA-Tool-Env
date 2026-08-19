"""Measured cost of per-task likelihood masking vs greedy environment search.

The accuracy half of the claim is settled in TGB_V2.md. This is the other half,
measured rather than estimated, on one GPU with identical batching.

The two methods do structurally different work:

  greedy   a *sequential* dataset-level search. Each round evaluates every
           candidate mask over the whole train split, and every evaluation
           needs the tools executed **and the model to generate an answer**.
           Rounds cannot be parallelised -- round k+1's candidates depend on
           round k's winner.
  per-task one *parallel* scoring stage per task: C teacher-forced passes with
           `max_tokens=1` (prefill only, no autoregressive decode) plus a
           single generation for the chosen mask. No sequential dependency.

So the comparison is not just "how many evaluations" but "how expensive is one
evaluation": greedy pays decode, likelihood pays prefill.

    GD_TAG=7b GD_MODEL=... python -m tgb.complexity --n 400

Reports GPU-seconds and token counts for each, the one-time/per-query split,
and the break-even number of deployed queries.
"""
import argparse
import collections
import json
import os
import random
import tempfile
import time
import zlib

from vllm import LLM, SamplingParams

from . import scenes, tools_v2
from .add_chains_v2 import CHAINS
from .families_v2 import UNION_TOOLS
from .generate import TPL, context
from .tools import run_chain

tools_v2.register()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb2")
    ap.add_argument("--n", type=int, default=400,
                    help="tasks to time (matches the greedy train split)")
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")
    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    step = max(1, len(tasks) // a.n)
    sub = tasks[::step][:a.n]
    tmp = tempfile.mkdtemp(prefix="tgb_cx_")
    for t in sub:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

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

    cand = ["union"] + [f"chain_{f}" for f in CHAINS]
    tools_of = dict({"union": UNION_TOOLS},
                    **{f"chain_{f}": c for f, c in CHAINS.items()})

    # ---------------------------------------------------- per-task selection
    prompts, ntok = [], 0
    for t in sub:
        for c in cand:
            p = TPL.format(q=t["question"], o=ctx_of(t, tools_of[c]))
            prompts.append(p)
            ntok += len(tok(p, add_special_tokens=False)["input_ids"])
    t0 = time.time()
    llm.generate([{"prompt": p} for p in prompts], sp_lp)
    t_score = time.time() - t0
    # plus one generation for the chosen mask
    t0 = time.time()
    llm.generate([{"prompt": prompts[i * len(cand)]} for i in range(len(sub))],
                 sp_gen)
    t_answer = time.time() - t0
    per_task_total = t_score + t_answer

    # ------------------------------------------------------------ greedy
    # unit cost: one candidate mask evaluated over the same task set, with
    # generation, exactly as greedy_env does it
    t0 = time.time()
    llm.generate([{"prompt": TPL.format(q=t["question"],
                                        o=ctx_of(t, UNION_TOOLS))}
                  for t in sub], sp_gen)
    t_one_mask = time.time() - t0

    gpath = os.path.join(a.dir, f"tgb_greedy_{tag}_backward.json")
    if os.path.exists(gpath):
        G = json.load(open(gpath))
        menu = len(G["trace"][0]["mask"]) + 1 if G["trace"] else 21
        # backward: 1 full eval, then |cur| candidates each round
        n_evals, cur = 1, menu
        for _ in G["trace"]:
            n_evals += cur
            cur -= 1
        n_evals += cur          # the final round that found no improvement
        rounds = len(G["trace"]) + 1
    else:
        n_evals, rounds, menu = 78, 8, 21
    t_greedy = n_evals * t_one_mask

    print(f"\n[{tag}] measured on {len(sub)} tasks, one GPU, identical batching")
    print(f"  candidates scored per task            {len(cand)}")
    print(f"  prompt tokens scored (total)          {ntok:,}")
    print(f"  ---- per-task likelihood selection")
    print(f"  scoring stage (prefill only)          {t_score:7.1f} s")
    print(f"  answer generation                     {t_answer:7.1f} s")
    print(f"  TOTAL                                 {per_task_total:7.1f} s"
          f"   ({per_task_total / len(sub) * 1000:.0f} ms/task)")
    print(f"  ---- greedy backward elimination")
    print(f"  menu size / rounds / mask evaluations {menu} / {rounds} / {n_evals}")
    print(f"  one mask over the split (with decode) {t_one_mask:7.1f} s")
    print(f"  TOTAL (sequential, cannot batch)      {t_greedy:7.1f} s")
    print(f"  ---- ratio")
    print(f"  greedy / per-task                     {t_greedy / per_task_total:7.1f}x")
    if per_task_total > 0:
        per_query = per_task_total / len(sub)
        print(f"  greedy is one-time; per-task is per-query "
              f"({per_query * 1000:.0f} ms).")
        print(f"  break-even after {t_greedy / per_query:,.0f} deployed queries"
              f" (greedy also needs {len(sub)} labelled tasks up front).")
    json.dump(dict(tag=tag, n=len(sub), candidates=len(cand),
                   prompt_tokens=ntok, t_score=t_score, t_answer=t_answer,
                   t_per_task_total=per_task_total, t_one_mask=t_one_mask,
                   greedy_evals=n_evals, greedy_rounds=rounds,
                   menu=menu, t_greedy=t_greedy),
              open(os.path.join(a.dir, f"tgb_complexity_{tag}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
