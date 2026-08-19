"""Is the per-task mask actually per-task, or is it a per-family mask in disguise?

    GD_TAG=7b GD_MODEL=... python -m tgb.granularity

The self_S selector emits 63-268 distinct masks and hits the task's exact
ground-truth chain on 69-82% of tasks, which reads as task-conditioned. But
`gt_tools` is a function of the family -- all 400 tasks in a family share one
chain -- so that number is really "identifies the family", and the family is
recoverable from a single keyword with no model at all. Collapsing the masks to
one per family already covers 79-91% of the per-task choices.

So the claim "task-level" rests entirely on the residual 9-21%, and this asks
whether that residual pays. Three deployments of the *same* selector output,
differing only in the granularity they are allowed to keep:

    global   one mask, the mode over the training split
    family   one mask per family, the mode within each family on train
    task     the per-task mask itself

Modes are taken on train only: reading them off test would let the coarse
policies borrow test labels the fine one never sees. If family matches task,
the per-task machinery is buying nothing a keyword classifier could not.
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

from . import scenes, tools_v2, tools_v4
from .generate import TPL, context
from .score_dl import correct
from .tools import run_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb4")
    ap.add_argument("--hyp", default="self_S")
    ap.add_argument("--train", type=int, default=400)
    a = ap.parse_args()
    tools_v2.register()
    tools_v4.register()
    tag = os.environ.get("GD_TAG", "7b")

    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    step = max(1, len(tasks) // a.train)
    tr_ids = {t["id"] for t in tasks[::step][:a.train]}
    by_id = {t["id"]: t for t in tasks}

    suf = "" if a.hyp == "gold" else f"_{a.hyp}"
    L = json.load(open(os.path.join(a.dir, f"tgb_loo_{tag}{suf}.json")))
    tau = L["selected"]["tau"]
    key = next(k for k in L["sweep"]
               if k != "S" and abs(float(k) - tau) < 1e-9)
    masks = {k: tuple(sorted(v)) for k, v in L["sweep"][key]["masks"].items()}

    test = [t for t in tasks if t["id"] not in tr_ids]
    train = [t for t in tasks if t["id"] in tr_ids]

    g_mode = collections.Counter(masks[t["id"]] for t in train).most_common(1)[0][0]
    f_mode = {}
    for f in {t["family"] for t in tasks}:
        c = collections.Counter(masks[t["id"]] for t in train
                                if t["family"] == f)
        f_mode[f] = c.most_common(1)[0][0] if c else g_mode

    plans = {
        "global": lambda t: list(g_mode),
        "family": lambda t: list(f_mode[t["family"]]),
        "task": lambda t: list(masks[t["id"]]),
    }
    agree = st.mean([masks[t["id"]] == f_mode[t["family"]] for t in test])
    print(f"[{tag}] hyp={a.hyp} tau={tau:+.2f}  test={len(test)}  "
          f"task-mask equals its family mode on {agree:.1%}")

    tmp = tempfile.mkdtemp(prefix="tgb_gran_")
    for t in tasks:
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
    sp = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                        temperature=0)

    prompts, meta = [], []
    for name, fn in plans.items():
        for t in test:
            prompts.append(TPL.format(q=t["question"], o=ctx_of(t, fn(t))))
            meta.append((name, t["id"], t["gold"], len(fn(t))))
    outs = llm.generate([{"prompt": p} for p in prompts], sp)

    acc = collections.defaultdict(list)
    tools = collections.defaultdict(list)
    per_fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for (name, tid, gold, k), o in zip(meta, outs):
        c = correct(o.outputs[0].text, gold)
        acc[name].append(c)
        tools[name].append(k)
        per_fam[name][by_id[tid]["family"]].append(c)

    print(f"\n    {'granularity':<10}{'acc':>8}{'tools':>8}   delta vs coarser")
    prev = None
    out = dict(tag=tag, hyp=a.hyp, tau=tau, agree_with_family_mode=agree,
               n_test=len(test), rows={})
    for name in ("global", "family", "task"):
        m = st.mean(acc[name])
        d = "" if prev is None else f"{m - prev:+.4f}"
        print(f"    {name:<10}{m:>8.3f}{st.mean(tools[name]):>8.2f}   {d}")
        out["rows"][name] = dict(acc=m, tools=st.mean(tools[name]),
                                 by_family={f: st.mean(v)
                                            for f, v in per_fam[name].items()})
        prev = m
    gain = out["rows"]["task"]["acc"] - out["rows"]["family"]["acc"]
    print(f"\n    what per-task granularity buys over per-family: {gain:+.4f}")
    p = os.path.join(a.dir, f"tgb_granularity_{tag}_{a.hyp}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
