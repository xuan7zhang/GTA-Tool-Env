"""Algorithm 1, Task-specific Toolset Optimization, run faithfully on TGB-v4.

    Acc_base = Acc(D_base)
    for d_i not in D_base:  D_i = D_base + {d_i};  delta_i = Acc(D_i) - Acc_base
    D* = D_base + { d_i : delta_i > 0 }

Single pass, each tool measured independently against a fixed base, every
positive one added at once. That is structurally the same shape as this
project's leave-one-out selector -- one parallel marginal per tool, threshold,
union -- and differs in exactly one place: the objective is accuracy where the
selector uses gold likelihood. Running it is therefore the clean head-to-head
the backward-elimination baseline could not give, because that one changed the
search direction and the number of rounds at the same time.

`Acc` is left ambiguous by the pseudocode, and the ambiguity matters, so both
readings are implemented:

  --level split : Acc over the 400-task labelled split. Well-posed, but the
                  output is one global toolset, so "task-specific" describes
                  the tuning loop rather than the deployed mask.
  --level task  : Acc on the single task being answered. Genuinely per-task,
                  and this is where it should break: accuracy on one task is
                  one bit, so delta_i lands in {-1, 0, +1} and is 0 for almost
                  every tool. Measured here rather than asserted.

    GD_TAG=7b GD_MODEL=... python -m tgb.algo1 --level split --base union
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
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--level", choices=["split", "task"], default="split")
    ap.add_argument("--base", choices=["empty", "union"], default="empty",
                    help="D_base. `empty` is the literal cold start; `union` "
                         "is the 7 real tools, the most generous base a "
                         "practitioner would pick.")
    a = ap.parse_args()
    tools_v2.register()
    tools_v4.register()
    tag = os.environ.get("GD_TAG", "7b")

    from .coalitions_v4 import FULL_MENU_V4
    from .families_v4 import UNION_TOOLS_V4
    MENU = list(FULL_MENU_V4)
    base = [] if a.base == "empty" else list(UNION_TOOLS_V4)

    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    step = max(1, len(tasks) // a.train)
    tr_ids = {t["id"] for t in tasks[::step][:a.train]}
    train = [t for t in tasks if t["id"] in tr_ids]
    test = [t for t in tasks if t["id"] not in tr_ids]
    tmp = tempfile.mkdtemp(prefix="tgb_algo1_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))
    print(f"[{tag}] Algorithm 1, level={a.level}, base={a.base} "
          f"({len(base)} tools), train {len(train)} / test {len(test)}")

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

    def hits(masks, subset):
        """Per-task correctness of each mask, one batch."""
        prompts, meta = [], []
        for mi, m in enumerate(masks):
            for t in subset:
                prompts.append(TPL.format(q=t["question"], o=ctx_of(t, m)))
                meta.append((mi, t["id"], t["gold"]))
        outs = llm.generate([{"prompt": p} for p in prompts], sp)
        h = collections.defaultdict(dict)
        for (mi, tid, g), o in zip(meta, outs):
            h[mi][tid] = correct(o.outputs[0].text, g)
        return h

    cand = [d for d in MENU if d not in base]
    out = dict(tag=tag, level=a.level, base=a.base, base_tools=base,
               n_train=len(train), n_test=len(test), test_ids=[t["id"] for t in test])

    if a.level == "split":
        H = hits([base] + [base + [d] for d in cand], train)
        acc0 = st.mean(H[0].values())
        keep, deltas = [], {}
        for i, d in enumerate(cand):
            deltas[d] = st.mean(H[i + 1].values()) - acc0
            if deltas[d] > 0:
                keep.append(d)
        star = base + keep
        print(f"    Acc(D_base)={acc0:.3f}   kept {len(keep)}: {keep}")
        for d, v in sorted(deltas.items(), key=lambda x: -x[1]):
            print(f"      {d:<18}{v:+.4f}")
        T = hits([star, base, MENU], test)
        out.update(deltas=deltas, star=star,
                   test_acc=dict(algo1=st.mean(T[0].values()),
                                 base=st.mean(T[1].values()),
                                 full=st.mean(T[2].values())),
                   tools=len(star))
        print(f"\n    TEST  algo1 {out['test_acc']['algo1']:.3f} "
              f"({len(star)} tools)   base {out['test_acc']['base']:.3f}   "
              f"full {out['test_acc']['full']:.3f}")
    else:
        # delta_i on one task is a difference of two bits. Everything below
        # just measures how often that is anything other than zero.
        H = hits([base] + [base + [d] for d in cand], test)
        star_of, ndelta, sizes = {}, collections.Counter(), []
        for t in test:
            tid, a0 = t["id"], H[0][t["id"]]
            keep = [d for i, d in enumerate(cand) if H[i + 1][tid] - a0 > 0]
            ndelta[len([d for i, d in enumerate(cand)
                        if H[i + 1][tid] - a0 != 0])] += 1
            star_of[tid] = base + keep
            sizes.append(len(base) + len(keep))
        by = collections.defaultdict(list)
        for tid, m in star_of.items():
            by[tuple(sorted(m))].append(tid)
        acc = st.mean(
            [v for mi, (m, ids) in enumerate(by.items())
             for v in hits([list(m)], [t for t in test if t["id"] in set(ids)])[0].values()])
        flat = ndelta[0] / len(test)
        out.update(test_acc=dict(algo1=acc, base=st.mean(H[0].values())),
                   tools=st.mean(sizes), distinct_masks=len(by),
                   frac_no_tool_moves_the_bit=flat,
                   star_of={tid: sorted(m) for tid, m in star_of.items()})
        print(f"\n    TEST  algo1 {acc:.3f}  ({st.mean(sizes):.2f} tools, "
              f"{len(by)} distinct masks)   base {st.mean(H[0].values()):.3f}")
        print(f"    on {flat:.1%} of tasks NO tool changes the bit -> "
              f"D* collapses to D_base")

    p = os.path.join(a.dir, f"tgb_algo1_{tag}_{a.level}_{a.base}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
