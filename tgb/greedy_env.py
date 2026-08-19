"""Greedy environment search vs per-task likelihood selection.

The `all tools` row in the closed-loop table is the *unoptimised* default. The
honest competitor is the companion analysis's method: greedily search the tool
menu for the single global mask that maximises accuracy, using labels on a
training split, then deploy that one mask everywhere. On real GTA that search
returned OCR-only and doubled AnsAcc, so it is a strong baseline.

    GD_TAG=7b GD_MODEL=... python -m tgb.greedy_env --train 400

Protocol
--------
* Split: every 6th task is train (labels visible to the search), the rest test.
* Greedy **backward elimination** over the 12-tool menu, the same move the
  project's own `scripts_extra/greedy_search.py` makes: start from the full
  menu and drop the tool whose removal most improves train accuracy, until
  nothing does. `--direction forward` runs the other variant for contrast.
  Every candidate mask is *executed* -- the chain runs with exactly that tool
  set -- so the search sees the same contexts a deployment would.
* Report train and test accuracy of the winning mask, and for reference the
  test accuracy of each single-tool and hand-named mask.

Cost, stated correctly
----------------------
It is misleading to say "greedy pays labels on 400 tasks, likelihood pays a
label per task". The two costs live in different places:

* greedy runs a *sequential dataset-level search*: 8 rounds x ~10 candidate
  masks x 400 tasks is ~24k task-mask evaluations (a full path would be
  78 x 400 = 31.2k), each requiring the tools to be executed and the model to
  answer. That is paid once, offline; deployment afterwards costs nothing.
* per-task likelihood runs *one parallel scoring stage per task*: |candidates|
  teacher-forced passes, batched, no search loop, and it emits a mask tailored
  to that task. That is paid at selection time, and it needs the gold.

So the honest framing is not "cheaper vs more expensive" but "offline
sequential search producing one global mask" vs "online parallel scoring
producing a task-specific mask".
"""
import argparse
import json
import os
import random
import zlib

from vllm import LLM, SamplingParams

from .coalitions import FULL_MENU
from .coalitions_v2 import FULL_MENU_V2
from .generate import CTX_CHARS, TPL, context
from .score_dl import correct
from .tools import run_chain


def build_ctx(task, mask):
    rng = random.Random(zlib.crc32(f"{task['id']}/{'+'.join(sorted(mask))}".encode()))
    outs, _ = run_chain(task, task["scene"], task.get("boxes") or [], set(mask),
                        rng, corrupt_tools=())
    return context(outs, [s["tool"] for s in task["plan"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="/datasets/omni_pretraining/gta2/"
                                       "results/taco/tgb/tgb_tasks.json")
    ap.add_argument("--out", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb")
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--direction", choices=["backward", "forward"],
                    default="backward")
    ap.add_argument("--v2", action="store_true",
                    help="use the TGB-v2 menu and executors")
    ap.add_argument("--v4", action="store_true",
                    help="use the TGB-v4 text-only menu and executors")
    a = ap.parse_args()
    if a.v4 or os.environ.get("TGB_V4"):
        from . import tools_v2, tools_v4
        from .coalitions_v4 import FULL_MENU_V4
        tools_v2.register()
        tools_v4.register()
        globals()["FULL_MENU"] = FULL_MENU_V4
    elif a.v2 or os.environ.get("TGB_V2"):
        from . import tools_v2
        tools_v2.register()
        globals()["FULL_MENU"] = FULL_MENU_V2
    tag = os.environ.get("GD_TAG", "7b")
    tasks = json.load(open(a.tasks))

    # The rendered glyph boxes are not stored in the task file (they are
    # recoverable from the scene), so re-render into a scratch path to get them.
    from . import scenes
    import tempfile
    tmp = tempfile.mkdtemp(prefix="tgb_greedy_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    step = max(1, len(tasks) // a.train)
    train = tasks[::step][:a.train]
    tr_ids = {t["id"] for t in train}
    test = [t for t in tasks if t["id"] not in tr_ids]
    print(f"[{tag}] train {len(train)}  test {len(test)}")

    llm = LLM(model=os.environ.get("GD_MODEL", "/datasets/omni_pretraining/"
                                               "gta2/models/Qwen2.5-7B-Instruct"),
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    sp = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                        temperature=0)

    def acc_of(masks, subset):
        """Accuracy of each mask over `subset`, all masks in one batch."""
        prompts, meta = [], []
        for mi, m in enumerate(masks):
            for t in subset:
                prompts.append(TPL.format(q=t["question"], o=build_ctx(t, m)))
                meta.append((mi, t["gold"]))
        outs = llm.generate([{"prompt": p} for p in prompts], sp)
        hit = [0] * len(masks)
        for (mi, g), o in zip(meta, outs):
            hit[mi] += correct(o.outputs[0].text, g)
        return [h / len(subset) for h in hit]

    # ---------------------------------------------------------------- greedy
    # Backward elimination, matching the project's own baseline
    # (`scripts_extra/greedy_search.py`): start from the full menu and drop the
    # tool whose removal most improves validation accuracy, until nothing does.
    # Direction matters here more than usual. Forward selection cold-starts on
    # a chained task: no *single* tool has any marginal value (you need OCR and
    # Calculator together before accuracy leaves zero), so forward greedy can
    # stall at the empty mask. Backward never faces that, because it begins
    # with every chain intact.
    # checkpoint after every round: backward elimination is a long sequential
    # search and three runs on this cluster were killed mid-flight, losing all
    # of it. The state that matters is just (current mask, its train accuracy).
    ck = os.path.join(a.out, f"tgb_greedy_{tag}_{a.direction}.ckpt.json")
    trace = []
    if a.direction == "backward":
        cur = list(globals()["FULL_MENU"])
        cur_acc = None
        if os.path.exists(ck):
            st = json.load(open(ck))
            cur, cur_acc, trace = st["mask"], st["acc"], st["trace"]
            print(f"[{tag}] resuming greedy from a {len(cur)}-tool mask "
                  f"(train {cur_acc:.3f}, {len(trace)} removals done)")
        if cur_acc is None:
            cur_acc = acc_of([cur], train)[0]
        print(f"[{tag}] start mask=FULL ({len(cur)} tools) "
              f"train_acc={cur_acc:.3f}")
        while len(cur) > 1:
            accs = acc_of([[x for x in cur if x != t] for t in cur], train)
            best_i = max(range(len(cur)), key=lambda i: accs[i])
            if accs[best_i] <= cur_acc + 1e-9:
                print(f"[{tag}] no removal improves train acc ({cur_acc:.3f}); "
                      f"stop")
                break
            dropped = cur[best_i]
            cur = [x for x in cur if x != dropped]
            cur_acc = accs[best_i]
            trace.append(dict(removed=dropped, mask=list(cur),
                              train_acc=cur_acc))
            json.dump(dict(mask=cur, acc=cur_acc, trace=trace),
                      open(ck, "w"), indent=1)
            print(f"[{tag}] - {dropped:<20} train_acc={cur_acc:.3f}  "
                  f"mask={cur}", flush=True)
    else:
        cur, cur_acc = [], acc_of([[]], train)[0]
        print(f"[{tag}] start mask=() train_acc={cur_acc:.3f}")
        while True:
            cand = [t for t in globals()["FULL_MENU"] if t not in cur]
            if not cand:
                break
            accs = acc_of([cur + [t] for t in cand], train)
            best_i = max(range(len(cand)), key=lambda i: accs[i])
            if accs[best_i] <= cur_acc + 1e-9:
                print(f"[{tag}] no tool improves train acc ({cur_acc:.3f}); "
                      f"stop")
                break
            cur = cur + [cand[best_i]]
            cur_acc = accs[best_i]
            trace.append(dict(added=cand[best_i], mask=list(cur),
                              train_acc=cur_acc))
            print(f"[{tag}] + {cand[best_i]:<20} train_acc={cur_acc:.3f}  "
                  f"mask={cur}")

    # --------------------------------------------------- reference envelope
    MENU = globals()["FULL_MENU"]
    if a.v4 or os.environ.get("TGB_V4"):
        from .families_v4 import UNION_TOOLS_V4 as UNION_TOOLS
        refs = {"greedy": cur, "none": [], "full": MENU,
                "union": UNION_TOOLS,
                "union_minus_calc": [t for t in UNION_TOOLS
                                     if t != "Calculator"]}
    elif a.v2 or os.environ.get("TGB_V2"):
        from .families_v2 import UNION_TOOLS
        # `union` is the best a global policy can do without abstaining: every
        # family's chain stays runnable. In v2 it is also contaminated, which
        # is the whole point of the comparison.
        refs = {"greedy": cur, "none": [], "full": MENU,
                "union": UNION_TOOLS,
                "union_minus_convert": [t for t in UNION_TOOLS
                                        if t not in ("UnitConvert",
                                                     "TempConvert",
                                                     "CurrencyConvert")],
                "OCR+Calc": ["OCR", "Calculator"]}
    else:
        refs = {"greedy": cur, "none": [], "full": MENU,
                "OCR_only": ["OCR"], "OCR+Calc": ["OCR", "Calculator"],
                "perception+Calc": ["OCR", "CountGivenObject", "GoogleSearch",
                                    "Calculator"]}
    names = list(refs)
    test_accs = acc_of([refs[n] for n in names], test)
    res = dict(tag=tag, direction=a.direction, greedy_mask=cur, greedy_train_acc=cur_acc,
               trace=trace, n_train=len(train), n_test=len(test),
               test_acc={n: v for n, v in zip(names, test_accs)},
               test_ids=[t["id"] for t in test])
    p = os.path.join(a.out, f"tgb_greedy_{tag}_{a.direction}.json")
    json.dump(res, open(p, "w"), indent=1)
    print(f"\n[{tag}] test accuracy of fixed global masks")
    for n, v in zip(names, test_accs):
        print(f"    {n:<18} {v:.3f}   {refs[n]}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
