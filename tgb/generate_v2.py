"""Build TGB-v2: eight families over a conflicting tool menu.

    python -m tgb.generate_v2 --n 500 --out $GTA_BIG/results/taco/tgb2

Same invariants as v1 (A no context contains the gold, B the question does not,
C the useful coalition is sufficient, D the orphaned downstream errors, E the
corruption changes the number), plus one that only v2 can state:

    F. the union of every family's chain is *not* sufficient on its own --
       for at least some families it must lose accuracy relative to the exact
       chain, because the off-family tools fire blind and emit competing
       numbers.

F is not asserted per task (it is a model-level property, measured later); what
is asserted is its precondition: on every task, the `union` context must
contain at least one numeric value that is neither the intermediate nor an
input the chain used. That is checked as `contamination`, and a task with no
contamination is discarded -- otherwise v2 would silently degrade back to v1.
"""
import argparse
import json
import os
import random
import re
import zlib

from . import families_v2 as F2
from . import scenes, tools_v2
from .coalitions_v2 import candidates_v2
from .generate import CTX_CHARS, context
from .tools import run_chain

tools_v2.register()

_LASTNUM = re.compile(r"(-?\d+(?:\.\d+)?)(?!.*\d)", re.S)


def final_step(task, out_str):
    """Replay the model-side easy step on the final tool's output: read its
    trailing number and combine it with k. Returns None if the tool errored.

    The result is rendered in the *same notation as the task's own gold*. A
    family whose answer is a count -- minutes, units, a capacity -- writes gold
    as "6", and hard-coding two decimals here made every such task fail the
    sufficiency check and be discarded silently. `schedule_gap` survived only
    on the samples that happened to slip through.
    """
    if not out_str or "Error" in out_str or "NameError" in out_str:
        return None
    m = _LASTNUM.search(out_str.strip())
    if not m:
        return None
    v, k = float(m.group(1)), float(task["meta"]["k"])
    op = task["meta"]["final_op"]
    r = round(k - v, 2) if op in ("k_minus_v", "paid_minus_total") \
        else round(v - k, 2)
    gold = str(task.get("gold", ""))
    if "." not in gold and r == int(r):
        return str(int(r))
    return f"{r:.2f}"


def contamination(task, union_ctx):
    """Numbers the union context adds that the chain never produced or read."""
    known = set()
    for s in task["conditions"]["useful"]["trace"]:
        known |= set(re.findall(r"\d+(?:\.\d+)?", str(s["output"])))
    known |= set(re.findall(r"\d+(?:\.\d+)?", task["question"]))
    extra = set(re.findall(r"(?<![\w.])(\d+\.\d{2})(?![\d])", union_ctx)) - known
    return sorted(extra)


def build(n_per_family, out_dir, seed=20260804):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rng = random.Random(seed)
    tasks, reject = [], {}
    for fam, fn in F2.FAMILIES_V2.items():
        made = tries = 0
        while made < n_per_family and tries < n_per_family * 40:
            tries += 1
            t = fn(rng, made)
            tid = f"{fam}_{made:04d}"
            path = os.path.join(img_dir, tid + ".png")
            boxes = scenes.render(t["scene"], path)
            t["id"] = tid
            t["image"] = os.path.relpath(path, out_dir)

            conds, bad = {}, None
            for c in candidates_v2(t):
                erng = random.Random(zlib.crc32(f"{tid}/{c['name']}".encode()))
                outs, trace = run_chain(t, t["scene"], boxes, set(c["tools"]),
                                        erng, corrupt_tools=set(c["corrupt"]))
                conds[c["name"]] = dict(tools=c["tools"], label=c["label"],
                                        corrupt=c["corrupt"],
                                        context=context(outs, [s["tool"]
                                                               for s in t["plan"]]),
                                        outputs=outs, trace=trace)
            t["conditions"] = conds
            g = t["gold"]
            for name, c in conds.items():
                if c["context"] and F2.gold_in(c["context"], g):
                    bad = f"echo:{name}"
            if F2.gold_in(t["question"], g):
                bad = bad or "echo:question"
            down = [x for x in t["gt_tools"] if x not in t["upstream"]][-1]
            if final_step(t, conds["useful"]["outputs"].get(down, "")) != g:
                bad = bad or "insufficient"
            if final_step(t, conds["no_upstream"]["outputs"].get(down, "")) is not None:
                bad = bad or "orphan_ok"
            if conds["corrupt"]["outputs"].get(down) == \
                    conds["useful"]["outputs"].get(down):
                bad = bad or "corrupt_noop"
            extra = contamination(t, conds["union"]["context"])
            if not extra:
                bad = bad or "no_contamination"

            if bad:
                reject[bad] = reject.get(bad, 0) + 1
                os.remove(path)
                continue
            t["contamination"] = extra
            tasks.append(t)
            made += 1
    return tasks, reject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb2")
    ap.add_argument("--seed", type=int, default=20260804)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks, reject = build(a.n, a.out, a.seed)
    p = os.path.join(a.out, "tgb_tasks.json")
    json.dump(tasks, open(p, "w"), indent=1)
    print(f"wrote {p}: {len(tasks)} tasks")
    by = {}
    for t in tasks:
        by[t["family"]] = by.get(t["family"], 0) + 1
    for f, c in sorted(by.items()):
        print(f"  {f:<22} {c}")
    print("  rejected:", reject or "none")
    ncont = [len(t["contamination"]) for t in tasks]
    print(f"  contaminating values in the union context: "
          f"mean {sum(ncont)/len(ncont):.2f}, min {min(ncont)}")


if __name__ == "__main__":
    main()
