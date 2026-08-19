"""Build TGB-v3: nine families, OCR in exactly one chain.

    python -m tgb.generate_v3 --n 450 --out $GTA_BIG/results/taco/tgb3

The v1/v2 invariants carry over, with two families exempted for structural
reasons and one new check:

  * `no_tool` has no chain at all -- its correct environment is the empty one --
    so the sufficiency / orphan / corruption checks do not apply to it. That is
    the point of including it: it is the first TGB family whose best action is
    to call nothing.
  * `compute_only` has a chain of length one with no upstream, so the orphan
    check does not apply either. It is tool-*optional*: every number is already
    in the question, so E_none is genuinely non-zero and dA stops being a
    relabelling of Acc.
  * NEW: for the six non-OCR families the rendered image must contain **no
    glyphs at all**, checked by asserting the OCR tool returns nothing. Without
    that the benchmark would quietly re-admit the read-then-compute shortcut it
    exists to remove.
"""
import argparse
import json
import os
import random
import re
import zlib

from . import families_v3 as F3
from . import scenes, tools_v2, tools_v3
from .coalitions_v3 import candidates_v3
from .generate import context
from .generate_v2 import contamination, final_step
from .tools import TOOLS, run_chain

tools_v2.register()
tools_v3.register()

NO_TEXT = {"count_price", "count_compare", "attr_size", "retrieve_calc",
           "table_lookup", "sensor_convert", "compute_only", "no_tool"}
NO_CHAIN = {"no_tool"}
NO_UPSTREAM = {"no_tool", "compute_only"}


def build(n_per_family, out_dir, seed=20260805):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rng = random.Random(seed)
    tasks, reject = [], {}
    for fam, fn in F3.FAMILIES_V3.items():
        made = tries = 0
        while made < n_per_family and tries < n_per_family * 40:
            tries += 1
            t = fn(rng, made)
            t["family"] = fam
            tid = f"{fam}_{made:04d}"
            path = os.path.join(img_dir, tid + ".png")
            boxes = scenes.render(t["scene"], path)
            t["id"] = tid
            t["image"] = os.path.relpath(path, out_dir)

            bad = None
            # the v3 check: a non-OCR family must carry no readable glyph
            if fam in NO_TEXT and boxes:
                bad = "text_leaked_into_a_non_ocr_family"

            conds = {}
            for c in candidates_v3(t):
                erng = random.Random(zlib.crc32(f"{tid}/{c['name']}".encode()))
                outs, trace = run_chain(t, t["scene"], boxes, set(c["tools"]),
                                        erng, corrupt_tools=set(c["corrupt"]))
                conds[c["name"]] = dict(tools=c["tools"], label=c["label"],
                                        corrupt=c["corrupt"],
                                        context=context(
                                            outs, [s["tool"] for s in t["plan"]]),
                                        outputs=outs, trace=trace)
            t["conditions"] = conds
            g = t["gold"]
            for name, c in conds.items():
                if c["context"] and F3.gold_in(c["context"], g):
                    bad = bad or f"echo:{name}"
            if F3.gold_in(t["question"], g):
                bad = bad or "echo:question"

            if fam not in NO_CHAIN:
                down = [x for x in t["gt_tools"] if x not in t["upstream"]][-1]
                if final_step(t, conds["useful"]["outputs"].get(down, "")) != g:
                    bad = bad or "insufficient"
                if fam not in NO_UPSTREAM:
                    nu = conds["no_upstream"]["outputs"].get(down, "")
                    if final_step(t, nu) is not None:
                        bad = bad or "orphan_ok"
                    if conds["corrupt"]["outputs"].get(down) == \
                            conds["useful"]["outputs"].get(down):
                        bad = bad or "corrupt_noop"

            if bad:
                reject[bad] = reject.get(bad, 0) + 1
                if os.path.exists(path):
                    os.remove(path)
                continue
            t["contamination"] = contamination(t, conds["union"]["context"]) \
                if "union" in conds else []
            t["has_ocr"] = int("OCR" in t["gt_tools"])
            tasks.append(t)
            made += 1
    return tasks, reject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--out", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb3")
    ap.add_argument("--seed", type=int, default=20260805)
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
        ex = next(t for t in tasks if t["family"] == f)
        print(f"  {f:<18}{c:>5}   chain: "
              f"{' -> '.join(ex['gt_tools']) or '(none)'}")
    n_ocr = sum(t["has_ocr"] for t in tasks)
    print(f"  tasks whose chain uses OCR: {n_ocr}/{len(tasks)} "
          f"= {n_ocr / len(tasks):.0%}")
    print("  rejected:", reject or "none")


if __name__ == "__main__":
    main()
