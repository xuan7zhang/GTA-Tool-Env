"""Build the TGB tool-grounded benchmark: scenes -> PNGs -> executed tool
coalitions -> validated task records.

    python -m tgb.generate --n 800 --out $GTA_BIG/results/taco/tgb

Every record ships the raw input (an image path), the candidate coalitions with
their **executed** outputs, the execution trace, and the gold. Nothing here
looks at a model; per-model derivability and dL come later (tgb/score_dl.py).

Validation is not advisory -- a task that fails any invariant is discarded and
counted, so the printed reject tally is the audit of how tight the conditions
are. The invariants:

  A. no coalition's context contains the gold (answer-echo)
  B. the question alone does not contain the gold
  C. the `useful` coalition is *sufficient*: replaying the final easy step on
     the Calculator's real output reproduces the gold exactly
  D. the `no_upstream` coalition really fails (the downstream tool errors)
  E. the `corrupt` coalition really produces a different number than `useful`
"""
import argparse
import json
import os
import random
import re
import zlib

from . import families, scenes
from .coalitions import candidates
from .tools import run_chain

CTX_CHARS = int(os.environ.get("TGB_CTX_CHARS", "1500"))
TPL = ("Answer the question with a short final answer only.\n\n"
       "Question: {q}\n\nTool output:\n{o}\n\nFinal answer:")


# Perception, then retrieval, then computation: the order a ReAct agent
# produces, and the order the real-GTA contexts are in.
CANON = ["OCR", "ImageDescription", "CountGivenObject", "TextToBbox",
         "GoogleSearch", "Calculator", "Solver"]


def context(outputs, order):
    """Same surface form as the real-GTA scorer: '<Tool>: <output>' lines, so
    dL values are directly comparable with the §7 real-GTA numbers."""
    rank = {t: i for i, t in enumerate(CANON)}
    seen = sorted(outputs, key=lambda t: (rank.get(t, 99), t))
    return "\n".join(f"{t}: {str(outputs[t])[:CTX_CHARS]}" for t in seen
                     if outputs.get(t))


def final_step(task, calc_out):
    """Replay the model-side easy step on a Calculator output. Returns None if
    the Calculator errored (which is what `no_upstream` is supposed to do)."""
    m = re.match(r"^-?\d+(?:\.\d+)?$", calc_out.strip())
    if not m:
        return None
    v = float(calc_out)
    k = float(task["meta"]["k"])
    return f"{round(k - v, 2):.2f}" if task["meta"]["final_op"] == \
        "paid_minus_total" else f"{round(v - k, 2):.2f}"


def build(n_per_family, out_dir, seed=20260803):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    rng = random.Random(seed)
    tasks, reject = [], {}
    for fam, fn in families.FAMILIES.items():
        made = tries = 0
        while made < n_per_family and tries < n_per_family * 40:
            tries += 1
            t = fn(rng, made)
            tid = f"{fam}_{made:04d}"
            path = os.path.join(img_dir, tid + ".png")
            boxes = scenes.render(t["scene"], path)
            t["id"] = tid
            t["image"] = os.path.relpath(path, out_dir)

            # ---- execute every candidate coalition on this scene
            conds, bad = {}, None
            for c in candidates(t):
                # crc32, not hash(): PYTHONHASHSEED must not touch the data
                erng = random.Random(zlib.crc32(f"{tid}/{c['name']}".encode()))
                outs, trace = run_chain(t, t["scene"], boxes, set(c["tools"]),
                                        erng, corrupt_tools=set(c["corrupt"]))
                ctx = context(outs, [s["tool"] for s in t["plan"]])
                conds[c["name"]] = dict(tools=c["tools"], label=c["label"],
                                        corrupt=c["corrupt"], context=ctx,
                                        outputs=outs, trace=trace)

            g = t["gold"]
            # A. no context may contain the gold
            for name, c in conds.items():
                if c["context"] and families.gold_in(c["context"], g):
                    bad = f"echo:{name}"
            # B. the question must not give it away
            if families.gold_in(t["question"], g):
                bad = bad or "echo:question"
            # C. useful must be sufficient
            calc = conds["useful"]["outputs"].get("Calculator", "")
            if final_step(t, calc) != g:
                bad = bad or "insufficient"
            # D. orphaned downstream must actually fail
            nu = conds["no_upstream"]["outputs"].get("Calculator", "")
            if final_step(t, nu) is not None:
                bad = bad or "orphan_ok"
            # E. corruption must actually change the computed number
            if conds["corrupt"]["outputs"].get("Calculator", "") == calc:
                bad = bad or "corrupt_noop"

            if bad:
                reject[bad] = reject.get(bad, 0) + 1
                os.remove(path)
                continue
            t["conditions"] = conds
            tasks.append(t)
            made += 1
    return tasks, reject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="tasks per family")
    ap.add_argument("--out", default=os.environ.get(
        "TGB_OUT", "/datasets/omni_pretraining/gta2/results/taco/tgb"))
    ap.add_argument("--seed", type=int, default=20260803)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks, reject = build(a.n, a.out, a.seed)
    path = os.path.join(a.out, "tgb_tasks.json")
    json.dump(tasks, open(path, "w"), indent=1)
    print(f"wrote {path}: {len(tasks)} tasks")
    by = {}
    for t in tasks:
        by[t["family"]] = by.get(t["family"], 0) + 1
    for f, c in by.items():
        print(f"  {f:<20} {c}")
    print("  rejected:", reject or "none")


if __name__ == "__main__":
    main()
