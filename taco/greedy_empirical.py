"""Empirical greedy forward search on calibration accuracy (spec §21 baseline).

This is the *outcome-based* control: unlike the TACO searches, which score a
candidate environment with a fitted utility model, this one pays for a real
calibration evaluation at every step. It is the same family as the OctoTools
greedy baseline already characterised in this repository -- label-hungry,
O(n) evaluations -- and it is included so the comparison is against a method
that spends real compute, not only against paper baselines.

Stages (each is a global forced menu, native output):
  1. all 14 singletons                          14 runs
  2. best singleton + each remaining tool       13 runs
  3. best pair + each remaining tool            12 runs

39 runs at ~7.5 min each on 80 calibration tasks. Runs already present with a
DONE marker are reused, so overlaps with the subset design cost nothing.

Usage: python taco/greedy_empirical.py --lane 1 --model qwen2.5-7b-instruct --tag 7b
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taco.paths import TACO, SEARCH, TOOLMETA, DS_PATH   # noqa: E402
from taco import runner as R                              # noqa: E402


def evaluate(tools, lane, model, tag, splits, ds, all_tools, step):
    key = "-".join(sorted(t[:4] for t in tools))
    rid = f"ge{step}_{tag}_{abs(hash(key)) % 10**8}"
    cond = dict(run_id=rid, stage="greedy_emp", kind="calibration",
                task_set="calibration",
                menu={"mode": "forced", "tools": sorted(tools)},
                taco_spec=None, label="GLOBAL",
                notes=f"empirical greedy stage {step}: {','.join(sorted(tools))}")
    rec = R.run_one(cond, lane, model, splits, ds, all_tools)
    return rec.get("answer_acc", 0.0) if rec.get("status") == "OK" else -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, default=1)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stages", type=int, default=3)
    a = ap.parse_args()

    splits = json.load(open(f"{TACO}/splits.json"))
    ds = json.load(open(DS_PATH))
    all_tools = list(json.load(open(TOOLMETA)))

    cur, traj = [], []
    for step in range(1, a.stages + 1):
        scored = []
        for t in all_tools:
            if t in cur:
                continue
            cand = sorted(cur + [t])
            acc = evaluate(cand, a.lane, a.model, a.tag, splits, ds, all_tools, step)
            scored.append(dict(tools=cand, added=t, calibration_acc=acc))
            print(f"[greedy] step {step} +{t:<28} acc={acc:.2f}")
        if not scored:
            break
        best = max(scored, key=lambda r: r["calibration_acc"])
        cur = best["tools"]
        traj.append(dict(step=step, chosen=best["added"], tools=list(cur),
                         calibration_acc=best["calibration_acc"],
                         candidates=scored))
        print(f"[greedy] step {step} -> {cur} acc={best['calibration_acc']:.2f}")

    os.makedirs(SEARCH, exist_ok=True)
    p = f"{SEARCH}/greedy_empirical_{a.tag}.json"
    json.dump(dict(model=a.model, trajectory=traj, selected=cur,
                   n_runs=sum(len(s["candidates"]) for s in traj)),
              open(p, "w"), indent=1)
    print("[greedy] ->", p)


if __name__ == "__main__":
    main()
