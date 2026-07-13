#!/usr/bin/env python3
"""Compute per-tool corruption-attribution from probe runs and (re)write all
tool selectors including ours_corruption.

attribution(tool) = clean_subset_AnsAcc - corrupt_tool_subset_AnsAcc
(read from each probe run's evaluator results json — no re-scoring needed).
Low attribution = the agent's answers don't causally depend on this tool's
output (a poison tool, whose output was already wrong) -> pruned by 'ours'.
"""
import argparse
import glob
import json
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(__file__))
from tool_selectors import degraded_trajectory_stats, select  # noqa: E402

BIG = "/datasets/omni_pretraining/gta2/results"


def ansacc(run_id):
    f = glob.glob(f"{BIG}/{run_id}/*/results/*/gta_bench_end.json")
    if not f:
        return None
    return json.load(open(sorted(f)[-1])).get("answer_acc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--injected", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--degraded-run", required=True)
    ap.add_argument("--out-selectors", required=True)
    args = ap.parse_args()

    pool = args.pool.split(",")
    injected = args.injected.split(",")
    clean = ansacc("probe_clean_subset")
    attr = {}
    for run in glob.glob(f"{BIG}/probe_*"):
        rid = osp.basename(run)
        if rid in ("probe_clean_subset",):
            continue
        tool = rid[len("probe_"):]
        a = ansacc(rid)
        if a is not None and clean is not None:
            attr[tool] = round(clean - a, 3)  # drop when this tool is corrupted
    json.dump({"clean_subset_acc": clean, "attribution": attr},
              open(f"{args.work}/attribution.json", "w"), indent=1)
    print(f"clean_subset_acc={clean}")
    for t, v in sorted(attr.items(), key=lambda x: x[1]):
        print(f"  attr {t:22s} {v:+.3f}  {'(injected)' if t in injected else ''}")

    calls, errs = degraded_trajectory_stats(args.degraded_run)
    methods = ["keep_all", "oracle", "random", "call_frequency", "error_rate", "ours_corruption"]
    out = {}
    for m in methods:
        keep = select(m, pool, args.k, injected, calls, errs, attr, seed=0)
        out[m] = {"keep": keep, "pruned": [t for t in pool if t not in keep]}
    json.dump(out, open(args.out_selectors, "w"), indent=1)
    print("\nselectors:")
    for m, v in out.items():
        hit = len(set(v["pruned"]) & set(injected))
        print(f"  {m:16s} pruned {len(v['pruned'])} ({hit}/{len(injected)} poison): {v['pruned']}")


if __name__ == "__main__":
    main()
