#!/usr/bin/env python3
"""Tool-selection methods for the environment-optimization experiment.

Given a DEGRADED environment (base pool + injected poison tools), each selector
produces a pruned tool set (which tools to keep). We compare our probe-based
method against baselines and the oracle.

Signals are computed from behavioral data only (no oracle knowledge of what was
injected) EXCEPT the oracle selector:

  keep_all        no pruning (lower bound)
  oracle          prune exactly the injected poison names (upper bound)
  random          prune k random tools (seeded)
  call_frequency  prune the k most-called tools in the degraded run
                  (naive "these are used a lot, maybe distracting")
  error_rate      prune the k tools with the highest per-tool error rate
  ours_corruption CORRUPTION-ATTRIBUTION: prune the k tools whose per-tool
                  output-corruption causes the LEAST accuracy drop, i.e. tools
                  the agent's correct answers do not causally depend on. A
                  poison tool contributes wrong content the agent submits; its
                  corruption barely changes (already-wrong) outcomes -> low
                  attribution -> pruned. A real contributor's corruption hurts
                  -> high attribution -> kept.

The corruption-attribution signal is produced by `attribution_from_probes()`
given per-tool corrupt-probe AnsAcc values (computed by driving the eval with
proxy per_tool_modes={tool: corrupt_output}); this module only does the ranking.

Usage (ranking only, signals passed in as JSON):
  python tool_selectors.py --degraded-run <dir> --injected name1,name2 \
     --k 4 --corrupt-probes probes.json --out selectors.json
"""
import argparse
import glob
import json
import os.path as osp
import random
from collections import Counter


def degraded_trajectory_stats(run_dir):
    """Per-tool call count and error count from a degraded passthrough run."""
    f = sorted(glob.glob(osp.join(run_dir, "*", "predictions", "*", "gta_bench_end.json")))
    calls, errs = Counter(), Counter()
    if not f:
        return calls, errs
    data = json.load(open(f[-1]))
    for k, v in data.items():
        if not k.isdigit():
            continue
        pred = v.get("prediction") or []
        rounds = pred if (pred and isinstance(pred[0], list)) else [pred]
        for step in (rounds[0] or []):
            if isinstance(step, dict) and "tool_calls" in step:
                nm = step["tool_calls"][0]["function"]["name"]
                calls[nm] += 1
                if "error" in step:
                    errs[nm] += 1
    return calls, errs


def select(method, pool, k, injected, calls, errs, corrupt_attr, seed=0):
    """Return the list of tools to KEEP (prune k)."""
    prunable = [t for t in pool]  # allow pruning anything (method must find poison)
    if method == "keep_all":
        return list(pool)
    if method == "oracle":
        drop = set(injected[:k]) if k else set(injected)
        return [t for t in pool if t not in drop]
    if method == "random":
        rng = random.Random(f"{seed}")
        drop = set(rng.sample(prunable, min(k, len(prunable))))
        return [t for t in pool if t not in drop]
    if method == "call_frequency":
        ranked = sorted(prunable, key=lambda t: -calls.get(t, 0))
        drop = set(ranked[:k])
        return [t for t in pool if t not in drop]
    if method == "error_rate":
        def er(t):
            c = calls.get(t, 0)
            return (errs.get(t, 0) / c) if c else -1
        ranked = sorted(prunable, key=lambda t: -er(t))
        drop = set(ranked[:k])
        return [t for t in pool if t not in drop]
    if method == "ours_corruption":
        # lowest corruption-attribution = least causal contribution -> prune.
        # tools never probed (never called) get attribution 0 -> prunable.
        ranked = sorted(prunable, key=lambda t: corrupt_attr.get(t, 0.0))
        drop = set(ranked[:k])
        return [t for t in pool if t not in drop]
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded-run", required=True)
    ap.add_argument("--pool", required=True, help="comma list of all tools in degraded env")
    ap.add_argument("--injected", required=True, help="comma list of injected poison names")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--corrupt-probes", help="JSON {tool: attribution} (clean_acc - corrupt_tool_acc)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="selectors.json")
    args = ap.parse_args()

    pool = args.pool.split(",")
    injected = args.injected.split(",") if args.injected else []
    calls, errs = degraded_trajectory_stats(args.degraded_run)
    corrupt_attr = json.load(open(args.corrupt_probes)) if args.corrupt_probes else {}

    methods = ["keep_all", "oracle", "random", "call_frequency", "error_rate", "ours_corruption"]
    out = {}
    for m in methods:
        keep = select(m, pool, args.k, injected, calls, errs, corrupt_attr, args.seed)
        out[m] = {"keep": keep, "pruned": [t for t in pool if t not in keep]}
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    for m, v in out.items():
        print(f"  {m:16s} prune: {v['pruned']}")


if __name__ == "__main__":
    main()
