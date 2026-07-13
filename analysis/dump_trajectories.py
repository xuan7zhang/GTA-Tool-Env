#!/usr/bin/env python3
"""Convert OpenCompass prediction JSONs of a run into trajectories.jsonl.gz.

One line per (dataset, sample): {run_id, dataset, task_id, gold?, steps[...]}
where steps include every tool call (name, args), tool return, thought, and
the final answer — the raw material for per-sample attribution.

Usage: python dump_trajectories.py --run-dir .../results/baseline [--out FILE]
"""
import argparse
import glob
import gzip
import json
import os.path as osp


def latest_ts_dir(run_dir):
    cands = [d for d in glob.glob(osp.join(run_dir, "*")) if osp.isdir(d)
             and osp.basename(d)[:8].isdigit()]
    return max(cands) if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()

    ts = latest_ts_dir(args.run_dir)
    assert ts, f"no timestamped opencompass dir under {args.run_dir}"
    out = args.out or osp.join(args.run_dir, "trajectories.jsonl.gz")
    run_id = osp.basename(args.run_dir.rstrip("/"))

    n = 0
    with gzip.open(out, "wt") as w:
        for f in sorted(glob.glob(osp.join(ts, "predictions", "*", "*.json"))):
            dataset = osp.splitext(osp.basename(f))[0]
            data = json.load(open(f))
            for k, v in sorted(data.items(), key=lambda kv: int(kv[0])):
                rec = {"run_id": run_id, "dataset": dataset, "task_id": int(k)}
                rec.update(v)
                w.write(json.dumps(rec, default=str) + "\n")
                n += 1
    print(f"wrote {n} trajectories to {out}")


if __name__ == "__main__":
    main()
