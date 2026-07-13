#!/usr/bin/env python3
"""Aggregate sweep runs into one CSV row per run.

Walks results/<run_id>/ dirs (each containing manifest_row.json, run_env.txt,
an OpenCompass workdir <timestamp>/ with summary + predictions, and
proxy_calls.jsonl[.gz]) and emits:

run_id, variant params, probe_mode, seed, InstAcc, ToolAcc, ArgAcc, SummAcc,
AnsAcc, AnsAcc_w_imggen, P/O/L/C F1, tool_call, tool_call_error,
mean_tool_calls_per_task, mean_pred_chars (token proxy: chars/4)

Usage: python aggregate_results.py --results /datasets/.../gta2/results --out sweep.csv
"""
import argparse
import csv
import glob
import gzip
import json
import os
import os.path as osp

STEP_KEYS = {"inst_align": "InstAcc", "tool_acc": "ToolAcc",
             "arg_acc": "ArgAcc", "answer_acc": "SummAcc"}
END_KEYS = {"answer_acc": "AnsAcc", "answer_acc_w_imggen": "AnsAcc_w_imggen",
            "p_f1": "P_F1", "o_f1": "O_F1", "l_f1": "L_F1", "c_f1": "C_F1",
            "tool_call": "tool_call", "tool_call_error": "tool_call_error"}


def latest_ts_dir(run_dir):
    cands = [d for d in glob.glob(osp.join(run_dir, "*")) if osp.isdir(d)
             and osp.basename(d)[:8].isdigit()]
    return max(cands) if cands else None


def load_results_json(ts_dir):
    """OpenCompass writes results/<model>/<dataset>.json with the metric dict."""
    out = {}
    for f in glob.glob(osp.join(ts_dir, "results", "*", "*.json")):
        ds = osp.splitext(osp.basename(f))[0]
        try:
            out[ds] = json.load(open(f))
        except json.JSONDecodeError:
            pass
    return out


def proxy_stats(run_dir):
    path = None
    for cand in ("proxy_calls.jsonl.gz", "proxy_calls.jsonl"):
        if osp.exists(osp.join(run_dir, cand)):
            path = osp.join(run_dir, cand)
            break
    if not path:
        return {}
    opener = gzip.open if path.endswith(".gz") else open
    n_calls, tasks = 0, set()
    with opener(path, "rt") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event"):
                continue
            n_calls += 1
            if rec.get("task_id") not in (None, ""):
                tasks.add(rec["task_id"])
    return {"proxy_calls": n_calls,
            "mean_tool_calls_per_task": round(n_calls / max(1, len(tasks)), 2)
            if tasks else None}


def pred_char_stats(ts_dir):
    total, n = 0, 0
    for f in glob.glob(osp.join(ts_dir, "predictions", "*", "*.json")):
        try:
            data = json.load(open(f))
        except json.JSONDecodeError:
            continue
        for v in data.values():
            total += len(json.dumps(v.get("prediction", "")))
            n += 1
    return {"mean_pred_chars": round(total / n, 1) if n else None,
            "approx_mean_tokens": round(total / n / 4, 1) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="sweep.csv")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(glob.glob(osp.join(args.results, "*"))):
        if not osp.isdir(run_dir):
            continue
        row = {"run_id": osp.basename(run_dir)}
        mrow = osp.join(run_dir, "manifest_row.json")
        if osp.exists(mrow):
            m = json.load(open(mrow))
            row.update({"probe_mode": m.get("probe_mode"), "seed": m.get("seed"),
                        "toolmeta": m.get("toolmeta"),
                        "mask": ",".join(m["mask"]) if m.get("mask") else "",
                        "extra_tools": m.get("extra_tools", "")})
        ts = latest_ts_dir(run_dir)
        if ts:
            res = load_results_json(ts)
            for ds, metrics in res.items():
                keymap = STEP_KEYS if "step" in ds else END_KEYS
                for k, name in keymap.items():
                    if k in metrics:
                        row[name] = round(metrics[k], 2) if isinstance(
                            metrics[k], float) else metrics[k]
            row.update(pred_char_stats(ts))
        row.update(proxy_stats(run_dir))
        if len(row) > 1:
            rows.append(row)

    cols = ["run_id", "probe_mode", "seed", "mask", "toolmeta", "extra_tools",
            "InstAcc", "ToolAcc", "ArgAcc", "SummAcc", "AnsAcc",
            "AnsAcc_w_imggen", "P_F1", "O_F1", "L_F1", "C_F1", "tool_call",
            "tool_call_error", "proxy_calls", "mean_tool_calls_per_task",
            "mean_pred_chars", "approx_mean_tokens"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
