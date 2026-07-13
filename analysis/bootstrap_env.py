#!/usr/bin/env python3
"""Bootstrap-CI comparison of tool environments from a single (deterministic)
run each. Resamples the 229 tasks (10^4 resamples) to get a 95% CI on AnsAcc —
the same task-bootstrap the lo_geom paper uses — so environments can be compared
without seed replication.

Reuses GTABenchEvaluator scoring: objective refs via whitelist/blacklist regex,
subjective refs via sentence-transformer simscore binarized at --sim-threshold.

Usage:
  python bootstrap_env.py --dataset .../dataset.json --sim-threshold 0.5 \
    --runs baseline=.../results/baseline optcon=.../results/optenv_conservative_pass_s0 ...
Run in the opencompass env (needs sentence_transformers).
"""
import argparse
import glob
import json
import os.path as osp
import re

import numpy as np


def latest_ts(run_dir):
    c = [d for d in glob.glob(osp.join(run_dir, "*")) if osp.isdir(d) and osp.basename(d)[:8].isdigit()]
    return max(c) if c else None


def load_end_preds(run_dir):
    ts = latest_ts(run_dir)
    preds = {}
    for f in glob.glob(osp.join(ts, "predictions", "*", "gta_bench_end*.json")):
        for k, v in json.load(open(f)).items():
            preds[int(k)] = v
    return preds


def final_answer(rec):
    pred = rec.get("prediction")
    if not pred:
        return None
    rounds = pred if isinstance(pred[0], list) else [pred]
    last = rounds[0][-1] if rounds[0] else None
    if not isinstance(last, dict) or "tool_calls" in last or last.get("role") != "assistant":
        return None
    return last.get("content")


class Scorer:
    def __init__(self, thr):
        self.thr = thr
        self._st = None

    def _sim(self, pred, refs):
        if self._st is None:
            from sentence_transformers import SentenceTransformer, util
            self._st = SentenceTransformer("all-mpnet-base-v2")
            self._util = util
        pe = self._st.encode(pred, convert_to_tensor=True)
        best = 0.0
        for s in refs:
            ge = self._st.encode(s, convert_to_tensor=True)
            best = max(best, float(np.maximum(self._util.cos_sim(pe, ge).cpu().numpy(), 0)[0][0]))
        return best

    def correct(self, pred, ref):
        if pred is None or not ref:
            return None if not ref else 0
        if isinstance(ref, dict):
            cnt = 0
            for al in ref["whitelist"]:
                if re.search(r"\b(?:" + "|".join(re.escape(a) for a in al) + r")\b", pred, re.IGNORECASE):
                    cnt += 1
            if ref.get("blacklist"):
                bk = r"\b(?:" + "|".join(re.escape(a) for g in ref["blacklist"] for a in g) + r")\b"
                return int(cnt == len(ref["whitelist"]) and not re.search(bk, pred, re.IGNORECASE))
            return int(cnt == len(ref["whitelist"]))
        return int(self._sim(pred, ref) >= self.thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sim-threshold", type=float, default=0.5)
    ap.add_argument("--runs", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--out", default="analysis/bootstrap_env.md")
    args = ap.parse_args()

    data = json.load(open(args.dataset))
    refs = {int(k): v["gt_answer"] for k, v in data.items()}
    scorer = Scorer(args.sim_threshold)
    idx_sorted = sorted(refs)

    rng = np.random.RandomState(0)
    results = {}
    per_sample = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        preds = load_end_preds(path)
        vec = []  # per-task correctness over answer-type tasks
        for i in idx_sorted:
            if not refs[i]:
                continue
            c = scorer.correct(final_answer(preds.get(i, {})), refs[i])
            vec.append(c if c is not None else 0)
        vec = np.array(vec, float)
        n = len(vec)
        boot = np.array([vec[rng.randint(0, n, n)].mean() for _ in range(args.nboot)]) * 100
        results[name] = dict(acc=vec.mean() * 100, lo=np.percentile(boot, 2.5),
                             hi=np.percentile(boot, 97.5), n=n, boot=boot)
        per_sample[name] = vec

    names = [s.split("=", 1)[0] for s in args.runs]
    base = names[0]
    lines = [f"# Environment bootstrap comparison (task-resample, n={results[base]['n']}, "
             f"{args.nboot} resamples, sim-thr {args.sim_threshold})", "",
             "| environment | AnsAcc | 95% CI | Δ vs " + base + " | P(env>base) |",
             "|---|---|---|---|---|"]
    for nm in names:
        r = results[nm]
        if nm == base:
            lines.append(f"| {nm} | {r['acc']:.2f} | [{r['lo']:.2f},{r['hi']:.2f}] | — | — |")
        else:
            # paired bootstrap: resample the SAME task indices for both envs so
            # the per-task correctness difference is what gets resampled.
            v_e, v_b = per_sample[nm], per_sample[base]
            m = min(len(v_e), len(v_b))
            rng2 = np.random.RandomState(1)
            dboot = np.array([(v_e[s] - v_b[s]).mean() for s in
                              (rng2.randint(0, m, m) for _ in range(args.nboot))]) * 100
            pgt = float((dboot > 0).mean())
            lines.append(f"| {nm} | {r['acc']:.2f} | [{r['lo']:.2f},{r['hi']:.2f}] | "
                         f"{r['acc']-results[base]['acc']:+.2f} | {pgt:.3f} |")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
