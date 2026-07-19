#!/usr/bin/env python3
"""Arc-2 (composition null) effect-size upper bound. Given the paired mask-sweep
results, report the largest true effect that could have gone undetected at the
observed n and the chosen alpha -- i.e. an upper bound on the null. CPU only.

Reads existing mask runs (full vs oracle / LOO) from $GTA_BIG/results, computes the
per-seed paired differences, and inverts a paired t-test power calculation to get
the minimum detectable effect (MDE) at power 0.8. Writes metrics.json.
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path

BIG = Path(os.environ.get("GTA_BIG", "/datasets/omni_pretraining/gta2"))


def acc(rid):
    fs = sorted(glob.glob(str(BIG / f"results/{rid}/*/results/*/gta_bench_end.json")))
    return json.load(open(fs[-1]))["answer_acc"] if fs else None


def paired_diffs(a_prefix, b_prefix, seeds=range(1, 4)):
    d = []
    for s in seeds:
        a, b = acc(f"{a_prefix}_s{s}"), acc(f"{b_prefix}_s{s}")
        if a is not None and b is not None:
            d.append(a - b)
    return d


def mde(sd, n, alpha=0.05, power=0.8):
    """Minimum detectable effect for a paired t-test (normal approx)."""
    if n < 2 or sd == 0:
        return None
    # z-approx: MDE = (z_alpha/2 + z_power) * sd / sqrt(n)
    z_a = 1.959963985  # two-sided 0.05
    z_b = 0.841621234  # power 0.8
    return (z_a + z_b) * sd / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-log", default="auto")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # candidate mask contrasts present on disk (full vs per-task oracle across seeds)
    contrasts = {
        "full_vs_oracle_7B": ("pm7b_full", "pm7b_pertask"),
        "full_vs_oracle_3B": ("pm3b_full", "pm3b_pertask"),
    }
    results = {}
    for name, (pa, pb) in contrasts.items():
        d = paired_diffs(pa, pb)
        if len(d) < 2:
            continue
        mean = sum(d) / len(d)
        var = sum((x - mean) ** 2 for x in d) / (len(d) - 1)
        sd = math.sqrt(var)
        results[name] = {
            "n": len(d), "mean_diff": round(mean, 3), "sd": round(sd, 3),
            "mde_at_power0.8": round(mde(sd, len(d), a.alpha) or float("nan"), 3),
        }
    out = {"contrasts": results,
           "note": "MDE = smallest true effect detectable at alpha, power 0.8; "
                   "an observed |mean_diff| below MDE is consistent with the null.",
           "mean_acc": 0.0}  # analysis job: no accuracy, but orchestrator wants a dict
    Path(a.out, "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
