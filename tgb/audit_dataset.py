"""Audit a built TGB-v4 file at the size it actually ships.

    python -m tgb.audit_dataset --tasks $T4/tgb_tasks.json

The unit tests build 40 tasks per family, and some properties simply do not
hold at 40 that fail at 400: `schedule_gap` has enough distinct answers in a
40-task sample to clear a 0.6*n bound, and 187 in the shipped 400 -- the bound
scales with n, the parameter space does not. A test that passes on a sample and
a dataset that ships at ten times the sample are two different claims, so this
runs against the file a collaborator will actually load.

Exit code is the number of failed checks, so it can gate a release.
"""
import argparse
import collections
import json


def audit(tasks):
    fams = collections.defaultdict(list)
    for t in tasks:
        fams[t["family"]].append(t)
    rows, fails = [], []

    q = collections.Counter(t["question"] for t in tasks)
    dup = sum(c - 1 for c in q.values() if c > 1)
    rows.append(("questions", f"{len(q)}/{len(tasks)} unique",
                 "OK" if dup == 0 else f"{dup} duplicate"))
    if dup:
        fails.append(f"{dup} duplicate questions")

    ocr = sum(t.get("has_ocr", 0) for t in tasks)
    rows.append(("text-only", f"has_ocr sum = {ocr}",
                 "OK" if ocr == 0 else "FAIL"))
    if ocr:
        fails.append("has_ocr is not zero")

    for f in sorted(fams):
        ts = fams[f]
        golds = collections.Counter(t["gold"] for t in ts)
        distinct, modal = len(golds), golds.most_common(1)[0][1]
        # Two independent ways gold diversity fails: too few distinct answers,
        # or one answer that a constant guesser could ride. The second is what
        # actually costs accuracy headroom, so it is the harder bound.
        ok_d = distinct >= 0.6 * len(ts)
        ok_m = modal <= 0.05 * len(ts)
        rows.append((f, f"{distinct} distinct, modal x{modal} of {len(ts)}",
                     "OK" if ok_d and ok_m else
                     ("thin" if ok_m else "CONCENTRATED")))
        if not ok_d:
            fails.append(f"{f}: {distinct} distinct golds < 0.6*{len(ts)}")
        if not ok_m:
            fails.append(f"{f}: modal gold appears {modal} times")
    return rows, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="/datasets/omni_pretraining/gta2/"
                                       "results/taco/tgb4/tgb_tasks.json")
    a = ap.parse_args()
    tasks = json.load(open(a.tasks))
    rows, fails = audit(tasks)
    w = max(len(r[0]) for r in rows)
    for name, detail, verdict in rows:
        print(f"  {name:<{w}}  {detail:<34} {verdict}")
    print(f"\n{len(rows) - len(fails)}/{len(rows)} checks clean"
          if not fails else f"\n{len(fails)} problem(s):")
    for f in fails:
        print(f"  - {f}")
    return len(fails)


if __name__ == "__main__":
    raise SystemExit(main())
