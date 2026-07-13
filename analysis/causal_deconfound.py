#!/usr/bin/env python3
"""De-confound the causal 2x2 by splitting on dead external tools.

With no Serper/Mathpix keys, GoogleSearch/MathOCR are 'unavailable' at the
proxy, so tasks touching them are already partly tool-free in the passthrough
(A) run. This inflates the "tools don't help" reading. Here we recompute the
2x2 + headline stats on the tool-clean complement, using existing per-sample
predictions (no model re-run).

Usage: python causal_deconfound.py \
  --per-sample analysis/causal_report_per_sample.json \
  --dataset .../gta_dataset/dataset.json \
  --dead GoogleSearch MathOCR --out analysis/causal_deconfound.md
"""
import argparse
import json


def decompose(ps, idxs):
    S = {i: ps[str(i)] for i in idxs
         if str(i) in ps and ps[str(i)]["A"] is not None and ps[str(i)]["B"] is not None}
    n = len(S)
    both = sum(1 for r in S.values() if r["A"] and r["B"])
    resc = sum(1 for r in S.values() if r["A"] and not r["B"])
    hurt = sum(1 for r in S.values() if not r["A"] and r["B"])
    neither = sum(1 for r in S.values() if not r["A"] and not r["B"])
    ac = [r for r in S.values() if r["A"]]
    act = [r for r in ac if r.get("A_tools")]
    surv = [r for r in act if r.get("C")]
    return dict(n=n, both=both, rescued=resc, hurt=hurt, neither=neither,
                b_correct=both + hurt, a_correct=len(ac),
                a_tool=len(act), survive_c=len(surv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sample", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dead", nargs="*", default=["GoogleSearch", "MathOCR"])
    ap.add_argument("--out", default="analysis/causal_deconfound.md")
    args = ap.parse_args()

    data = json.load(open(args.dataset))
    ps = json.load(open(args.per_sample))
    dead = set(args.dead)
    touches = {int(k): bool(dead & {t["name"] for t in v["tools"]})
               for k, v in data.items()}
    allidx = [int(k) for k in data]
    clean = [i for i in allidx if not touches[i]]
    dirty = [i for i in allidx if touches[i]]

    lines = [f"# Causal de-confound: dead tools {sorted(dead)}", "",
             f"229 tasks | touching dead tools: {len(dirty)} | tool-clean: {len(clean)}",
             "", "| subset | n | both | rescued | hurt | neither | %solvable-no-tools | %survive-corrupt |",
             "|---|---|---|---|---|---|---|---|"]
    for name, idxs in [("all", allidx), ("tool-clean", clean), ("dead-tool-only", dirty)]:
        d = decompose(ps, idxs)
        sc = f"{d['survive_c']}/{d['a_tool']} = {d['survive_c']/d['a_tool']*100:.1f}%" if d["a_tool"] else "n/a"
        lines.append(f"| {name} | {d['n']} | {d['both']} | {d['rescued']} | {d['hurt']} | "
                     f"{d['neither']} | {d['b_correct']}/{d['n']} = {d['b_correct']/d['n']*100:.1f}% | {sc} |")
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
