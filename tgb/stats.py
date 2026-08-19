"""Dataset card for TGB: what is in it, and what a record looks like.

    python -m tgb.stats [--example]
"""
import argparse
import collections
import json
import os
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.environ.get(
        "TGB_OUT", "/datasets/omni_pretraining/gta2/results/taco/tgb"))
    ap.add_argument("--example", action="store_true")
    a = ap.parse_args()
    T = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    print(f"tasks: {len(T)}")
    fam = collections.Counter(t["family"] for t in T)
    for f, c in sorted(fam.items()):
        ex = next(t for t in T if t["family"] == f)
        print(f"  {f:<22} {c:>5}   chain: {' -> '.join(ex['gt_tools'])}")
    print(f"\nconditions per task: "
          f"{sorted(collections.Counter(len(t['conditions']) for t in T).items())}")
    names = collections.Counter()
    for t in T:
        names.update(t["conditions"].keys())
    print("condition coverage:", dict(names))

    print("\ncontext length (chars) by condition")
    L = collections.defaultdict(list)
    for t in T:
        for n, c in t["conditions"].items():
            L[n].append(len(c["context"]))
    for n in sorted(L, key=lambda k: -st.mean(L[k])):
        print(f"  {n:<15} mean {st.mean(L[n]):>7.0f}  max {max(L[n]):>6}")

    tot = collections.Counter()
    for t in T:
        tot.update(t["conditions"]["useful"]["tools"])
    print("\nGT-chain tool frequency:", dict(tot))
    print("distinct images:", len({t["image"] for t in T}))

    if a.example:
        t = T[0]
        print("\n" + "=" * 70 + "\nEXAMPLE RECORD\n" + "=" * 70)
        print(f"id       {t['id']}\nfamily   {t['family']}\nimage    {t['image']}")
        print(f"question {t['question']}")
        print(f"gold     {t['gold']}   (= {t['meta']['final_op']} of "
              f"intermediate {t['meta']['intermediate']} and k={t['meta']['k']})")
        for n, c in t["conditions"].items():
            print(f"\n-- {n}  tools={c['tools']}  label={c['label']}")
            print("   " + (c["context"] or "(no tools)").replace("\n", "\n   "))


if __name__ == "__main__":
    main()
