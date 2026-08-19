"""Per-task likelihood masking against greedy global search, on TGB-v4.

    python -m tgb.compare_v4

Both methods are read on the same 3,600-task test split and the same 15-tool
action space, and both are allowed the same 400 labelled tasks: greedy searches
with them, the per-task selector picks its tau with them. Reporting a tau
chosen on test would reproduce, invisibly, the overfitting greedy is being
measured for.

The comparison that matters is not only "does it beat greedy" -- on two of four
models greedy loses to the fixed union, so that bar is low. Both deltas are
printed: against greedy, and against the best constant policy.
"""
import argparse
import json
import os

T4D = "/datasets/omni_pretraining/gta2/results/taco/tgb4"
MODELS = ["7b", "llama8b", "mistral7b", "14b"]


def load(d, tag):
    g = os.path.join(d, f"tgb_greedy_{tag}_backward.json")
    l = os.path.join(d, f"tgb_loo_{tag}.json")
    G = json.load(open(g)) if os.path.exists(g) else None
    L = json.load(open(l)) if os.path.exists(l) else None
    return G, L


def refs_of(G):
    """greedy_env stores test accuracy per named mask; shapes have drifted
    across versions, so read defensively rather than assume one key."""
    for k in ("test", "test_acc", "refs_test", "refs"):
        if isinstance(G.get(k), dict):
            return G[k]
    return {k: v for k, v in G.items() if isinstance(v, (int, float))}


def split_guard(d, G):
    """greedy stores the exact ids it tested on. loo_mask recomputes the split
    from the same rule, which is not the same thing as computing the same set --
    a drift in either would silently compare two different test sets and look
    like a method effect."""
    tasks = json.load(open(os.path.join(d, "tgb_tasks.json")))
    step = max(1, len(tasks) // G["n_train"])
    tr = {t["id"] for t in tasks[::step][:G["n_train"]]}
    mine = {t["id"] for t in tasks if t["id"] not in tr}
    theirs = set(G["test_ids"])
    return mine == theirs, len(mine ^ theirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=T4D)
    a = ap.parse_args()

    print(f"{'model':<11}{'greedy':>9}{'best const':>12}{'LOO test':>10}"
          f"{'tau':>7}{'tools':>7}{'vs greedy':>11}{'vs const':>10}")
    rows = []
    for m in MODELS:
        G, L = load(a.dir, m)
        if G is None or L is None:
            print(f"{m:<11}  {'greedy' if G is None else 'loo'} missing")
            continue
        ok, diff = split_guard(a.dir, G)
        if not ok:
            print(f"{m:<11}  SPLIT MISMATCH: {diff} ids differ -- not comparable")
            continue
        R = refs_of(G)
        gr = R.get("greedy")
        const = max(v for k, v in R.items()
                    if k in ("none", "full", "union", "union_minus_calc"))
        sel = L["selected"]
        d_g, d_c = sel["test"] - gr, sel["test"] - const
        rows.append((m, gr, const, sel, d_g, d_c))
        print(f"{m:<11}{gr:>9.3f}{const:>12.3f}{sel['test']:>10.3f}"
              f"{sel['tau']:>+7.2f}{sel['tools']:>7.2f}"
              f"{d_g:>+11.3f}{d_c:>+10.3f}")

    if rows:
        n = len(rows)
        wg = sum(r[4] > 0 for r in rows)
        wc = sum(r[5] > 0 for r in rows)
        print(f"\n  beats greedy on {wg}/{n}   beats best constant on {wc}/{n}")
        print(f"  mean delta: vs greedy {sum(r[4] for r in rows)/n:+.4f}   "
              f"vs constant {sum(r[5] for r in rows)/n:+.4f}")
        print("\n  Sign agreement across models is the claim; a mean over four "
              "models is not a significance test.")


if __name__ == "__main__":
    main()
