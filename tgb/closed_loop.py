"""TGB analysis: coalition ranking + closed-loop tool selection.

    python -m tgb.closed_loop --tag 7b [--tag 14b ...]

Two questions, in the order the paper needs them.

1. RANKING (the §5/§6 claim, moved up to the tool level). Does dL put the
   complete clean coalition above every degraded one? Reported as AUROC of
   `useful` against each competing condition, on the derivable subset.

2. CLOSED LOOP (the gap §9 admits). A selector sees the candidate coalitions,
   picks one, that coalition's tools *actually ran*, and the model answers
   under exactly that context. We report the accuracy it ends up with and what
   it paid in tool calls -- which is the claim "the signal improves downstream
   agent performance", not just "the signal ranks".

   The action space is the selectable tool subsets. `corrupt` is excluded from
   it because it is not an action -- it is the same tool set as `useful` in a
   degraded world -- and is reported separately as the output-quality probe.
"""
import argparse
import collections
import json
import os
import random
import statistics as st

ACTIONS = ["none", "useful", "partial_1of2", "no_downstream", "no_upstream",
           "wrong", "full"]
TOOLDESC = None


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / \
        (len(pos) * len(neg))


def cheapest_near_max(r, avail, score, T, eps=0.10):
    """Cost-aware selection: among coalitions within `eps` of the best score,
    take the one with the fewest tools. Without this the selector is
    indifferent between `useful` and `full`, which look alike to dL and cost
    several times as much to run."""
    vals = {c: score(r, c) for c in avail}
    best = max(vals.values())
    near = [c for c, v in vals.items() if v >= best - eps]
    return min(near, key=lambda c: len(T[r["id"]]["conditions"][c]["tools"]))


def load_tooldesc(path):
    meta = json.load(open(path))
    return {k: (v.get("description") or "") for k, v in meta.items()}


def relevance_score(question, tools, desc):
    """Label-free lexical relevance -- the selection baseline from the
    companion analysis. Mean over the coalition's tools of the token overlap
    between the question and the tool's own description."""
    qt = set(w.strip(".,?$").lower() for w in question.split() if len(w) > 3)
    if not tools:
        return 0.0
    s = []
    for t in tools:
        dt = set(w.strip(".,?").lower() for w in desc.get(t, "").split()
                 if len(w) > 3)
        s.append(len(qt & dt) / max(1, len(dt)) if dt else 0.0)
    return sum(s) / len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--dir", default=os.environ.get(
        "TGB_OUT", "/datasets/omni_pretraining/gta2/results/taco/tgb"))
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--toolmeta", default=os.environ.get(
        "TOOLMETA",
        "/datasets/omni_pretraining/gta2/data/gta_dataset/toolmeta.json"))
    a = ap.parse_args()
    tasks = json.load(open(a.tasks or os.path.join(a.dir, "tgb_tasks.json")))
    T = {t["id"]: t for t in tasks}
    desc = load_tooldesc(a.toolmeta)

    for tag in a.tag:
        rows = json.load(open(os.path.join(a.dir, f"tgb_scored_{tag}.json")))
        sc_path = os.path.join(a.dir, f"tgb_selfcons_{tag}.json")
        SC = {r["id"]: r["conds"] for r in json.load(open(sc_path))} \
            if os.path.exists(sc_path) else {}
        deriv = [r for r in rows if r["derivable"]]
        print(f"\n{'='*72}\n[{tag}] n={len(rows)}  derivable={len(deriv)} "
              f"({len(deriv)/len(rows):.1%})\n{'='*72}")

        # ---------------------------------------------------- 1. ranking
        print("\n-- dL ranking: useful vs each degraded coalition "
              "(derivable subset)")
        print(f"   {'competitor':<15}{'AUROC':>8}{'useful>comp':>13}"
              f"{'mean dL(useful)':>17}{'mean dL(comp)':>15}")
        for comp in ["no_downstream", "partial_1of2", "corrupt", "no_upstream",
                     "wrong", "full", "none"]:
            pos = [r["conds"]["useful"]["dL"] for r in deriv
                   if comp in r["conds"] and r["conds"]["useful"]["dL"] is not None
                   and r["conds"][comp]["dL"] is not None]
            neg = [r["conds"][comp]["dL"] for r in deriv
                   if comp in r["conds"] and r["conds"]["useful"]["dL"] is not None
                   and r["conds"][comp]["dL"] is not None]
            if not pos:
                continue
            au = auroc(pos, neg)
            win = sum(p > n for p, n in zip(pos, neg)) / len(pos)
            print(f"   {comp:<15}{au:>8.3f}{win:>12.0%}"
                  f"{st.mean(pos):>17.3f}{st.mean(neg):>15.3f}")

        print("\n-- per family (AUROC useful vs the union of degraded)")
        byfam = collections.defaultdict(lambda: ([], []))
        for r in deriv:
            u = r["conds"]["useful"]["dL"]
            if u is None:
                continue
            byfam[r["family"]][0].append(u)
            for n, c in r["conds"].items():
                if n not in ("useful", "none") and c["dL"] is not None:
                    byfam[r["family"]][1].append(c["dL"])
        for f, (p, n) in sorted(byfam.items()):
            print(f"   {f:<22} n={len(p):<5} AUROC={auroc(p, n):.3f}")

        # ------------------------------------------------- 2. closed loop
        print("\n-- closed loop: select a coalition, run it, answer\n")
        print(f"   {'selector':<20}{'accuracy':>10}{'tools/task':>12}"
              f"{'GT recall':>11}{'GT Jaccard':>12}{'needs gold':>12}")
        rnd = random.Random(11)

        def report(name, pick, needs_gold, reps=1):
            acc, cost, jac, rec = [], [], [], []
            for _ in range(reps):
                for r in rows:
                    avail = [c for c in ACTIONS if c in r["conds"]]
                    ch = pick(r, avail)
                    cond = T[r["id"]]["conditions"][ch]
                    acc.append(r["conds"][ch]["acc"])
                    cost.append(len([t for t in cond["tools"]]))
                    gt = set(T[r["id"]]["gt_tools"])
                    s = set(cond["tools"])
                    jac.append(len(gt & s) / len(gt | s) if (gt | s) else 1.0)
                    rec.append(len(gt & s) / len(gt))
            print(f"   {name:<20}{st.mean(acc):>10.3f}{st.mean(cost):>12.2f}"
                  f"{st.mean(rec):>11.3f}{st.mean(jac):>12.3f}"
                  f"{'yes' if needs_gold else 'no':>12}")

        def dl(r, c):
            v = r["conds"][c]["dL"]
            return v if v is not None else -99

        report("all tools", lambda r, av: "full", False)
        report("no tools", lambda r, av: "none", False)
        report("random", lambda r, av: rnd.choice(av), False, reps=20)
        report("relevance", lambda r, av: max(
            av, key=lambda c: relevance_score(
                T[r["id"]]["question"],
                T[r["id"]]["conditions"][c]["tools"], desc)), False)
        report("gold likelihood", lambda r, av: max(av, key=lambda c: dl(r, c)),
               True)
        report("gold likelihood +c", lambda r, av: cheapest_near_max(r, av, dl, T),
               True)
        if SC:
            sub = [r for r in rows if r["id"] in SC]
            saved, rows_all = rows, None
            rows_all, rows = rows, sub
            def sc(r, c):
                return SC[r["id"]].get(c, {}).get("agree", 0)

            report("self-consistency", lambda r, av: max(
                av, key=lambda c: sc(r, c)), False)
            report("self-consistency +c", lambda r, av: cheapest_near_max(
                r, av, sc, T, eps=0.01), False)
            report("gold likelihood*", lambda r, av: max(
                av, key=lambda c: dl(r, c)), True)
            print(f"   (* the two rows above are the n={len(sub)} "
                  f"self-consistency subsample)")
            rows = rows_all
        report("oracle", lambda r, av: max(
            av, key=lambda c: (r["conds"][c]["acc"],
                               -len(T[r["id"]]["conditions"][c]["tools"]))), True)


if __name__ == "__main__":
    main()
