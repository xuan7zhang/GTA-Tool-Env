"""Stage 4 — global tool-space search (spec §19), calibration data only.

Searches for ONE fixed environment per model

    E* = argmax_E  mean_{q in D_cal} U(q, E, m)  -  lambda * Cost(E)

over

    M    which tools are globally exposed (every query sees exactly M)
    Phi  the output policy for the retained tools (format, length budget)
    C    composition: fixed (the one predefined safe macro option is left out
         of this first experiment, per the spec)

Utility is predicted, not measured, so the search is cheap; it is decomposed
because the two axes are measured by different families of runs:

    U(E)  ~=  acc_mask(M)          absolute, fitted on the subset-design runs
            + delta_phi(Phi | M)   relative, fitted on the paired attribute runs

Nothing here reads a held-out task, a gold answer, or an annotation: the
leakage guard in the runner covers the runs, and this module only consumes
calibration-fitted models plus the calibration run table.

Searches run: greedy forward, greedy backward, beam (b in 3/5/10), random,
and coordinate descent over (mask, format, length). An *empirical* greedy
(calibration accuracy rather than predicted utility) is run separately as the
outcome-based baseline, because it costs O(n) real evaluations.
"""
import itertools
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taco.paths import TACO, FT, MODELS, SEARCH, TOOLMETA   # noqa: E402

FORMATS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
LENGTHS = ["L4", "L0", "L1", "L2", "L3", "L5"]
BEAMS = [3, 5, 10]
K_BUDGETS = [3, 5, 7, 9, 11, 14]
LAMBDA = float(os.environ.get("TACO_LAMBDA", "0.0"))   # cost weight, tokens/1k


def load():
    tools = list(json.load(open(TOOLMETA)))
    sf = pd.read_parquet(f"{FT}/subset_features.parquet")
    runs = pd.read_parquet(f"{FT}/runs.parquet")
    mask_model = None
    if os.path.exists(f"{MODELS}/taco_mask.pkl"):
        mask_model = pickle.load(open(f"{MODELS}/taco_mask.pkl", "rb"))
    attr = {}
    for name in ("taco_intrinsic", "taco_conditional"):
        p = f"{MODELS}/{name}.pkl"
        if os.path.exists(p):
            attr[name] = pickle.load(open(p, "rb"))
    return tools, sf, runs, mask_model, attr


CATEGORY = {
    "OCR": "text_extraction", "MathOCR": "text_extraction",
    "ImageDescription": "image_description",
    "TextToBbox": "localization", "DrawBox": "localization",
    "RegionAttributeDescription": "regional_inspection",
    "CountGivenObject": "regional_inspection",
    "Calculator": "calculation", "Solver": "calculation", "Plot": "calculation",
    "GoogleSearch": "retrieval",
    "TextToImage": "generation", "ImageStylization": "generation", "AddText": "generation",
}


def mask_features(M, toolmeta):
    cats = [CATEGORY.get(t, "other") for t in M]
    return dict(menu_size=len(M), n_categories=len(set(cats)),
                max_same_category=max([cats.count(c) for c in set(cats)] or [0]),
                desc_tokens=sum(len((toolmeta.get(t, {}).get("description") or "").split())
                                for t in M),
                n_image_output=sum(1 for t in M if CATEGORY.get(t) == "generation"),
                n_unavailable=sum(1 for t in M if t in ("GoogleSearch", "MathOCR")))


def phi_row(fmt, length, menu_size, model, columns, ctx_full=1):
    """One design row for the attribute model, for a uniform (fmt, length)."""
    from taco.model import MODEL_SCALE
    r = {c: 0.0 for c in columns}
    if f"fmt_{fmt}" in r:
        r[f"fmt_{fmt}"] = 1.0
    if f"len_{length}" in r:
        r[f"len_{length}"] = 1.0
    r["is_json"] = float(fmt == "F4")
    r["is_kv"] = float(fmt == "F3")
    r["menu_size"] = float(menu_size)
    r["context_full"] = float(ctx_full)
    r["model_scale"] = float(MODEL_SCALE.get(model, 7))
    return r


class Utility:
    """Predicted calibration utility of a global environment E=(M, fmt, len)."""

    def __init__(self, model, toolmeta, mask_model, attr_model, sf, lam=LAMBDA):
        self.model, self.toolmeta = model, toolmeta
        self.mask_model, self.attr = mask_model, attr_model
        self.sf, self.lam = sf, lam
        # empirical fallback for the mask term: mean calibration accuracy of
        # the observed subsets, nearest by size, when no mask model was fitted
        self.by_size = (sf.groupby("menu_size")["answer_acc"].mean().to_dict()
                        if len(sf) else {})
        self.tok_by_size = (sf.groupby("menu_size")["visible_tool_tokens"].mean().to_dict()
                            if len(sf) else {})

    def acc_mask(self, M):
        if self.mask_model is not None:
            f = mask_features(M, self.toolmeta)
            X = pd.DataFrame([[f[c] for c in self.mask_model["features"]]],
                             columns=self.mask_model["features"])
            return float(self.mask_model["reg"].predict(X)[0])
        if self.by_size:
            k = min(self.by_size, key=lambda s: abs(s - len(M)))
            return float(self.by_size[k])
        return 0.0

    def delta_phi(self, fmt, length, M):
        if not self.attr:
            return 0.0
        m = self.attr.get("taco_intrinsic")
        if m is None or m.get("reg") is None:
            return 0.0
        r = phi_row(fmt, length, len(M), self.model, m["columns"])
        X = pd.DataFrame([[r[c] for c in m["columns"]]], columns=m["columns"])
        return 100.0 * float(m["reg"].predict(X)[0])       # delta_acc is 0/1 scale

    def cost(self, M, fmt, length):
        base = self.tok_by_size.get(min(self.tok_by_size, key=lambda s: abs(s - len(M))),
                                    0.0) if self.tok_by_size else 0.0
        mult = {"F0": 1.0, "F1": 0.3, "F2": 0.7, "F3": 0.6, "F4": 2.2, "F5": 0.4,
                "F6": 1.6}.get(fmt, 1.0)
        lmult = {"L0": 0.15, "L1": 0.3, "L2": 0.5, "L3": 0.8, "L4": 1.0,
                 "L5": 2.0}.get(length, 1.0)
        return base * mult * lmult

    def __call__(self, M, fmt="F0", length="L4"):
        if not M:
            return -1e9
        u = self.acc_mask(M) + self.delta_phi(fmt, length, M)
        return u - self.lam * self.cost(M, fmt, length) / 1000.0


def greedy_forward(U, tools, kmax=14):
    cur, traj = [], []
    remaining = list(tools)
    while remaining and len(cur) < kmax:
        best = max(remaining, key=lambda t: U(cur + [t]))
        gain = U(cur + [best]) - (U(cur) if cur else -1e9)
        cur = sorted(cur + [best])
        remaining.remove(best)
        traj.append(dict(step=len(cur), added=best, tools=list(cur), utility=U(cur)))
    return traj


def greedy_backward(U, tools):
    cur, traj = list(tools), [dict(step=len(tools), removed=None,
                                   tools=list(tools), utility=U(list(tools)))]
    while len(cur) > 1:
        worst = max(cur, key=lambda t: U([x for x in cur if x != t]))
        cur = [x for x in cur if x != worst]
        traj.append(dict(step=len(cur), removed=worst, tools=list(cur), utility=U(cur)))
    return traj


def beam_search(U, tools, width, kmax=14):
    beam = [([], U([t for t in tools]))]
    beam = [([], -1e9)]
    traj = []
    for k in range(1, kmax + 1):
        cand = {}
        for cur, _ in beam:
            for t in tools:
                if t in cur:
                    continue
                nxt = tuple(sorted(cur + [t]))
                if nxt not in cand:
                    cand[nxt] = U(list(nxt))
        if not cand:
            break
        beam = sorted(((list(k_), v) for k_, v in cand.items()),
                      key=lambda x: -x[1])[:width]
        traj.append(dict(k=k, best_tools=beam[0][0], best_utility=beam[0][1],
                         beam=[b[0] for b in beam]))
    return traj


def random_search(U, tools, n=400, seed=0):
    rng = np.random.default_rng(seed)
    best, rows = None, []
    for i in range(n):
        k = int(rng.integers(1, len(tools) + 1))
        M = sorted(rng.choice(tools, size=k, replace=False).tolist())
        u = U(M)
        rows.append(dict(tools=M, utility=u))
        if best is None or u > best["utility"]:
            best = dict(tools=M, utility=u)
    return best, rows


def coordinate_descent(U, tools, M0, rounds=3):
    M, fmt, length = list(M0), "F0", "L4"
    traj = []
    for r in range(rounds):
        fmt = max(FORMATS, key=lambda f: U(M, f, length))
        length = max(LENGTHS, key=lambda l: U(M, fmt, l))
        # one mask sweep: try dropping / adding a single tool
        best = (U(M, fmt, length), list(M))
        for t in tools:
            cand = [x for x in M if x != t] if t in M else sorted(M + [t])
            if not cand:
                continue
            u = U(cand, fmt, length)
            if u > best[0]:
                best = (u, cand)
        M = best[1]
        traj.append(dict(round=r, tools=list(M), format=fmt, length=length,
                         utility=best[0]))
    return dict(tools=sorted(M), format=fmt, length=length, utility=U(M, fmt, length)), traj


def topk_individual(U, tools, k):
    """Global top-k by individual utility (a required baseline, not a search)."""
    scores = {t: U([t]) for t in tools}
    keep = sorted(sorted(scores, key=lambda t: -scores[t])[:k])
    return dict(tools=keep, utility=U(keep), individual_scores=scores)


def empirical_from_runs(runs: pd.DataFrame, model: str):
    """Best environment actually observed on calibration (empirical search)."""
    d = runs[(runs.model == model) & (runs.kind == "calibration")
             & (runs.menu_mode == "forced")]
    if not len(d):
        return None
    b = d.sort_values("answer_acc", ascending=False).iloc[0]
    return dict(run_id=b.run_id, tools=b.menu_tools.split(","),
                answer_acc=float(b.answer_acc), n_scored=int(b.n_scored),
                note="best observed calibration environment (empirical greedy proxy)")


def main():
    os.makedirs(SEARCH, exist_ok=True)
    tools, sf, runs, mask_model, attr = load()
    toolmeta = json.load(open(TOOLMETA))
    models = sorted(runs[runs.kind == "calibration"]["model"].dropna().unique())
    out = {}
    for model in models:
        sfm = sf[sf.model == model] if len(sf) else sf
        U = Utility(model, toolmeta, mask_model, attr, sfm)
        res = dict(model=model, lam=LAMBDA)
        res["greedy_forward"] = greedy_forward(U, tools)
        res["greedy_backward"] = greedy_backward(U, tools)
        res["beam"] = {str(b): beam_search(U, tools, b) for b in BEAMS}
        best_rand, rand_rows = random_search(U, tools)
        res["random"] = dict(best=best_rand,
                             median_utility=float(np.median([r["utility"] for r in rand_rows])),
                             n=len(rand_rows))
        res["topk_individual"] = {str(k): topk_individual(U, tools, k) for k in K_BUDGETS}
        # start coordinate descent from the best mask any search found
        cands = [max(res["greedy_forward"], key=lambda r: r["utility"])["tools"],
                 max(res["greedy_backward"], key=lambda r: r["utility"])["tools"],
                 best_rand["tools"]]
        for b in BEAMS:
            if res["beam"][str(b)]:
                cands.append(max(res["beam"][str(b)],
                                 key=lambda r: r["best_utility"])["best_tools"])
        start = max(cands, key=lambda M: U(M))
        cd, cd_traj = coordinate_descent(U, tools, start)
        res["coordinate_descent"] = dict(best=cd, trajectory=cd_traj)
        res["empirical_best_observed"] = empirical_from_runs(runs, model)
        res["selected"] = cd
        out[model] = res
        print(f"[search] {model}: selected |M|={len(cd['tools'])} "
              f"{cd['format']}/{cd['length']} u={cd['utility']:.2f}")

    json.dump(out, open(f"{SEARCH}/search_results.json", "w"), indent=1, default=str)
    print("[search] ->", f"{SEARCH}/search_results.json")
    return out


if __name__ == "__main__":
    main()
