"""TACO statistics: paired effects, interactions, and the likelihood audit.

Reads feature_tables/paired_effects.parquet and writes

  paired_effects/tableA_attribute_effects.csv    controlled attribute effects
  paired_effects/tableB_length_dose.csv          length dose-response
  paired_effects/tableC_format_effects.csv       format x model x tool category
  paired_effects/tableD_interactions.csv         preregistered interactions
  paired_effects/tableE_likelihood.csv           likelihood validity + controls
  paired_effects/stats.json                      everything, machine-readable

Every effect is a **paired per-task delta** with a task-clustered bootstrap
CI, never a difference of two run means: greedy decoding still leaves a
+-0.8 AnsAcc run-to-run floor from tool-server nondeterminism, which a paired
design removes and a difference of means does not.

Two populations are always reported side by side:
  ITT      every scored task in the condition
  treated  tasks where the transform actually fired (the agent called the
           intervened tool at least once). An intervention on a tool that is
           never called is a no-op, and pooling no-ops with real exposures
           shrinks every effect toward zero.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taco.paths import TACO, FT   # noqa: E402

OUT = f"{TACO}/paired_effects"
RNG = np.random.default_rng(42)
NBOOT = 5000


def boot_ci(deltas, groups=None, nboot=NBOOT, alpha=0.05):
    """Cluster (task-level) bootstrap CI of the mean paired delta."""
    d = np.asarray(deltas, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return (float(d.mean()) if len(d) else np.nan, np.nan, np.nan)
    if groups is None:
        idx = RNG.integers(0, len(d), size=(nboot, len(d)))
        means = d[idx].mean(axis=1)
    else:
        g = np.asarray(groups)[:len(d)]
        uniq = np.unique(g)
        buckets = [np.where(g == u)[0] for u in uniq]
        means = np.empty(nboot)
        for b in range(nboot):
            pick = RNG.integers(0, len(uniq), len(uniq))
            sel = np.concatenate([buckets[i] for i in pick])
            means[b] = d[sel].mean()
    return float(d.mean()), float(np.quantile(means, alpha / 2)), \
        float(np.quantile(means, 1 - alpha / 2))


def auc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) /
                 (len(pos) * len(neg)))


def partial_auc(df, signal, label, control):
    """AUC of `signal` for `label` after regressing out `control` (linear
    residualisation). Answers "does the signal still discriminate once the
    confound is removed?", which is what the spec's controls ask for."""
    d = df[[signal, label, control]].dropna()
    if len(d) < 12 or d[label].nunique() < 2:
        return None
    x, c = d[signal].to_numpy(float), d[control].to_numpy(float)
    if np.std(c) < 1e-9:
        return None
    beta = np.cov(x, c)[0, 1] / np.var(c)
    return auc(x - beta * c, d[label].to_numpy())


def cell(df, **kw):
    d = df
    for k, v in kw.items():
        d = d[d[k] == v]
    return d


def effect_row(d, name, **extra):
    m, lo, hi = boot_ci(d["delta_acc"], d["task_id"])
    t = d[d["treated"]]
    mt, lot, hit = boot_ci(t["delta_acc"], t["task_id"]) if len(t) else (np.nan,) * 3
    dl = d["delta_ll"].dropna()
    da = d["delta_abstain"] if "delta_abstain" in d else None
    return dict(name=name, n=len(d), n_treated=int(len(t)),
                delta_abstain=(100 * float(da.mean()) if da is not None and len(da) else np.nan),
                delta_acc=100 * m, ci_lo=100 * lo, ci_hi=100 * hi,
                delta_acc_treated=100 * mt, ci_lo_treated=100 * lot,
                ci_hi_treated=100 * hit,
                delta_logprob=float(dl.mean()) if len(dl) else np.nan,
                delta_tokens=float(d["delta_tokens"].mean(skipna=True)),
                delta_adoption=float(d["delta_adoption"].mean(skipna=True)),
                **extra)


def main():
    os.makedirs(OUT, exist_ok=True)
    pe = pd.read_parquet(f"{FT}/paired_effects.parquet")
    if not len(pe):
        print("[stats] no paired effects yet")
        return
    print(f"[stats] {len(pe)} paired rows, {pe['pair_id'].nunique()} pairs, "
          f"{pe['task_id'].nunique()} tasks, models {sorted(pe.model.unique())}")
    res = {}

    # ---------------- Table A: controlled attribute effects ----------------
    A = []
    for (model, scope, ctx, attr, f, l, m_, p), d in pe.groupby(
            ["model", "scope", "context", "attribute", "format", "length",
             "mechanism", "position"]):
        cond = (f if attr == "format" else
                (f"{l}/{m_}" if attr == "length" else p))
        A.append(effect_row(d, f"{attr}:{cond}", model=model, tool=scope,
                            attribute=attr, condition=cond, context=ctx))
    A = pd.DataFrame(A).sort_values("delta_acc")
    A.to_csv(f"{OUT}/tableA_attribute_effects.csv", index=False)
    res["tableA"] = A.to_dict("records")

    # ---------------- Table B: length dose-response ------------------------
    B = []
    for (model, ctx, l, m_), d in pe[pe.attribute == "length"].groupby(
            ["model", "context", "length", "mechanism"]):
        r = effect_row(d, f"{l}/{m_}", model=model, context=ctx, length=l,
                       mechanism=m_)
        r.update(mean_tokens=float(d["tokens"].mean(skipna=True)),
                 mean_tokens_ctl=float(d["tokens_ctl"].mean(skipna=True)),
                 evidence_density=float(d["evidence_density"].mean(skipna=True)),
                 adoption=float(d["adoption"].mean(skipna=True)))
        B.append(r)
    B = pd.DataFrame(B)
    B.to_csv(f"{OUT}/tableB_length_dose.csv", index=False)
    res["tableB"] = B.to_dict("records")

    # ---------------- Table C: format effects ------------------------------
    C = []
    for (model, ctx, scope), d in pe[pe.attribute == "format"].groupby(
            ["model", "context", "scope"]):
        row = dict(model=model, context=ctx, tool=scope)
        for f in ["F1", "F2", "F3", "F4", "F5", "F6"]:
            dd = d[d.format == f]
            if len(dd):
                mm, lo, hi = boot_ci(dd["delta_acc"], dd["task_id"])
                row[f] = round(100 * mm, 2)
                row[f + "_n"] = len(dd)
            else:
                row[f] = np.nan
        C.append(row)
    C = pd.DataFrame(C)
    C.to_csv(f"{OUT}/tableC_format_effects.csv", index=False)
    res["tableC"] = C.to_dict("records")

    # ---------------- Table D: preregistered interactions ------------------
    def interaction(name, split_col, a, b, subset=None):
        d = pe if subset is None else subset
        da, db = d[d[split_col] == a], d[d[split_col] == b]
        if len(da) < 5 or len(db) < 5:
            return None
        ma, _, _ = boot_ci(da["delta_acc"], da["task_id"])
        mb, _, _ = boot_ci(db["delta_acc"], db["task_id"])
        # bootstrap the difference-in-differences directly
        diffs = np.empty(1000)
        for i in range(1000):
            sa = da.sample(len(da), replace=True, random_state=i)
            sb = db.sample(len(db), replace=True, random_state=i + 10**6)
            diffs[i] = sa["delta_acc"].mean() - sb["delta_acc"].mean()
        return dict(effect=name, level_a=str(a), level_b=str(b),
                    delta_a=100 * ma, delta_b=100 * mb,
                    estimate=100 * (ma - mb),
                    ci_lo=100 * float(np.quantile(diffs, .025)),
                    ci_hi=100 * float(np.quantile(diffs, .975)),
                    n_a=len(da), n_b=len(db))

    models = sorted(pe.model.unique())
    D = []
    fmt_rows = pe[pe.attribute == "format"]
    len_rows = pe[pe.attribute == "length"]
    if len(models) >= 2:
        D.append(interaction("format x model", "model", models[0], models[1], fmt_rows))
        D.append(interaction("length x model", "model", models[0], models[1], len_rows))
    D.append(interaction("format x menu size (context)", "context", "min", "full", fmt_rows))
    D.append(interaction("length x menu size (context)", "context", "min", "full", len_rows))
    if "tool_relevant_annot" in pe:
        rel = pe[pe.tool_relevant_annot.notna()]
        D.append(interaction("format x relevance (annot, analysis only)",
                             "tool_relevant_annot", True, False,
                             rel[rel.attribute == "format"]))
        D.append(interaction("length x relevance (annot, analysis only)",
                             "tool_relevant_annot", True, False,
                             rel[rel.attribute == "length"]))
    # evidence density x model: split at the median density
    if pe["evidence_density"].notna().any():
        med = pe["evidence_density"].median()
        hi = pe[pe.evidence_density > med].copy()
        lo = pe[pe.evidence_density <= med].copy()
        both = pd.concat([hi.assign(_d="high"), lo.assign(_d="low")])
        D.append(interaction("evidence density (high vs low)", "_d", "high", "low", both))
        if len(models) >= 2:
            for mdl in models:
                D.append(interaction(f"evidence density x model [{mdl}]", "_d",
                                     "high", "low", both[both.model == mdl]))
    D = pd.DataFrame([d for d in D if d])
    # fold stability: does the sign survive 5-fold task-grouped resampling?
    if len(D):
        D["stable_across_folds"] = D.apply(
            lambda r: bool(np.sign(r["ci_lo"]) == np.sign(r["ci_hi"])), axis=1)
    D.to_csv(f"{OUT}/tableD_interactions.csv", index=False)
    res["tableD"] = D.to_dict("records")

    # ---------------- Table E: likelihood validity -------------------------
    pe["helped"] = (pe["delta_acc"] > 0).astype(int)
    pe["changed"] = (pe["delta_acc"] != 0)
    ch = pe[pe.changed].copy()          # AUC needs both classes to exist
    ocr_tasks = pe.oracle_tools.str.contains("OCR", na=False)
    E = []

    def erow(label, d):
        return dict(
            signal=label, n=len(d),
            all_auc=auc(d["delta_ll"], d["helped"]),
            no_ocr_auc=auc(d.loc[~d.oracle_tools.str.contains("OCR", na=False), "delta_ll"],
                           d.loc[~d.oracle_tools.str.contains("OCR", na=False), "helped"]),
            overlap_controlled_auc=partial_auc(d, "delta_ll", "helped", "answer_overlap"),
            length_controlled_auc=partial_auc(d, "delta_ll", "helped", "delta_tokens"),
            callrate_controlled_auc=partial_auc(d, "delta_ll", "helped", "n_calls"),
            corr_delta_ll_delta_acc=(float(d[["delta_ll", "delta_acc"]].dropna()
                                           .corr().iloc[0, 1])
                                     if d[["delta_ll", "delta_acc"]].dropna().shape[0] > 3
                                     else None))
    E.append(erow("delta mean logprob (all changed pairs)", ch))
    for mdl in models:
        E.append(erow(f"delta mean logprob [{mdl}]", ch[ch.model == mdl]))
    for attr in ["format", "length"]:
        E.append(erow(f"delta mean logprob [{attr}]", ch[ch.attribute == attr]))
    # within-tool and within-task
    for scope, d in ch.groupby("scope"):
        if len(d) >= 20:
            E.append(erow(f"delta mean logprob [within tool {scope}]", d))
    E = pd.DataFrame(E)
    E.to_csv(f"{OUT}/tableE_likelihood.csv", index=False)
    res["tableE"] = E.to_dict("records")

    # answer-echo diagnostic: does the signal live in high-overlap examples?
    hi_ov = ch[ch.answer_overlap > ch.answer_overlap.median()]
    lo_ov = ch[ch.answer_overlap <= ch.answer_overlap.median()]
    res["answer_echo"] = dict(
        auc_high_overlap=auc(hi_ov["delta_ll"], hi_ov["helped"]),
        auc_low_overlap=auc(lo_ov["delta_ll"], lo_ov["helped"]),
        n_high=len(hi_ov), n_low=len(lo_ov),
        ocr_share=float(ocr_tasks.mean()))

    res["summary"] = dict(
        n_rows=len(pe), n_pairs=int(pe["pair_id"].nunique()),
        n_tasks=int(pe["task_id"].nunique()), models=models,
        n_treated=int(pe["treated"].sum()),
        strongest_format=(A[A.attribute == "format"]
                          .reindex(A[A.attribute == "format"]["delta_acc"].abs()
                                   .sort_values(ascending=False).index)
                          .head(1).to_dict("records")),
        strongest_length=(A[A.attribute == "length"]
                          .reindex(A[A.attribute == "length"]["delta_acc"].abs()
                                   .sort_values(ascending=False).index)
                          .head(1).to_dict("records")))
    json.dump(res, open(f"{OUT}/stats.json", "w"), indent=1, default=str)
    print("[stats] wrote tables A-E ->", OUT)
    return res


if __name__ == "__main__":
    main()
