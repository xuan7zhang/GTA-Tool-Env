"""TACO utility models (spec §15/§16/§18).

Two predictors, deliberately different in what they are allowed to see:

  TACO-Intrinsic     tool attributes + model attributes + subset attributes.
                     Never inspects the query. This is the one that can drive
                     a *task-level* global environment choice, because it does
                     not need the query to score an environment.

  TACO-Conditional   adds label-free task-tool relation features (query/tool
                     description similarity, query length). Used two ways:
                     averaged over calibration tasks as a stronger global
                     optimiser, and per-query as a SELECTOR BASELINE. The two
                     uses are never mixed in reporting.

Annotation-derived columns (annotated relevance, reference-chain membership)
exist only in the *analysis* model. The deployable models are fitted on a
matrix from which those columns are physically dropped, and the code asserts
their absence rather than trusting a flag.

Utility is decomposed, because the two axes are measured by different runs:

  U(E) ~= acc_mask(M)            fitted on the subset-design runs (absolute)
        + delta_phi(Phi | M)     fitted on the paired attribute runs (relative)

Model family is not a single black box: a ridge logistic/linear model, a
gradient-boosted tree, and a spline GAM are all fitted and compared under
grouped CV (group = task), plus leave-one-tool-out and
leave-one-category-out.
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taco.paths import TACO, FT, MODELS   # noqa: E402

ANNOTATION_COLS = ["tool_relevant_annot", "annot_relevant", "annot_in_ref_chain",
                   "oracle_tools", "ref_chain_len", "task_category"]

INTRINSIC_COLS = [
    # tool / attribute
    "fmt_F1", "fmt_F2", "fmt_F3", "fmt_F4", "fmt_F5", "fmt_F6",
    "is_json", "is_kv", "len_L0", "len_L1", "len_L2", "len_L3", "len_L5",
    "mech_irrelevant", "mech_redundant", "pos_front", "pos_middle",
    "tokens", "delta_tokens", "evidence_density",
    # subset / context
    "menu_size", "context_full",
    # model
    "model_scale", "model_notool_acc", "model_full_acc",
]
CONDITIONAL_EXTRA = ["query_tool_jaccard", "query_len"]

MODEL_SCALE = {"qwen2.5-3b-instruct": 3, "qwen2.5-7b-instruct": 7,
               "qwen2.5-14b-instruct": 14, "qwen2.5-32b-instruct": 32,
               "llama-3.1-8b-instruct": 8}


def design(pe: pd.DataFrame, tt: pd.DataFrame, conditional: bool):
    d = pe.copy()
    for f in ["F1", "F2", "F3", "F4", "F5", "F6"]:
        d[f"fmt_{f}"] = (d["format"] == f).astype(int)
    for l in ["L0", "L1", "L2", "L3", "L5"]:
        d[f"len_{l}"] = (d["length"] == l).astype(int)
    d["mech_irrelevant"] = (d["mechanism"] == "irrelevant").astype(int)
    d["mech_redundant"] = (d["mechanism"] == "redundant").astype(int)
    d["pos_front"] = (d["position"] == "front").astype(int)
    d["pos_middle"] = (d["position"] == "middle").astype(int)
    d["is_json"] = d["fmt_F4"]
    d["is_kv"] = d["fmt_F3"]
    d["context_full"] = (d["context"] == "full").astype(int)
    d["model_scale"] = d["model"].map(MODEL_SCALE).fillna(7)
    # model-level features measured from this run set, not assumed
    base = d.groupby("model")["y_ctl"].mean()
    d["model_full_acc"] = d["model"].map(base)
    d["model_notool_acc"] = d["model"].map(base)      # refreshed if a no-tool run exists

    cols = list(INTRINSIC_COLS)
    if conditional:
        # label-free task-tool relation: similarity of the query to the
        # intervened tool's description (uniform scope -> mean over text tools)
        jac = tt.groupby(["task_id", "tool"])["query_tool_jaccard"].mean()
        qlen = tt.groupby("task_id")["query_len"].first()
        d["query_len"] = d["task_id"].map(qlen)
        d["query_tool_jaccard"] = [
            jac.get((t, s), np.nan) if s != "uniform"
            else tt[tt.task_id == t]["query_tool_jaccard"].mean()
            for t, s in zip(d["task_id"], d["scope"])]
        cols += CONDITIONAL_EXTRA

    X = d[cols].astype(float).fillna(0.0)
    leaked = [c for c in X.columns if c in ANNOTATION_COLS]
    assert not leaked, f"annotation feature leaked into deployable matrix: {leaked}"
    y_bin = (d["delta_acc"] > 0).astype(int)
    y_reg = d["delta_acc"].astype(float)
    groups = d["task_id"].to_numpy()
    return X, y_bin, y_reg, groups, d


def grouped_scores(X, y_bin, y_reg, groups, n_splits=5):
    """Grouped CV (group = task) for three model families."""
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2 or y_bin.nunique() < 2:
        return {}, {}
    gkf = GroupKFold(n_splits=n_splits)
    fams = {
        "ridge_logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "gbt": lambda: HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.08, random_state=0),
        "gam_spline": lambda: make_pipeline(
            StandardScaler(), SplineTransformer(n_knots=4, degree=2),
            LogisticRegression(C=0.3, max_iter=3000)),
    }
    regs = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=2.0)),
        "gbt": lambda: HistGradientBoostingRegressor(
            max_depth=3, max_iter=150, learning_rate=0.08, random_state=0),
    }
    auc_out, r2_out = {}, {}
    for name, mk in fams.items():
        preds = np.full(len(y_bin), np.nan)
        for tr, te in gkf.split(X, y_bin, groups):
            if y_bin.iloc[tr].nunique() < 2:
                continue
            m = mk().fit(X.iloc[tr], y_bin.iloc[tr])
            preds[te] = m.predict_proba(X.iloc[te])[:, 1]
        ok = ~np.isnan(preds)
        auc_out[name] = (float(roc_auc_score(y_bin[ok], preds[ok]))
                         if ok.sum() > 10 and y_bin[ok].nunique() > 1 else None)
    for name, mk in regs.items():
        preds = np.full(len(y_reg), np.nan)
        for tr, te in gkf.split(X, y_reg, groups):
            m = mk().fit(X.iloc[tr], y_reg.iloc[tr])
            preds[te] = m.predict(X.iloc[te])
        ok = ~np.isnan(preds)
        ss_res = float(((y_reg[ok] - preds[ok]) ** 2).sum())
        ss_tot = float(((y_reg[ok] - y_reg[ok].mean()) ** 2).sum())
        r2_out[name] = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return auc_out, r2_out


def leave_one_out(X, y_bin, groups, keys, label):
    """Leave-one-<key>-out generalisation (tool / category / model)."""
    out = {}
    for k in sorted(set(keys)):
        tr = np.array([kk != k for kk in keys])
        te = ~tr
        if te.sum() < 8 or y_bin[tr].nunique() < 2 or y_bin[te].nunique() < 2:
            out[str(k)] = None
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        m.fit(X[tr], y_bin[tr])
        p = m.predict_proba(X[te])[:, 1]
        out[str(k)] = float(roc_auc_score(y_bin[te], p))
    return {label: out}


def fit_mask_model(sf: pd.DataFrame):
    """Absolute accuracy as a function of subset composition (subset design)."""
    if len(sf) < 8:
        return None, {}
    feats = ["menu_size", "n_categories", "max_same_category", "desc_tokens",
             "n_image_output", "n_unavailable"]
    X, y = sf[feats].astype(float), sf["answer_acc"].astype(float)
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    # leave-one-subset-out
    errs = []
    for i in range(len(sf)):
        tr = np.ones(len(sf), bool)
        tr[i] = False
        mm = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[tr], y[tr])
        errs.append(float(mm.predict(X.iloc[[i]])[0] - y.iloc[i]))
    ss_res = float(np.sum(np.square(errs)))
    ss_tot = float(np.sum(np.square(y - y.mean())))
    return m, dict(features=feats, loo_r2=(1 - ss_res / ss_tot) if ss_tot else None,
                   loo_mae=float(np.mean(np.abs(errs))), n=len(sf))


def main():
    os.makedirs(MODELS, exist_ok=True)
    pe = pd.read_parquet(f"{FT}/paired_effects.parquet")
    tt = pd.read_parquet(f"{FT}/task_tool_features.parquet")
    sf = (pd.read_parquet(f"{FT}/subset_features.parquet")
          if os.path.exists(f"{FT}/subset_features.parquet") else pd.DataFrame())
    report = {}
    if not len(pe):
        print("[model] no paired effects yet")
        return

    for name, conditional in [("taco_intrinsic", False), ("taco_conditional", True)]:
        X, yb, yr, g, d = design(pe, tt, conditional)
        auc_out, r2_out = grouped_scores(X, yb, yr, g)
        info = dict(n=len(X), n_features=X.shape[1], features=list(X.columns),
                    grouped_auc=auc_out, grouped_r2=r2_out,
                    positive_rate=float(yb.mean()))
        info.update(leave_one_out(X, yb, g, d["scope"].tolist(), "leave_one_tool_out"))
        info.update(leave_one_out(X, yb, g, d["task_category"].tolist(),
                                  "leave_one_category_out"))
        if d["model"].nunique() > 1:
            info.update(leave_one_out(X, yb, g, d["model"].tolist(),
                                      "leave_one_model_out"))
        # final fit on everything, for the search stage
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        reg = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
        if yb.nunique() > 1:
            clf.fit(X, yb)
        reg.fit(X, yr)
        coefs = None
        if yb.nunique() > 1:
            coefs = dict(zip(X.columns,
                             clf.named_steps["logisticregression"].coef_[0].round(4)))
        info["coefficients_logistic"] = coefs
        info["coefficients_linear"] = dict(zip(
            X.columns, reg.named_steps["ridge"].coef_.round(5)))
        pickle.dump(dict(clf=clf if yb.nunique() > 1 else None, reg=reg,
                         columns=list(X.columns), conditional=conditional,
                         model_scale=MODEL_SCALE),
                    open(f"{MODELS}/{name}.pkl", "wb"))
        report[name] = info
        print(f"[model] {name}: grouped AUC {auc_out}  R2 {r2_out}")

    mask_model, mask_info = fit_mask_model(sf) if len(sf) else (None, {})
    if mask_model is not None:
        pickle.dump(dict(reg=mask_model, **mask_info),
                    open(f"{MODELS}/taco_mask.pkl", "wb"))
    report["mask_model"] = mask_info
    json.dump(report, open(f"{MODELS}/model_report.json", "w"), indent=1, default=str)
    print("[model] ->", f"{MODELS}/model_report.json")
    return report


if __name__ == "__main__":
    main()
