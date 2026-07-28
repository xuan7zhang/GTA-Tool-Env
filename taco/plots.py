"""TACO figures (spec §29). Writes PNGs to results/<root>/plots/.

Style follows the project's data-viz rules: one axis per chart (never a
second y-scale), categorical hues assigned in fixed order and never cycled,
thin marks with recessive grid/axes, a legend whenever two or more series are
present, and paired confidence intervals drawn rather than implied.

Figures
  1 ansacc_vs_length            AnsAcc vs response length, by model
  2 adoption_vs_length          tool-output adoption vs response length
  3 format_by_model_category    format effect by model and tool category
  4 predicted_vs_observed       predicted vs observed marginal utility
  5 likelihood_vs_accuracy      delta logprob vs delta accuracy
  6 heldout_acc_vs_cost         held-out accuracy vs environment cost
  7 search_trajectory           greedy and beam search trajectories
  8 selected_environments       model-specific selected environments
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taco.paths import TACO, FT, PLOTS, SEARCH   # noqa: E402

# Categorical slots in fixed order (light mode), from the validated palette.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
LENGTH_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5"]
FORMAT_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]


def style(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    return ax


def save(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    fig.tight_layout()
    p = f"{PLOTS}/{name}.png"
    fig.savefig(p, dpi=160, facecolor="white")
    plt.close(fig)
    print("[plots]", p)


def short(m):
    return m.replace("qwen2.5-", "Qwen ").replace("-instruct", "").upper()


def fig1_2(pe):
    """AnsAcc and adoption against the length ladder, by model."""
    for kind, col, ylab, name in [
            ("acc", "abs_acc", "AnsAcc on treated tasks (%)", "1_ansacc_vs_length"),
            ("adopt", "adoption", "tool-output adoption rate", "2_adoption_vs_length")]:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        d = pe[pe.attribute == "length"].copy()
        d["abs_acc"] = d["y"] * 100
        d["adoption"] = d["adoption"] * 100 if kind == "adopt" else d["adoption"]
        drew = 0
        for i, (m, g) in enumerate(d.groupby("model")):
            xs, ys, los, his = [], [], [], []
            for lv in LENGTH_ORDER:
                gg = g[g.length == lv]
                if len(gg) < 3:
                    continue
                v = gg[col].dropna()
                if not len(v):
                    continue
                bs = [v.sample(len(v), replace=True, random_state=s).mean()
                      for s in range(400)]
                xs.append(LENGTH_ORDER.index(lv))
                ys.append(v.mean())
                los.append(np.quantile(bs, .025))
                his.append(np.quantile(bs, .975))
            if not xs:
                continue
            c = SERIES[i % len(SERIES)]
            ax.plot(xs, ys, "-o", color=c, linewidth=2, markersize=5, label=short(m))
            ax.fill_between(xs, los, his, color=c, alpha=0.15, linewidth=0)
            drew += 1
        ax.set_xticks(range(len(LENGTH_ORDER)))
        ax.set_xticklabels(LENGTH_ORDER)
        style(ax, f"{'Accuracy' if kind == 'acc' else 'Adoption'} across the length ladder",
              "response-length level (L0 shortest -> L5 padded 2x)", ylab)
        if drew >= 2:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
        save(fig, name)


def fig3(pe):
    """Format effect (paired delta) by model, one panel per tool scope."""
    d = pe[pe.attribute == "format"]
    scopes = [s for s in d.scope.unique() if len(d[d.scope == s]) >= 10]
    if not scopes:
        return
    fig, axes = plt.subplots(1, len(scopes), figsize=(4.2 * len(scopes), 3.8),
                             squeeze=False)
    for j, sc in enumerate(scopes):
        ax = axes[0][j]
        g0 = d[d.scope == sc]
        w = 0.36
        for i, (m, g) in enumerate(g0.groupby("model")):
            xs, ys, err = [], [], [[], []]
            for k, f in enumerate(FORMAT_ORDER[1:]):
                gg = g[g.format == f]
                if len(gg) < 3:
                    continue
                v = gg["delta_acc"] * 100
                bs = [v.sample(len(v), replace=True, random_state=s).mean()
                      for s in range(400)]
                xs.append(k + (i - 0.5) * w)
                ys.append(v.mean())
                err[0].append(max(0, v.mean() - np.quantile(bs, .025)))
                err[1].append(max(0, np.quantile(bs, .975) - v.mean()))
            if xs:
                ax.bar(xs, ys, width=w, color=SERIES[i % len(SERIES)],
                       label=short(m), linewidth=0)
                ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK2,
                            elinewidth=1, capsize=2)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(range(len(FORMAT_ORDER) - 1))
        ax.set_xticklabels(FORMAT_ORDER[1:])
        style(ax, f"Format effect — {sc}", "format (vs F0 native)",
              "paired delta AnsAcc (pts)" if j == 0 else "")
        if j == 0:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(fig, "3_format_by_model_category")


def fig4(pe, model_report):
    """Predicted vs observed marginal utility, per condition cell."""
    g = (pe.groupby(["model", "attribute", "format", "length", "context"])
           .agg(observed=("delta_acc", "mean"), n=("delta_acc", "size"))
           .reset_index())
    g = g[g.n >= 5]
    if not len(g):
        return
    pred_col = None
    coefs = ((model_report or {}).get("taco_intrinsic", {})
             .get("coefficients_linear") or {})
    if coefs:
        def pred(r):
            v = 0.0
            v += coefs.get(f"fmt_{r['format']}", 0.0)
            v += coefs.get(f"len_{r['length']}", 0.0)
            v += coefs.get("context_full", 0.0) * (r["context"] == "full")
            return v
        g["predicted"] = g.apply(pred, axis=1)
        pred_col = "predicted"
    if pred_col is None:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for i, (m, gg) in enumerate(g.groupby("model")):
        ax.scatter(gg[pred_col] * 100, gg["observed"] * 100, s=34,
                   color=SERIES[i % len(SERIES)], edgecolor="white", linewidth=1,
                   label=short(m))
    lim = max(abs(np.r_[g[pred_col].values, g["observed"].values]).max() * 100, 1) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=GRID, linewidth=1.2, zorder=0)
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    style(ax, "Predicted vs observed marginal utility",
          "TACO-predicted delta AnsAcc (pts)", "observed paired delta (pts)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(fig, "4_predicted_vs_observed")


def fig5(pe):
    """Likelihood delta vs accuracy delta — the RQ3 picture."""
    d = pe.dropna(subset=["delta_ll"])
    if len(d) < 10:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for i, (m, g) in enumerate(d.groupby("model")):
        jitter = (np.random.default_rng(0).random(len(g)) - 0.5) * 0.06
        ax.scatter(g["delta_ll"], g["delta_acc"] + jitter, s=22, alpha=0.7,
                   color=SERIES[i % len(SERIES)], edgecolor="white", linewidth=0.6,
                   label=short(m))
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    r = d[["delta_ll", "delta_acc"]].corr().iloc[0, 1]
    style(ax, f"Likelihood delta vs accuracy delta (r = {r:+.3f})",
          "delta mean logprob (intervention - control)",
          "delta AnsAcc (paired, jittered)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(fig, "5_likelihood_vs_accuracy")


def fig6(runs):
    """Held-out accuracy against environment cost (tokens)."""
    d = runs[runs.kind == "heldout"]
    if not len(d):
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, (m, g) in enumerate(d.groupby("model")):
        ax.scatter(g["visible_tool_tokens"], g["answer_acc"], s=40,
                   color=SERIES[i % len(SERIES)], edgecolor="white", linewidth=1,
                   label=short(m))
        # direct-label only the extremes, never every point
        for _, r in g.iterrows():
            if r["answer_acc"] in (g["answer_acc"].max(), g["answer_acc"].min()):
                ax.annotate(r["run_id"].replace("ho_", "").rsplit("_", 1)[0],
                            (r["visible_tool_tokens"], r["answer_acc"]),
                            fontsize=7, color=INK2,
                            xytext=(4, 3), textcoords="offset points")
    style(ax, "Held-out accuracy vs environment cost",
          "visible tool-output tokens (whole held-out set)", "held-out AnsAcc (%)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    save(fig, "6_heldout_acc_vs_cost")


def fig7(search):
    """Search trajectories: greedy forward/backward and beam."""
    if not search:
        return
    models = list(search)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.8),
                             squeeze=False)
    for j, m in enumerate(models):
        ax = axes[0][j]
        r = search[m]
        gf = r.get("greedy_forward") or []
        ax.plot([x["step"] for x in gf], [x["utility"] for x in gf], "-o",
                color=SERIES[0], linewidth=2, markersize=4, label="greedy forward")
        gb = r.get("greedy_backward") or []
        ax.plot([x["step"] for x in gb], [x["utility"] for x in gb], "-s",
                color=SERIES[1], linewidth=2, markersize=4, label="greedy backward")
        for bi, (b, traj) in enumerate((r.get("beam") or {}).items()):
            ax.plot([x["k"] for x in traj], [x["best_utility"] for x in traj], "--",
                    color=SERIES[2 + bi % 3], linewidth=1.5, label=f"beam b={b}")
        style(ax, f"Search trajectory — {short(m)}", "menu size |M|",
              "predicted calibration utility" if j == 0 else "")
        ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
    save(fig, "7_search_trajectory")


def fig8(frozen):
    """Model-specific selected environments, side by side."""
    if not frozen:
        return
    envs = frozen.get("environments", {})
    models = list(envs)
    if not models:
        return
    tools = sorted({t for m in models for t in envs[m]["keep_all"]["tools"]})
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(tools) + 1.6))
    for i, m in enumerate(models):
        sel = set(envs[m]["taco_intrinsic"]["tools"])
        ys = [k for k, t in enumerate(tools) if t in sel]
        ax.scatter([i] * len(ys), ys, s=90, marker="s",
                   color=SERIES[i % len(SERIES)], label=short(m))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([short(m) for m in models])
    ax.set_yticks(range(len(tools)))
    ax.set_yticklabels(tools, fontsize=8)
    ax.set_xlim(-0.6, len(models) - 0.4)
    style(ax, "Selected global environment per model (TACO-Intrinsic mask)", "", "")
    save(fig, "8_selected_environments")


def main():
    pe = pd.read_parquet(f"{FT}/paired_effects.parquet")
    runs = pd.read_parquet(f"{FT}/runs.parquet")
    mr = None
    if os.path.exists(f"{TACO}/models/model_report.json"):
        mr = json.load(open(f"{TACO}/models/model_report.json"))
    search = (json.load(open(f"{SEARCH}/search_results.json"))
              if os.path.exists(f"{SEARCH}/search_results.json") else None)
    frozen_p = f"{TACO}/frozen_environments/frozen_envs.json"
    frozen = json.load(open(frozen_p)) if os.path.exists(frozen_p) else None
    if len(pe):
        fig1_2(pe)
        fig3(pe)
        fig4(pe, mr)
        fig5(pe)
    fig6(runs)
    fig7(search)
    fig8(frozen)
    print("[plots] done ->", PLOTS)


if __name__ == "__main__":
    main()
