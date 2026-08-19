"""Emit the TGB-v4 result tables as LaTeX, read from the result files.

    python -m tgb.make_tex > tables.tex

Written to read the JSON rather than transcribe numbers by hand: several of
these tables were rebuilt after the scorer fix, and a hand-copied digit from
the pre-fix run would be indistinguishable from a real one.
"""
import json
import os
import statistics as st

T4 = "/datasets/omni_pretraining/gta2/results/taco/tgb4"
MODELS = [("7b", "Qwen2.5-7B"), ("llama8b", "Llama-3.1-8B"),
          ("mistral7b", "Mistral-7B-v0.3"), ("14b", "Qwen2.5-14B")]
CONDS = ["none", "useful", "no_downstream", "no_upstream",
         "wrong", "corrupt", "union", "full"]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def scored(m):
    R = {}
    with open(f"{T4}/tgb_scored_{m}.jsonl") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            R[r["id"]] = r
    return R


def fmt(v, d=3, bold=False):
    if v is None:
        return "--"
    s = f"{v:.{d}f}"
    return r"\textbf{" + s + "}" if bold else s


def row(label, vals, bold=False, d=3):
    return label + " & " + " & ".join(fmt(v, d, bold) for v in vals) + r" \\"


def table_main(out):
    r = {}
    for m, _ in MODELS:
        au = jload(f"{T4}/tgb_algo1_{m}_split_union.json")["test_acc"]
        ae = jload(f"{T4}/tgb_algo1_{m}_split_empty.json")
        G = jload(f"{T4}/tgb_greedy_{m}_backward.json")["test_acc"]
        Lg = jload(f"{T4}/tgb_loo_{m}.json")["selected"]
        Ls = jload(f"{T4}/tgb_loo_{m}_self_S.json")["selected"]
        Gr = jload(f"{T4}/tgb_granularity_{m}_self_S.json")["rows"]
        r[m] = dict(full=G["full"], union=G["union"],
                    ae=ae["test_acc"]["algo1"] if ae else None,
                    au=au["algo1"], bwd=G["greedy"],
                    fam=Gr["family"]["acc"], task=Ls["test"], gold=Lg["test"])
    col = lambda k: [r[m][k] for m, _ in MODELS]
    out += [
        r"% ---------------------------------------------------- Table 1",
        r"\begin{table}[t]\centering\small",
        r"\caption{Toolset selection on TGB-v4 (test split, $n{=}3{,}600$). All",
        r"methods share the same 15-tool action space and the same test split,",
        r"verified id-by-id. Algorithm~1 is forward single-pass selection;",
        r"backward elimination is the multi-round variant. LOO is leave-one-out",
        r"inside the joint coalition. The gold row reads one label per test task",
        r"and is therefore an upper bound, not a deployable method.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Method & " + " & ".join(n for _, n in MODELS) + r" \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{No selection}} \\",
        row(r"\quad Full menu (15 tools)", col("full")),
        row(r"\quad Union of ground-truth chains (7)", col("union")),
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Search-based selection (400 labels)}} \\",
        row(r"\quad Algorithm~1, $\mathcal{D}_{\text{base}}=\emptyset$", col("ae")),
        row(r"\quad Algorithm~1, $\mathcal{D}_{\text{base}}=$ union", col("au")),
        row(r"\quad Backward elimination", col("bwd")),
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Likelihood selection (400 labels for $\tau$)}} \\",
        row(r"\quad LOO, per-family", col("fam"), bold=True),
        row(r"\quad LOO, per-task", col("task")),
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Upper bound (one label per test task)}} \\",
        row(r"\quad LOO, per-task, gold", col("gold")),
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]


def table_headroom(out):
    rows = {}
    for m, _ in MODELS:
        R = scored(m)
        ids = [i for i in R if R[i]["family"] != "no_tool"]
        ca = {c: st.mean([R[i]["conds"][c]["acc"] for i in ids]) for c in CONDS}
        bc = max(ca, key=ca.get)
        orc = st.mean([max(R[i]["conds"][c]["acc"] for c in CONDS) for i in ids])
        sel = st.mean([R[i]["conds"][max(
            CONDS, key=lambda c: R[i]["conds"][c]["L"]
            if R[i]["conds"][c].get("L") is not None else -99)]["acc"]
            for i in ids])
        w = t = 0
        for i in ids:
            u = R[i]["conds"]["useful"].get("L")
            if u is None:
                continue
            for b in ("none", "wrong", "corrupt", "no_upstream"):
                x = R[i]["conds"].get(b, {}).get("L")
                if x is None:
                    continue
                w += (u > x) + 0.5 * (u == x)
                t += 1
        rows[m] = dict(const=ca[bc], name=bc, dl=sel, orc=orc,
                       hr=orc - ca[bc], auc=w / t)
    col = lambda k: [rows[m][k] for m, _ in MODELS]
    out += [
        r"% ---------------------------------------------------- Table 2",
        r"\begin{table}[t]\centering\small",
        r"\caption{Is there anything for a per-task selector to win? Oracle",
        r"routing against the best \emph{constant} policy over the eight",
        r"executed conditions ($n{=}3{,}600$, controls excluded). The constant",
        r"baseline is chosen post hoc on the same data and is therefore a",
        r"generous opponent. On the earlier visual benchmark this headroom was",
        r"negative, and no selector could pay.}",
        r"\label{tab:headroom}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        " & " + " & ".join(n for _, n in MODELS) + r" \\",
        r"\midrule",
        row("Best constant policy", col("const")),
        row(r"$\Delta L$ routing", col("dl")),
        row("Oracle routing", col("orc")),
        r"\midrule",
        row("Headroom", col("hr")),
        row("Within-task paired AUROC", col("auc")),
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]


def table_gran(out):
    r = {m: jload(f"{T4}/tgb_granularity_{m}_self_S.json") for m, _ in MODELS}
    col = lambda k: [r[m]["rows"][k]["acc"] for m, _ in MODELS]
    delta = [r[m]["rows"]["task"]["acc"] - r[m]["rows"]["family"]["acc"]
             for m, _ in MODELS]
    agree = [r[m]["agree_with_family_mode"] for m, _ in MODELS]
    out += [
        r"% ---------------------------------------------------- Table 3",
        r"\begin{table}[t]\centering\small",
        r"\caption{Granularity ablation. The same label-free selector output,",
        r"deployed at three granularities: one global mask, one mask per task",
        r"family, and the per-task mask itself. Modes are taken on the training",
        r"split only. Per-task granularity is not merely unnecessary here --- it",
        r"is harmful on every model, so the within-family variation the selector",
        r"produces is noise.}",
        r"\label{tab:gran}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Deployed granularity & " + " & ".join(n for _, n in MODELS) + r" \\",
        r"\midrule",
        row("One global mask", col("global")),
        row("One mask per family", col("family"), bold=True),
        row("One mask per task", col("task")),
        r"\midrule",
        row(r"Per-task $-$ per-family", delta),
        row("Per-task mask equals its family mode", agree, d=2),
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]


def table_resolution(out):
    acc_n, l_n, flat = [], [], []
    for m, _ in MODELS:
        R = scored(m)
        ids = [i for i in R if R[i]["family"] != "no_tool"]
        acc_n.append(st.mean([len({R[i]["conds"][c]["acc"] for c in CONDS})
                              for i in ids]))
        l_n.append(st.mean([len({round(R[i]["conds"][c]["L"], 4) for c in CONDS
                                 if R[i]["conds"][c].get("L") is not None})
                            for i in ids]))
        flat.append(st.mean([len({R[i]["conds"][c]["acc"]
                                  for c in CONDS}) == 1 for i in ids]))
    out += [
        r"% ---------------------------------------------------- Table 4",
        r"\begin{table}[t]\centering\small",
        r"\caption{Why a search over accuracy cannot operate on one task.",
        r"Distinct values each objective takes across the eight candidate",
        r"coalitions of a \emph{single} task, averaged over tasks. Accuracy is",
        r"one bit and resolves fewer than two of eight; likelihood resolves",
        r"nearly all eight. A search whose objective is accuracy must average",
        r"over hundreds of tasks before it has a usable signal, which is why it",
        r"can only emit one global mask.}",
        r"\label{tab:resolution}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Distinct values per task (of 8) & " + " & ".join(n for _, n in MODELS) + r" \\",
        r"\midrule",
        row("Accuracy", acc_n, d=2),
        row("Log-likelihood", l_n, d=2, bold=True),
        row("Tasks where all 8 tie on accuracy", flat, d=3),
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]


def main():
    out = []
    table_main(out)
    table_headroom(out)
    table_gran(out)
    table_resolution(out)
    print("\n".join(out))


if __name__ == "__main__":
    main()
