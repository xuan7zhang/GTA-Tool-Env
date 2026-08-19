"""tau-sweep figure for the TGB ablation: held-out accuracy and kept-tool
count vs threshold tau, per model, full method (fixed y_S) and self_S variant.

  python -m tgb.plot_tau_sweep --out <dir>
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "/datasets/omni_pretraining/gta2/results/taco/tgb4"
TAGS = [("7b", "Qwen2.5-7B"), ("14b", "Qwen2.5-14B"),
        ("llama8b", "Llama-3.1-8B"), ("mistral7b", "Mistral-7B")]


def rows(path):
    d = json.load(open(path))
    sw = {float(k): v for k, v in d["sweep"].items() if k != "S"}
    xs = sorted(sw)
    return xs, [sw[x]["test"] for x in xs], [sw[x]["tools"] for x in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    colors = plt.get_cmap("tab10")
    for i, (tag, label) in enumerate(TAGS):
        p = f"{D}/tgb_loo_{tag}.json"
        ps = f"{D}/tgb_loo_{tag}_self_S.json"
        if os.path.exists(p):
            xs, acc, tools = rows(p)
            axes[0].plot(xs, acc, "-o", ms=3, color=colors(i), label=label)
            axes[1].plot(xs, tools, "-o", ms=3, color=colors(i))
        if os.path.exists(ps):
            xs, acc, tools = rows(ps)
            axes[0].plot(xs, acc, "--s", ms=3, color=colors(i), alpha=0.55)
            axes[1].plot(xs, tools, "--s", ms=3, color=colors(i), alpha=0.55)
    axes[0].set_xlabel(r"threshold $\tau$")
    axes[0].set_ylabel("held-out accuracy")
    axes[1].set_xlabel(r"threshold $\tau$")
    axes[1].set_ylabel("tools kept")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].plot([], [], "k-o", ms=3, label="fixed $y_S$ (gold)")
    axes[0].plot([], [], "k--s", ms=3, alpha=0.55, label="self $y_S$")
    axes[0].legend(fontsize=7.5, frameon=False)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(a.out, "tgb_tau_sweep.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
