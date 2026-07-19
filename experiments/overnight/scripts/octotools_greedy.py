#!/usr/bin/env python3
"""OctoTools-style greedy toolset search (decisive-table baseline).

Original OctoTools reduces the 2^n subset search to O(n) by scoring each tool's
MARGINAL contribution instead of searching combinations. Three stages:

  Stage 1 (marginal contribution, O(n) val evals):
     acc_full = val acc of the full pool. For each tool t, its marginal contribution
     is measured leave-one-out:  marginal(t) = acc_full - acc(full minus {t}).
     (An empty base breaks the ReAct harness -- the agent needs >=1 tool -- and a
     forward "single tool from empty" base is degenerate for chain-tool GTA tasks
     where one tool rarely solves a query. LOO from the full pool keeps the exact
     marginal-contribution + threshold semantics, is non-degenerate, and is what
     OctoTools' greedy pruning reduces to here.)
  Stage 2 (threshold prune):  keep {t : marginal(t) > 0} (removing it hurts => useful);
     drop {t : marginal(t) <= 0} (removing it helps or is neutral => poison / unused).
  Stage 3 (subset-vs-full safety valve):  eval the selected subset vs the full
     pool on val; if the subset is worse, revert to full (guards against greedy
     over-pruning).

Signal = validation accuracy => OUTCOME-based, expected regime-fragile (this is the
control; do NOT optimize it). In the injection setting the pool = 14 real + 8 poison
= 22 tools; poison tools lower val accuracy so their marginal is <=0 and Stage 2
should drop them -- an outcome-based (label-hungry, O(n) full evals) way to identify
unreliable tools, in contrast to the label-free per-tool behavioral probe.

Greedy decoding (constraint). Records every tool's marginal + selection decision.
  octotools_greedy.py --regime <attractive|subtle> --seed <s> --out <dir> [--valn 30]
"""
import argparse
import glob
import json
import os
import subprocess
import urllib.request
from pathlib import Path

BIG = Path(os.environ.get("GTA_BIG", "/datasets/omni_pretraining/gta2"))
REPO = Path(os.environ.get("GTA_LAB", "/project/6101776/xzhan576/gta2-envlab"))
OCRUN = BIG / "ocrun"
PROXY = os.environ.get("GTA_PROXY", "http://127.0.0.1:16281")
LLM = os.environ.get("GTA_LLM_URL", "http://127.0.0.1:12580/v1/chat/completions")
FULL = BIG / "data/gta_dataset/toolmeta.json"
IO = BIG / "results/inject_opt"
REAL = list(json.loads(FULL.read_text()).keys())                       # 14
POISON = json.loads((REPO / "envgen/variants/poison/names.json").read_text())  # 8
PSET = set(POISON)
POOL = REAL + POISON                                                    # 22 candidates


def set_proxy(keep, phi, log, rid):
    body = json.dumps({"mode": "passthrough", "mask": keep, "per_tool_modes": {},
                       "phi_toolmeta": phi, "unavailable_tools": ["GoogleSearch", "MathOCR"],
                       "log_path": log, "run_meta": {"run_id": rid}}).encode()
    urllib.request.urlopen(urllib.request.Request(PROXY + "/proxy_config", data=body,
                           method="POST", headers={"Content-Type": "application/json"}),
                           timeout=30).read()


def eval_set(S, valn, phi, seed, wd, rid):
    """Val (or full, valn=0) accuracy with the agent menu forced to exactly set S."""
    S = list(S)
    os.system(f"rm -rf {wd}"); Path(wd).mkdir(parents=True, exist_ok=True)
    set_proxy(S, phi, f"{wd}/proxy.jsonl", rid)      # mask = S (openapi filtered to S)
    env = dict(os.environ, GTA_MODEL_NAME="qwen2.5-32b-instruct", GTA_LLM_URL=LLM,
               GTA_TOOLSERVER=PROXY, GTA_EVAL_MODES="end", GTA_TOOLMETA=str(FULL),
               GTA_TEMP="0", GTA_EXTRA_TOOLS=",".join(S), GTA_SEED=str(seed))  # force all of S onto menu
    if valn:
        env["GTA_TASK_SUBSET"] = str(valn)
    else:
        env.pop("GTA_TASK_SUBSET", None)
    subprocess.run(["python", str(REPO / "GTA/opencompass/run.py"), "configs/gta_atomic_env.py",
                    "--max-num-workers", "1", "--debug", "-w", wd], cwd=str(OCRUN), env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fs = sorted(glob.glob(f"{wd}/*/results/*/gta_bench_end.json"))
    return round(json.load(open(fs[-1]))["answer_acc"], 3) if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--valn", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    phi = str(IO / "subtle_phi.json") if a.regime == "subtle" else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.regime}_s{a.seed}"

    # ---- Stage 1: leave-one-out marginal contribution of each candidate ----
    acc_full = eval_set(POOL, a.valn, phi, a.seed, str(out / "full"), f"oct_{tag}_full")
    marginal = {}
    for i, t in enumerate(POOL):
        acc_wo = eval_set([x for x in POOL if x != t], a.valn, phi, a.seed,
                          str(out / f"cand_{i}"), f"oct_{tag}_wo_{t}")
        marginal[t] = None if (acc_wo is None or acc_full is None) else round(acc_full - acc_wo, 3)

    # ---- Stage 2: keep marginal > 0 (removing the tool hurts => keep it) ----
    selected = [t for t in POOL if (marginal[t] or 0) > 0]

    # ---- Stage 3: subset-vs-full safety valve ----
    acc_sel = eval_set(selected, a.valn, phi, a.seed, str(out / "sel"), f"oct_{tag}_sel") if selected else -1.0
    acc_base = acc_full  # kept for the record schema
    reverted = acc_sel < acc_full
    final_set = POOL if reverted else selected

    # ---- final: full-229 test accuracy of the chosen set ----
    final_acc = eval_set(final_set, 0, phi, a.seed, str(out / "final"), f"oct_{tag}_final")

    res = {"mean_acc": final_acc, "final_full229": final_acc,
           "regime": a.regime, "seed": a.seed, "valn": a.valn,
           "acc_base": acc_base, "marginal": marginal,
           "selected_stage2": sorted(selected),
           "stage3": {"acc_selected": acc_sel, "acc_full": acc_full, "reverted_to_full": reverted},
           "final_set": sorted(final_set),
           "kept_poison": [t for t in final_set if t in PSET],
           "pruned_genuine": [t for t in REAL if t not in final_set]}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("mean_acc", "selected_stage2", "kept_poison",
                                          "pruned_genuine", "stage3")}, indent=2))


if __name__ == "__main__":
    main()
