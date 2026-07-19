#!/usr/bin/env python3
"""OctoTools-style greedy validation-accuracy toolset search (Table-1 baseline).
Backward greedy elimination over the 22-tool poison pool: each step evaluates
validation accuracy with each remaining tool removed (the tool's marginal score),
removes the best, stops when nothing improves. Signal = validation accuracy =>
OUTCOME-based and expected to be regime-fragile. This is the control; do not
"optimize" it. Greedy decoding (constraint). Records per-tool marginal score and
the selection decision at every step. Writes metrics.json with the full-229 acc
of the selected set.

  octotools_greedy.py --regime <attractive|subtle> --seed <s> --out <dir> [--valn 25]
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
REAL = list(json.loads(FULL.read_text()).keys())
POISON = json.loads((REPO / "envgen/variants/poison/names.json").read_text())
PSET = set(POISON)


def set_proxy(keep, phi, log, rid):
    body = json.dumps({"mode": "passthrough", "mask": keep, "per_tool_modes": {},
                       "phi_toolmeta": phi, "unavailable_tools": ["GoogleSearch", "MathOCR"],
                       "log_path": log, "run_meta": {"run_id": rid}}).encode()
    urllib.request.urlopen(urllib.request.Request(PROXY + "/proxy_config", data=body,
                           method="POST", headers={"Content-Type": "application/json"}),
                           timeout=30).read()


def evaluate(keep, valn, phi, seed, wd, rid):
    extra = ",".join(t for t in keep if t in PSET)
    os.system(f"rm -rf {wd}"); Path(wd).mkdir(parents=True, exist_ok=True)
    set_proxy(keep, phi, f"{wd}/proxy.jsonl", rid)
    env = dict(os.environ, GTA_MODEL_NAME="qwen2.5-32b-instruct", GTA_LLM_URL=LLM,
               GTA_TOOLSERVER=PROXY, GTA_EVAL_MODES="end", GTA_TOOLMETA=str(FULL),
               GTA_TEMP="0", GTA_EXTRA_TOOLS=extra, GTA_SEED=str(seed))
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
    ap.add_argument("--valn", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    phi = str(IO / "subtle_phi.json") if a.regime == "subtle" else None
    out = Path(a.out)
    keep = REAL + POISON
    tag = f"{a.regime}_s{a.seed}"
    cur = evaluate(keep, a.valn, phi, a.seed, str(out / "base"), f"gs_{tag}_base")
    history = [{"step": 0, "removed": None, "val_acc": cur, "keep_size": len(keep)}]

    while True:
        marginal = {}
        for i, t in enumerate(keep):
            v = evaluate([x for x in keep if x != t], a.valn, phi, a.seed,
                         str(out / f"cand_{len(keep)}_{i}"), f"gs_{tag}_{len(keep)}_{i}")
            marginal[t] = v            # marginal score = val acc with t removed
        best = max((t for t in keep if marginal[t] is not None),
                   key=lambda t: marginal[t], default=None)
        improve = best is not None and marginal[best] > cur
        history.append({"step": len(history), "candidates": marginal,
                        "chosen": best if improve else None,
                        "chosen_val": marginal.get(best) if improve else None,
                        "prev_val": cur})
        if not improve:
            break
        keep = [x for x in keep if x != best]; cur = marginal[best]

    final = evaluate(keep, 0, phi, a.seed, str(out / "final"), f"gs_{tag}_final")
    kept_poison = [t for t in keep if t in PSET]
    pruned_genuine = [t for t in REAL if t not in keep]
    res = {"mean_acc": final, "final_full229": final, "keep": sorted(keep),
           "kept_poison": kept_poison, "pruned_genuine": pruned_genuine,
           "regime": a.regime, "seed": a.seed, "history": history}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("mean_acc", "kept_poison", "pruned_genuine")}, indent=2))


if __name__ == "__main__":
    main()
