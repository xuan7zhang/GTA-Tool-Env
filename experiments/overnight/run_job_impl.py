#!/usr/bin/env python3
"""The only repo-wired part of the orchestrator. dispatch(job, g, outdir) runs a
job and returns a metrics dict (eval jobs return at least {'mean_acc': float}).

Wiring:
  kind == 'analysis' -> python scripts/<script> <args ...>   (writes metrics.json)
  kind == 'probe'    -> python scripts/probes.py --mode ...   (writes metrics.json)
  kind == 'eval'     -> the existing OpenCompass/Lagent pipeline via the shared
                        ToolProxy (proxy_url) + EvalAdapter; reuses the selector
                        keep-lists in results/inject_opt and configs/gta_atomic_env.py.
"""
import glob
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
REPO = Path(os.environ.get("GTA_LAB", HERE.parent.parent))
BIG = Path(os.environ.get("GTA_BIG", "/datasets/omni_pretraining/gta2"))
IO = BIG / "results" / "inject_opt"
OCRUN = BIG / "ocrun"
POISON = json.loads((REPO / "envgen/variants/poison/names.json").read_text())
PSET = set(POISON)

# method -> (selector file, key). ground_plus/declutter live in their own files.
SEL_ATTR = IO / "selectors.json"
SEL_SUBTLE = IO / "selectors_subtle.json"
SEL_GPLUS = IO / "selectors_gplus.json"
SUBTLE_PHI = IO / "subtle_phi.json"


def _read_metrics(outdir):
    p = Path(outdir) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _run_script(script, args, outdir):
    env = dict(os.environ, JOB_OUT=str(outdir))
    log = open(Path(outdir) / "job.log", "w")
    subprocess.run(["python", str(SCRIPTS / script), *map(str, args), "--out", str(outdir)],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return _read_metrics(outdir)


# ---------------------------------------------------------------- eval wiring
def _selector(method, regime):
    """Return (keep_list, kept_poison, phi_path_or_None) for a method+regime."""
    if method == "ground_plus":
        keep = json.loads(SEL_GPLUS.read_text())["ours_grounding_plus"]["keep"]
    elif method == "ground_plus_declutter3":
        # iter-2: prune 3 image tools instead of 5 -> keep 2 of the image-output tools
        base = json.loads(SEL_GPLUS.read_text())["ours_grounding_plus"]["keep"]
        readd = ["DrawBox", "AddText"]  # keep 2 image tools back in
        keep = sorted(set(base) | set(readd))
    else:
        key = {"keep_all": "keep_all", "call_frequency": "call_frequency",
               "answer_echo": "ours_echo", "ground": "ours_grounding",
               "oracle": "oracle"}[method]
        selfile = SEL_SUBTLE if regime == "subtle" else SEL_ATTR
        keep = json.loads(selfile.read_text())[key]["keep"]
    kept_poison = [t for t in keep if t in PSET]
    phi = str(SUBTLE_PHI) if regime == "subtle" else None
    return keep, kept_poison, phi


def _set_proxy(g, keep, phi, log_path, rid):
    import urllib.request
    body = json.dumps({"mode": "passthrough", "mask": keep, "per_tool_modes": {},
                       "phi_toolmeta": phi, "unavailable_tools": ["GoogleSearch", "MathOCR"],
                       "log_path": log_path, "run_meta": {"run_id": rid}}).encode()
    urllib.request.urlopen(urllib.request.Request(
        g["proxy_url"] + "/proxy_config", data=body, method="POST",
        headers={"Content-Type": "application/json"}), timeout=30).read()


def _one_eval(g, method, regime, seed, outdir, rid):
    keep, kept_poison, phi = _selector(method, regime)
    wd = Path(outdir) / f"s{seed}"; wd.mkdir(parents=True, exist_ok=True)
    _set_proxy(g, keep, phi, str(wd / "proxy_calls.jsonl"), rid)
    env = dict(os.environ,
               GTA_MODEL_NAME=g["model"].lower(),
               GTA_LLM_URL=g["llm_url"], GTA_TOOLSERVER=g["proxy_url"],
               GTA_EVAL_MODES="end", GTA_TOOLMETA=str(BIG / "data/gta_dataset/toolmeta.json"),
               GTA_TEMP="0",                                  # greedy (constraint)
               GTA_EXTRA_TOOLS=",".join(kept_poison), GTA_SEED=str(seed))
    log = open(wd / "eval.log", "w")
    subprocess.run(["python", str(REPO / "GTA/opencompass/run.py"),
                    "configs/gta_atomic_env.py", "--max-num-workers", "1", "--debug",
                    "-w", str(wd)], cwd=str(OCRUN), env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
    fs = sorted(glob.glob(str(wd / "*/results/*/gta_bench_end.json")))
    if not fs:
        raise RuntimeError(f"no eval result for {rid} s{seed}")
    return round(json.load(open(fs[-1]))["answer_acc"], 3)


def _run_eval(job, g, outdir):
    method = job.get("method") or job.get("condition")
    regime = job["regime"]
    # seeds to run: matrix jobs carry a single 'seed'; aggregate jobs average a set.
    seeds = job.get("aggregate_seeds") or [job["seed"]]
    accs = {}
    for s in seeds:
        rid = f"{job['id']}_s{s}"
        accs[s] = _one_eval(g, method, regime, s, outdir, rid)
    mean_acc = round(sum(accs.values()) / len(accs), 3)
    (Path(outdir) / "metrics.json").write_text(json.dumps(
        {"mean_acc": mean_acc, "seed_accs": accs, "method": method, "regime": regime}, indent=2))
    return {"mean_acc": mean_acc, "seed_accs": accs}


def _run_octotools(job, g, outdir):
    """Outcome-based greedy validation search (regime-fragile baseline, by design)."""
    env = dict(os.environ, JOB_OUT=str(outdir))
    log = open(Path(outdir) / "job.log", "w")
    subprocess.run(["python", str(SCRIPTS / "octotools_greedy.py"),
                    "--regime", job["regime"], "--seed", str(job["seed"]),
                    "--out", str(outdir)], env=env, stdout=log,
                   stderr=subprocess.STDOUT, check=True)
    return _read_metrics(outdir)


# ---------------------------------------------------------------- dispatch
def dispatch(job, g, outdir):
    kind = job["kind"]
    if kind == "analysis":
        return _run_script(job["script"], job.get("args", []), outdir)
    if kind == "probe":
        return _run_script(job.get("script", "probes.py"), job.get("args", []), outdir)
    if kind == "eval":
        if job.get("script") == "octotools_greedy.py":
            return _run_octotools(job, g, outdir)
        return _run_eval(job, g, outdir)
    raise ValueError(f"unknown kind: {kind}")
