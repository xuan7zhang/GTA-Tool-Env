#!/usr/bin/env python3
"""Unattended overnight orchestrator for the GTA-Atomic experiment queue.

Design goals (see run_config.yaml):
  * one vLLM server for the whole night (32B loads slowly): CPU/probe jobs run
    first, then vLLM is brought up ONCE for the GPU jobs, then shut down.
  * a single failing job never stops the queue: retry up to `retries`, then mark
    it failed and continue.
  * resumable: completed/failed/skipped jobs are recorded in state.json; --resume
    skips them, so a 3am crash costs one job, not the night.
  * gates/flags: a job can set a flag from its metrics (gate); later jobs with
    requires_flag run only if the flag is set.
  * heartbeat: current job + status written to logs/heartbeat.txt every N sec.
  * --dry-run validates the plan WITHOUT touching the GPU.

Only run_job() needs wiring to the repo (see below); everything else is generic.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- gate
def evaluate_gate(metrics: dict, spec: dict) -> bool:
    """Pure, unit-testable gate. spec = {metric, op, value}. No eval()."""
    if not spec:
        return False
    val = metrics.get(spec["metric"])
    if val is None:
        return False
    op, thr = spec["op"], float(spec["value"])
    return {">": val > thr, ">=": val >= thr,
            "<": val < thr, "<=": val <= thr,
            "==": val == thr}[op]


# ----------------------------------------------------------------------------- jobs
def expand_jobs(raw_jobs: list) -> list:
    """Expand matrix jobs into concrete instances; keep declaration order."""
    out = []
    for j in raw_jobs:
        j = dict(j)
        matrix = j.pop("matrix", None)
        if not matrix:
            j.setdefault("id", j["name"])
            out.append(j)
            continue
        keys = list(matrix.keys())
        def rec(i, acc):
            if i == len(keys):
                inst = dict(j)
                inst.update(acc)
                inst["id"] = j["name"] + "." + ".".join(str(acc[k]) for k in keys)
                out.append(inst)
                return
            for v in matrix[keys[i]]:
                rec(i + 1, {**acc, keys[i]: v})
        rec(0, {})
    return out


def is_gpu(job: dict) -> bool:
    return job.get("kind") == "eval" or job.get("gpu") is True


def plan(jobs: list) -> list:
    """Execution order: all non-GPU (analysis/probe) first, then GPU; stable."""
    non_gpu = [j for j in jobs if not is_gpu(j)]
    gpu = [j for j in jobs if is_gpu(j)]
    return non_gpu + gpu


def validate_wiring(jobs: list) -> list:
    """Return a list of problems (empty == self-consistent).
    Every requires_flag must be produced by some job's sets_flag."""
    produced = {j["sets_flag"]["name"] for j in jobs if j.get("sets_flag")}
    problems = []
    for j in jobs:
        rf = j.get("requires_flag")
        if rf and rf not in produced:
            problems.append(f"job '{j['id']}' requires_flag '{rf}' but no gate produces it")
    # ordering sanity: no GPU job before a non-GPU job in the plan
    ordered = plan(jobs)
    seen_gpu = False
    for j in ordered:
        if is_gpu(j):
            seen_gpu = True
        elif seen_gpu:
            problems.append(f"non-GPU job '{j['id']}' scheduled after a GPU job")
    return problems


# ----------------------------------------------------------------------------- state
class State:
    def __init__(self, path: Path):
        self.path = path
        self.d = {"completed": {}, "failed": {}, "skipped": {}, "flags": {}}
        if path.exists():
            self.d.update(json.loads(path.read_text()))

    def save(self):
        self.path.write_text(json.dumps(self.d, indent=2))

    def done(self, jid):  # completed OR failed OR skipped -> won't re-run on resume
        return jid in self.d["completed"] or jid in self.d["failed"] or jid in self.d["skipped"]

    def mark(self, bucket, jid, info):
        self.d[bucket][jid] = info
        self.save()

    def set_flag(self, name, value=True):
        self.d["flags"][name] = value
        self.save()

    def flag(self, name):
        return bool(self.d["flags"].get(name))


# ----------------------------------------------------------------------------- vLLM
class VLLM:
    """Single long-lived 32B server. fp16, greedy, no quantization. OOM degrade:
    first drop max_model_len -> 12288, then move caption/imggen tools to CPU (the
    tool server, not the LLM); NEVER swap to a quantized model."""
    def __init__(self, g, logdir):
        self.g = g
        self.logdir = logdir
        self.proc = None

    def _cmd(self, max_model_len):
        v = self.g["vllm"]
        return [
            "vllm", "serve", self.g["model"],
            "--dtype", self.g.get("dtype", "float16"),   # fp16, NO quantization
            "--gpu-memory-utilization", str(v["gpu_memory_utilization"]),
            "--max-model-len", str(max_model_len),
            "--port", str(self.g.get("llm_port", 12580)),
        ]

    def start(self):
        v = self.g["vllm"]
        mml = v["max_model_len"]
        for attempt, mml in enumerate([mml] + [d.get("max_model_len")
                                               for d in v.get("oom_degrade", [])
                                               if d.get("max_model_len")]):
            log = open(self.logdir / f"vllm_start_{attempt}.log", "w")
            self.proc = subprocess.Popen(self._cmd(mml), stdout=log, stderr=subprocess.STDOUT)
            if self._health(timeout=v.get("startup_timeout", 900)):
                return True
            self.stop()  # failed / OOM -> degrade and retry
        return False

    def _health(self, timeout):
        import urllib.request
        url = f"http://127.0.0.1:{self.g.get('llm_port', 12580)}/v1/models"
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                urllib.request.urlopen(url, timeout=5).read()
                return True
            except Exception:
                time.sleep(10)
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


# ----------------------------------------------------------------------------- heartbeat
class Heartbeat(threading.Thread):
    def __init__(self, path, interval):
        super().__init__(daemon=True)
        self.path, self.interval = path, interval
        self.status = "init"
        self._stop = threading.Event()

    def set(self, s):
        self.status = s

    def run(self):
        while not self._stop.wait(self.interval):
            self.path.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {self.status}\n")

    def stop(self):
        self._stop.set()


# ----------------------------------------------------------------------------- run_job (the ONLY repo-wired part)
def config_hash(job, g):
    payload = {k: job.get(k) for k in ("kind", "method", "condition", "regime", "seed", "script")}
    payload["decoding"] = g["decoding"]
    payload["model"] = g["model"]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def run_job(job, g, outdir):
    """Dispatch by kind. Returns metrics dict (eval jobs must return mean_acc).
    Wires to the existing repo scripts / eval pipeline — reuses EvalAdapter/ToolProxy,
    does not start a parallel harness. See run_job_impl.py for the concrete calls."""
    from run_job_impl import dispatch
    return dispatch(job, g, outdir)


# ----------------------------------------------------------------------------- main loop
def run_queue(cfg_path, resume, dry_run):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    g = cfg["global"]
    jobs = expand_jobs(cfg["jobs"])
    ordered = plan(jobs)
    problems = validate_wiring(jobs)

    logdir = HERE / "logs"; logdir.mkdir(exist_ok=True)
    outdir = Path(g.get("out_root", HERE / "runs")); outdir.mkdir(parents=True, exist_ok=True)
    state = State(HERE / "state.json")

    if dry_run:
        print("=== DRY RUN (no GPU touched) ===")
        print(f"\nExecution plan ({len(ordered)} jobs, non-GPU first):")
        seen_gpu = False
        for j in ordered:
            tag = "GPU " if is_gpu(j) else "cpu "
            if is_gpu(j) and not seen_gpu:
                print("  ---- vLLM start (once) ----"); seen_gpu = True
            extra = []
            if resume and state.done(j["id"]):
                extra.append("SKIP (done, --resume)")
            if j.get("requires_flag"):
                will = "SKIP (flag unset at plan time)" if not state.flag(j["requires_flag"]) else "run"
                extra.append(f"requires_flag={j['requires_flag']} -> {will}")
            if j.get("sets_flag"):
                sf = j["sets_flag"]
                extra.append(f"sets_flag {sf['name']} when {sf['metric']}{sf['op']}{sf['value']}")
            print(f"  [{tag}] {j['id']}" + (f"   ({'; '.join(extra)})" if extra else ""))
        print("\nWiring check:", "OK (self-consistent)" if not problems else "PROBLEMS:")
        for p in problems:
            print("  !", p)
        return 0 if not problems else 1

    if problems:
        print("Refusing to run: wiring problems:", *problems, sep="\n  ")
        return 1

    hb = Heartbeat(logdir / "heartbeat.txt", g.get("heartbeat_sec", 30)); hb.start()
    vllm = None
    try:
        # phase 1: non-GPU jobs
        for j in ordered:
            if is_gpu(j):
                continue
            _run_one(j, g, state, outdir, hb, vllm=None)

        # phase 2: bring vLLM up ONCE, then GPU jobs
        gpu_jobs = [j for j in ordered if is_gpu(j)]
        if any(not state.done(j["id"]) or j.get("requires_flag") for j in gpu_jobs):
            hb.set("starting vLLM (32B, fp16, greedy)")
            vllm = VLLM(g, logdir)
            if not vllm.start():
                hb.set("vLLM FAILED to start; GPU jobs aborted")
                for j in gpu_jobs:
                    if not state.done(j["id"]):
                        state.mark("failed", j["id"], {"error": "vllm_start_failed"})
                return 2
            for j in gpu_jobs:
                _run_one(j, g, state, outdir, hb, vllm=vllm)
    finally:
        if vllm:
            hb.set("shutting down vLLM"); vllm.stop()
        hb.set("done"); hb.stop()
        time.sleep(1)
    return 0


def _run_one(j, g, state, outdir, hb, vllm):
    jid = j["id"]
    if state.done(jid):
        return
    # gate/flag dependency
    rf = j.get("requires_flag")
    if rf and not state.flag(rf):
        hb.set(f"skip {jid} (requires_flag {rf} unset)")
        state.mark("skipped", jid, {"reason": f"requires_flag {rf} unset"})
        return
    job_out = outdir / jid; job_out.mkdir(parents=True, exist_ok=True)
    retries = int(g.get("retries", 2))
    for attempt in range(retries + 1):
        hb.set(f"run {jid} (attempt {attempt + 1}/{retries + 1})")
        try:
            metrics = run_job(j, g, job_out)
            state.mark("completed", jid, {"metrics": metrics,
                                          "config_hash": config_hash(j, g)})
            # gate: set flag from metrics
            sf = j.get("sets_flag")
            if sf and evaluate_gate(metrics, sf):
                state.set_flag(sf["name"], True)
                hb.set(f"{jid} set flag {sf['name']}")
            return
        except Exception as e:
            (job_out / f"error_attempt{attempt}.txt").write_text(repr(e))
            if attempt == retries:
                state.mark("failed", jid, {"error": repr(e)})
                hb.set(f"{jid} FAILED after {retries + 1} tries; continuing")
            else:
                time.sleep(min(30, 5 * (attempt + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "run_config.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.resume and not args.dry_run and (HERE / "state.json").exists():
        print("state.json exists; pass --resume to continue or delete it to restart.")
        return 1
    return run_queue(args.config, args.resume, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
