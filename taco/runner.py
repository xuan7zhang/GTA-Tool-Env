"""TACO condition runner.

Executes one manifest of conditions sequentially on one lane, writing a
self-describing record per run. Re-running is safe: a run with a `DONE`
marker is skipped, so a lane can be killed and restarted at any point.

Manifest = JSONL, one condition per line:

  {"run_id":   "s1_fmt_F2_min_7b",
   "stage":    "stage1",
   "kind":     "calibration" | "heldout",
   "task_set": "calibration" | "heldout" | [ids...],
   "menu":     {"mode": "per_task"}                      # oracle/minimal menu
            |  {"mode": "forced", "tools": [...]},       # fixed global env
   "taco_spec": {...} | null,                            # output intervention
   "predicted_mask": "<path.json>" | null,               # QUERY-LEVEL baseline
   "label":    "GLOBAL" | "QUERY-LEVEL TOOL SELECTION",
   "notes":    "..."}

Invariants held fixed across every condition (spec §2): task inputs, real tool
implementations, ReAct loop, system prompt, answer parser, max turns,
evaluator, greedy decoding, task ordering, model revision.

Usage:
  python taco/runner.py --manifest results/taco/configs/stage1.jsonl --lane 1 \
                        --model qwen2.5-7b-instruct
"""
import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

LAB = "/project/6101776/xzhan576/gta2-envlab"
BIG = "/datasets/omni_pretraining/gta2"
TACO = f"{BIG}/results/taco"
DS_PATH = f"{BIG}/data/gta_dataset/dataset.json"
TOOLMETA = f"{BIG}/data/gta_dataset/toolmeta.json"
UNAVAILABLE = ["GoogleSearch", "MathOCR"]      # no API keys; symmetric everywhere

LANE_PORTS = {1: dict(llm=12580, proxy=16281), 2: dict(llm=22580, proxy=26281)}


# ---------------------------------------------------------------- outcomes

def iscorrect(pred: str, ref: dict) -> bool:
    """Byte-for-byte the GTABenchEvaluator rule (mode='every', dict refs)."""
    count = 0
    for aliases in ref["whitelist"]:
        pattern = r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b"
        if re.search(pattern, pred, re.IGNORECASE):
            count += 1
    if not ref.get("blacklist"):
        return count == len(ref["whitelist"])
    pat_bk = r"\b(?:" + "|".join(re.escape(a) for g in ref["blacklist"] for a in g) + r")\b"
    return count == len(ref["whitelist"]) and not re.search(pat_bk, pred, re.IGNORECASE)


def per_task_outcomes(pred_file: str, ds: dict) -> dict:
    """Per-task primary + behavioural outcomes from one predictions file.

    Scoring replicates the evaluator: only the LAST message counts, and it
    must be an answer (not a tool call). Tasks whose gt_answer is null are
    unscorable (image-generation tasks) and excluded from AnsAcc, exactly as
    the evaluator excludes them. Tasks with list-valued refs need the
    sentence-embedding simscore and are marked `bert_ref` -- they are excluded
    from the paired binary analysis and reported separately.
    """
    d = json.load(open(pred_file))
    out = {}
    for k, v in d.items():
        p = v.get("prediction") or v.get("predictions")
        try:
            p = ast.literal_eval(p) if isinstance(p, str) else p
        except Exception:
            p = None
        turn = p[0] if p and isinstance(p[0], list) else p
        turn = turn or []
        ref = ds[str(k)].get("gt_answer")

        calls = [m["tool_calls"][0]["function"]["name"] for m in turn
                 if isinstance(m, dict) and m.get("tool_calls")]
        errors = [m for m in turn if isinstance(m, dict) and m.get("error")]
        last = turn[-1] if turn else {}
        is_answer = isinstance(last, dict) and not last.get("tool_calls") \
            and last.get("role") == "assistant" and bool(last.get("content"))
        answer = last.get("content", "") if is_answer else ""

        rec = dict(task_id=int(k), n_calls=len(calls), tools_called=calls,
                   n_invalid=len(errors), n_turns=len(turn),
                   engaged=bool(calls), answered=is_answer,
                   answer=answer[:2000], bert_ref=isinstance(ref, list),
                   scorable=ref is not None)
        if isinstance(ref, dict) and is_answer:
            rec["correct"] = 1.0 if iscorrect(answer, ref) else 0.0
        elif isinstance(ref, dict):
            rec["correct"] = 0.0            # ended on a tool call => wrong
        else:
            rec["correct"] = None
        out[str(k)] = rec
    return out


def adoption(rec: dict, tlog: list) -> dict:
    """Did the final answer reuse what the tool returned?

    `output_adoption` = fraction of this task's transformed tool outputs whose
    evidence contributes a content token to the final answer. Also the raw
    answer/tool-output overlap used as the answer-echo control in §14.
    """
    def toks(s):
        return set(re.findall(r"[a-z0-9]+", (s or "").lower())) - {
            "the", "a", "an", "of", "to", "in", "is", "and", "or", "it", "on"}
    ans = toks(rec.get("answer"))
    if not ans or not tlog:
        return dict(output_adoption=None, answer_tool_overlap=None)
    hits, ov = 0, 0.0
    for t in tlog:
        tt = toks(t.get("transformed") or t.get("original"))
        if not tt:
            continue
        inter = len(ans & tt)
        hits += 1 if inter else 0
        ov = max(ov, inter / max(1, len(ans)))
    return dict(output_adoption=hits / len(tlog), answer_tool_overlap=ov)


# ---------------------------------------------------------------- execution

def set_proxy(port: int, cfg: dict):
    body = json.dumps(cfg).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/proxy_config", data=body,
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()


def run_one(cond: dict, lane: int, model: str, splits: dict, ds: dict,
            all_tools: list, force: bool = False) -> dict:
    rid = cond["run_id"]
    wd = f"{TACO}/raw_runs/{rid}"
    if os.path.exists(f"{wd}/DONE") and not force:
        print(f"[taco] skip {rid} (DONE)")
        return json.load(open(f"{wd}/record.json"))
    os.makedirs(wd, exist_ok=True)

    # ---- task ids + leakage guard -------------------------------------
    ts = cond["task_set"]
    if ts == "calibration":
        ids = splits["calibration_ids"]
    elif ts == "heldout":
        ids = splits["heldout_ids"]
    else:
        ids = list(ts)
    held = set(splits["heldout_ids"])
    if cond.get("kind") != "heldout" and (set(ids) & held):
        raise SystemExit(f"LEAKAGE GUARD: {rid} is not a held-out run but its task "
                         f"set touches {len(set(ids) & held)} held-out ids")

    ports = LANE_PORTS[lane]
    tlog = f"{wd}/transform.jsonl"
    llog = f"{wd}/ll.jsonl"
    for f in (tlog, llog, f"{wd}/proxy.jsonl"):
        open(f, "w").close()

    # ---- environment E = (M, C, Phi) ----------------------------------
    menu = cond.get("menu") or {"mode": "per_task"}
    if menu["mode"] == "forced":
        pool = list(menu["tools"])
        extra = ",".join(pool)
    else:
        pool = list(all_tools)
        extra = ""

    set_proxy(ports["proxy"], dict(
        mode="passthrough", mask=pool, per_tool_modes={}, phi_toolmeta=None,
        unavailable_tools=UNAVAILABLE, log_path=f"{wd}/proxy.jsonl",
        taco_spec=cond.get("taco_spec"), taco_log_path=tlog,
        run_meta=dict(run_id=rid, stage=cond.get("stage"), model=model)))

    env = dict(os.environ)
    env.update(
        GTA_MODEL_NAME=model,
        GTA_LLM_URL=f"http://127.0.0.1:{ports['llm']}/v1/chat/completions",
        GTA_TOOLSERVER=f"http://127.0.0.1:{ports['proxy']}",
        GTA_TOOLMETA=TOOLMETA, GTA_EVAL_MODES="end", GTA_TEMP="0",
        GTA_TASK_IDS=",".join(str(i) for i in ids),
        GTA_EXTRA_TOOLS=extra, GTA_HIDE_TOOLS="",
        GTA_LL_LOG=llog, GTA_ENGAGE_LOG=f"{wd}/engage.jsonl",
        GTA_MAX_TURN="10",
    )
    if cond.get("predicted_mask"):
        env["GTA_PREDICTED_MASK"] = cond["predicted_mask"]
    else:
        env.pop("GTA_PREDICTED_MASK", None)

    t0 = time.time()
    cmd = ["python", f"{LAB}/GTA/opencompass/run.py",
           f"{LAB}/configs/gta_atomic_env.py", "--max-num-workers", "1",
           "--debug", "-w", wd]
    with open(f"{wd}/run.log", "w") as lf:
        proc = subprocess.run(cmd, env=env, cwd=f"{BIG}/ocrun",
                              stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0

    # ---- collect ------------------------------------------------------
    import glob
    res = sorted(glob.glob(f"{wd}/*/results/*/gta_bench_end.json"))
    pred = sorted(glob.glob(f"{wd}/*/predictions/*/gta_bench_end.json"))
    if not res or not pred:
        rec = dict(run_id=rid, status="FAILED", rc=proc.returncode, wall_s=wall,
                   **{k: cond[k] for k in ("stage", "kind", "label") if k in cond})
        json.dump(rec, open(f"{wd}/record.json", "w"), indent=1)
        print(f"[taco] FAILED {rid} rc={proc.returncode}")
        return rec

    metrics = json.load(open(res[-1]))
    tasks = per_task_outcomes(pred[-1], ds)

    tl = [json.loads(x) for x in open(tlog) if x.strip()]
    by_task = {}
    for t in tl:
        by_task.setdefault(str(t.get("task_id")), []).append(t)
    for k, r in tasks.items():
        r.update(adoption(r, by_task.get(k, [])))
        tt = by_task.get(k, [])
        r["taco_fired"] = any(x.get("applied") for x in tt)
        r["visible_tool_tokens"] = sum(x.get("tokens_after") or 0 for x in tt)
        r["native_tool_tokens"] = sum(x.get("tokens_before") or 0 for x in tt)
        r["evidence_density"] = ((sum(x.get("evidence_tokens") or 0 for x in tt) /
                                  max(1, r["visible_tool_tokens"])) if tt else None)

    px = [json.loads(x) for x in open(f"{wd}/proxy.jsonl") if x.strip()]
    lat = [p.get("latency_s", 0) for p in px if p.get("tool")]
    scored = [r for r in tasks.values() if r["correct"] is not None and not r["bert_ref"]]

    rec = dict(
        run_id=rid, status="OK", wall_s=round(wall, 1), model=model, lane=lane,
        stage=cond.get("stage"), kind=cond.get("kind"),
        label=cond.get("label", "GLOBAL"), notes=cond.get("notes", ""),
        menu=menu, taco_spec=cond.get("taco_spec"),
        predicted_mask=cond.get("predicted_mask"),
        n_tasks=len(ids), n_scored=len(scored),
        official_answer_acc=metrics.get("answer_acc"),
        answer_acc=100.0 * sum(r["correct"] for r in scored) / max(1, len(scored)),
        n_engaged=sum(1 for r in tasks.values() if r["engaged"]),
        n_taco_fired=sum(1 for r in tasks.values() if r["taco_fired"]),
        tool_calls=sum(r["n_calls"] for r in tasks.values()),
        invalid_calls=sum(r["n_invalid"] for r in tasks.values()),
        mean_turns=sum(r["n_turns"] for r in tasks.values()) / max(1, len(tasks)),
        visible_tool_tokens=sum(r["visible_tool_tokens"] for r in tasks.values()),
        native_tool_tokens=sum(r["native_tool_tokens"] for r in tasks.values()),
        mean_tool_latency_s=(sum(lat) / len(lat)) if lat else None,
        tool_exec_failures=sum(1 for p in px if p.get("status") not in (200, None)),
        n_transform_records=len(tl),
        transform_preservation_fail=sum(
            1 for x in tl if x.get("applied")
            and not (x.get("preservation") or {}).get("units_present", True)),
        env_hash=hashlib.sha256(json.dumps(
            dict(menu=menu, spec=cond.get("taco_spec"), model=model),
            sort_keys=True).encode()).hexdigest()[:16],
        tasks=tasks,
    )
    json.dump(rec, open(f"{wd}/record.json", "w"), indent=1)
    open(f"{wd}/DONE", "w").close()
    print("[taco] %-34s acc=%5.1f (n=%d) fired=%d/%d engaged=%d tok=%d  %.0fs"
          % (rid, rec["answer_acc"], rec["n_scored"], rec["n_taco_fired"],
             rec["n_tasks"], rec["n_engaged"], rec["visible_tool_tokens"], wall))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lane", type=int, default=1)
    ap.add_argument("--model", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    splits = json.load(open(f"{TACO}/splits.json"))
    ds = json.load(open(DS_PATH))
    all_tools = list(json.load(open(TOOLMETA)))
    conds = [json.loads(x) for x in open(args.manifest) if x.strip()]
    print(f"[taco] lane {args.lane} model {args.model}: {len(conds)} conditions")
    for c in conds:
        try:
            run_one(c, args.lane, args.model, splits, ds, all_tools, args.force)
        except SystemExit:
            raise
        except Exception as e:                       # keep the lane alive
            print(f"[taco] ERROR {c.get('run_id')}: {type(e).__name__}: {e}")
    print("[taco] manifest complete:", args.manifest)


if __name__ == "__main__":
    main()
