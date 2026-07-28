"""Build the paired-effect table and the feature tables from raw TACO runs.

Everything downstream (statistics, utility model, search, report) reads only
what this writes:

  results/taco/feature_tables/paired_effects.parquet   one row per (pair, task)
  results/taco/feature_tables/tool_attributes.parquet  per (run, tool)
  results/taco/feature_tables/task_tool_features.parquet
  results/taco/feature_tables/subset_features.parquet
  results/taco/feature_tables/runs.parquet             one row per run

Pairing rule
------------
An intervention run is paired with the control run that shares its model,
menu, task set and stage-block, and whose spec is the block's control cell
(F0/L4, no position). The delta is computed **per task**, so the noise floor
of a run-mean difference never enters an effect estimate.

Every row carries `fired` (the transform actually applied on that task) and
`engaged` (the agent called any tool), because an intervention on a tool the
agent never calls is a no-op: the intention-to-treat effect and the treated
effect are both reported, never silently mixed.
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taco.paths import BIG, TACO, FT   # noqa: E402

CATEGORY = {
    "OCR": "text_extraction", "MathOCR": "text_extraction",
    "ImageDescription": "image_description",
    "TextToBbox": "localization", "DrawBox": "localization",
    "RegionAttributeDescription": "regional_inspection",
    "CountGivenObject": "regional_inspection",
    "Calculator": "calculation", "Solver": "calculation", "Plot": "calculation",
    "GoogleSearch": "retrieval",
    "TextToImage": "generation", "ImageStylization": "generation", "AddText": "generation",
}
STOP = set("the a an of to in on at for and or is are was were be been it its this that "
           "as with by from we you i need answer question final so can will would should".split())


def is_abstention(ans: str) -> bool:
    return bool(ans and ABSTAIN_RE.search(ans))


def load_runs():
    runs = []
    for p in sorted(glob.glob(f"{TACO}/raw_runs/*/record.json")):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("status") != "OK":
            continue
        r["_dir"] = os.path.dirname(p)
        runs.append(r)
    return runs


def ll_by_answer(run_dir):
    """Map normalised final-answer text -> that generation's likelihood record."""
    p = f"{run_dir}/ll.jsonl"
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        out.append((norm(d.get("content", "")), d))
    return out


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"(final answer:|thought:|```|answer:)", "", s)
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def match_ll(ans, table):
    na = norm(ans)[:60]
    if not na:
        return None
    for c, d in table:
        if na and (na in c or c[:60] == na):
            return d
    return None


def spec_key(spec):
    """(format, length level, mechanism, position, scope) of a run's spec."""
    if not spec:
        return ("none", "none", "none", "none", "none")
    if spec.get("per_tool"):
        tool = sorted(spec["per_tool"])[0]
        s = spec["per_tool"][tool]
        scope = tool
    else:
        s = spec.get("default") or {}
        scope = "uniform"
    return (s.get("format", "F0"),
            (s.get("length") or {}).get("level", "L4"),
            (s.get("length") or {}).get("mechanism", "relevant"),
            s.get("position") or "native", scope)


def is_control(spec):
    f, l, m, p, _ = spec_key(spec)
    return (f, l, p) == ("F0", "L4", "native")


def build():
    os.makedirs(FT, exist_ok=True)
    runs = load_runs()
    ds = json.load(open(f"{BIG}/data/gta_dataset/dataset.json"))
    splits = json.load(open(f"{TACO}/splits.json"))
    tfeat = splits["task_features"]
    print(f"[effects] {len(runs)} OK runs")

    # ---------------- runs table ----------------
    rows = []
    for r in runs:
        f, l, m, p, scope = spec_key(r.get("taco_spec"))
        menu = r.get("menu") or {}
        rows.append(dict(
            run_id=r["run_id"], stage=r.get("stage"), kind=r.get("kind"),
            label=r.get("label"), model=r.get("model"),
            menu_mode=menu.get("mode"), menu_size=len(menu.get("tools") or []) or None,
            menu_tools=",".join(menu.get("tools") or []),
            format=f, length=l, mechanism=m, position=p, scope=scope,
            n_tasks=r["n_tasks"], n_scored=r["n_scored"],
            answer_acc=r["answer_acc"], official_answer_acc=r.get("official_answer_acc"),
            n_engaged=r["n_engaged"], n_fired=r["n_taco_fired"],
            abstention_rate=float(np.mean([is_abstention(t.get("answer"))
                                           for t in r["tasks"].values()])) if r["tasks"] else None,
            answered_rate=float(np.mean([bool(t.get("answered"))
                                         for t in r["tasks"].values()])) if r["tasks"] else None,
            tool_calls=r["tool_calls"], invalid_calls=r["invalid_calls"],
            visible_tool_tokens=r["visible_tool_tokens"],
            native_tool_tokens=r["native_tool_tokens"],
            mean_tool_latency_s=r.get("mean_tool_latency_s"),
            wall_s=r.get("wall_s"), env_hash=r.get("env_hash"),
            predicted_mask=r.get("predicted_mask") or "",
        ))
    runs_df = pd.DataFrame(rows)
    runs_df.to_parquet(f"{FT}/runs.parquet")

    # ---------------- pair the runs ----------------
    def block_key(r):
        menu = r.get("menu") or {}
        _, _, _, _, scope = spec_key(r.get("taco_spec"))
        # Stage prefix only ("stage1"/"stage2"/"heldou"): blocks 2A-2D share
        # one task set, one menu and one F0/L4 control, so keying on the full
        # block name would leave the length, mechanism and position blocks
        # with no control to pair against. Block 2E stays separate anyway --
        # it has its own per-tool task set and scope.
        return (r.get("model"), menu.get("mode"), ",".join(sorted(menu.get("tools") or [])),
                str(sorted(r_tasks(r))), scope, r.get("stage", "")[:6])

    def r_tasks(r):
        return sorted(int(k) for k in r["tasks"])

    controls = {}
    for r in runs:
        if is_control(r.get("taco_spec")):
            controls[block_key(r)] = r
    # a run with no intervention at all (taco_spec=None) is also a valid control
    for r in runs:
        k = block_key(r)
        if k not in controls and not r.get("taco_spec"):
            controls[k] = r

    prows = []
    for r in runs:
        if is_control(r.get("taco_spec")) or not r.get("taco_spec"):
            continue
        ctl = controls.get(block_key(r))
        if ctl is None:
            continue
        f, l, m, p, scope = spec_key(r["taco_spec"])
        ll_i, ll_c = ll_by_answer(r["_dir"]), ll_by_answer(ctl["_dir"])
        menu = r.get("menu") or {}
        for tid, ti in r["tasks"].items():
            tc = ctl["tasks"].get(tid)
            if tc is None or ti["correct"] is None or tc["correct"] is None:
                continue
            if ti["bert_ref"] or tc["bert_ref"]:
                continue
            di = match_ll(ti["answer"], ll_i)
            dc = match_ll(tc["answer"], ll_c)
            tf = tfeat[tid]
            prows.append(dict(
                pair_id=f"{r['run_id']}|{ctl['run_id']}", run_id=r["run_id"],
                control_id=ctl["run_id"], stage=r.get("stage"), model=r["model"],
                task_id=int(tid), scope=scope,
                attribute=("format" if l == "L4" and p == "native" else
                           "position" if p != "native" else "length"),
                format=f, length=l, mechanism=m, position=p,
                context=("min" if menu.get("mode") == "per_task" else "full"),
                menu_size=(len(menu.get("tools") or []) or tf["n_relevant_tools"]),
                # outcomes
                y=ti["correct"], y_ctl=tc["correct"],
                delta_acc=ti["correct"] - tc["correct"],
                fired=bool(ti.get("taco_fired")), fired_ctl=bool(tc.get("taco_fired")),
                engaged=bool(ti["engaged"]), engaged_ctl=bool(tc["engaged"]),
                n_calls=ti["n_calls"], n_calls_ctl=tc["n_calls"],
                abstained=int(is_abstention(ti.get("answer"))),
                abstained_ctl=int(is_abstention(tc.get("answer"))),
                n_turns=ti["n_turns"], invalid=ti["n_invalid"],
                tokens=ti.get("visible_tool_tokens"),
                tokens_ctl=tc.get("visible_tool_tokens"),
                evidence_density=ti.get("evidence_density"),
                adoption=ti.get("output_adoption"), adoption_ctl=tc.get("output_adoption"),
                answer_overlap=ti.get("answer_tool_overlap"),
                answer_overlap_ctl=tc.get("answer_tool_overlap"),
                mean_logprob=(di or {}).get("mean_logprob"),
                mean_logprob_ctl=(dc or {}).get("mean_logprob"),
                min_logprob=(di or {}).get("min_logprob"),
                min_logprob_ctl=(dc or {}).get("min_logprob"),
                mean_entropy=(di or {}).get("mean_entropy"),
                mean_entropy_ctl=(dc or {}).get("mean_entropy"),
                # task features (analysis-only ones flagged by name)
                task_category=tf["category"], ref_chain_len=tf["ref_chain_len"],
                n_relevant_tools=tf["n_relevant_tools"],
                oracle_tools=",".join(tf["tools"]),
                tool_relevant_annot=(scope in tf["tools"]) if scope != "uniform" else None,
                query_len=len(ds[tid]["dialogs"][0]["content"].split()),
            ))
    pe = pd.DataFrame(prows)
    if len(pe):
        pe["delta_ll"] = pe["mean_logprob"] - pe["mean_logprob_ctl"]
        pe["delta_tokens"] = pe["tokens"] - pe["tokens_ctl"]
        pe["delta_adoption"] = pe["adoption"] - pe["adoption_ctl"]
        pe["treated"] = pe["fired"] | pe["fired_ctl"]
        pe["delta_abstain"] = pe["abstained"] - pe["abstained_ctl"]
    pe.to_parquet(f"{FT}/paired_effects.parquet")
    print(f"[effects] paired_effects: {len(pe)} rows, "
          f"{pe['pair_id'].nunique() if len(pe) else 0} pairs")

    # ---------------- tool attributes (per run x tool) ----------------
    trows = []
    for r in runs:
        tl_path = f"{r['_dir']}/transform.jsonl"
        if not os.path.exists(tl_path):
            continue
        agg = defaultdict(list)
        for line in open(tl_path):
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
            except Exception:
                continue
            if x.get("applied"):
                agg[x["tool"]].append(x)
        for tool, xs in agg.items():
            trows.append(dict(
                run_id=r["run_id"], model=r["model"], tool=tool,
                category=CATEGORY.get(tool, "other"),
                format=xs[0]["format"], length=xs[0]["spec"]["length"]["level"],
                n_calls=len(xs),
                native_tokens=float(np.mean([x["tokens_before"] for x in xs])),
                transformed_tokens=float(np.mean([x["tokens_after"] for x in xs])),
                evidence_tokens=float(np.mean([x["evidence_tokens"] for x in xs])),
                evidence_density=float(np.mean([x["evidence_density"] for x in xs])),
                repetition_ratio=float(np.mean([x.get("repetition_ratio", 0) for x in xs])),
                n_units=float(np.mean([x["n_units_total"] for x in xs])),
                is_json=int(xs[0]["format"] == "F4"),
                is_kv=int(xs[0]["format"] == "F3"),
                parseable_json=int(xs[0]["format"] == "F4"),
                evidence_position=xs[0]["evidence_position"],
            ))
    pd.DataFrame(trows).to_parquet(f"{FT}/tool_attributes.parquet")
    print(f"[effects] tool_attributes: {len(trows)} rows")

    # ---------------- task-tool features (label-free + annotated) ----------
    ttrows = []
    tools = list(json.load(open(f"{BIG}/data/gta_dataset/toolmeta.json")).items())
    for tid, tf in tfeat.items():
        q = ds[tid]["dialogs"][0]["content"]
        qt = set(re.findall(r"[a-z]+", q.lower())) - STOP
        for name, meta in tools:
            dt = set(re.findall(r"[a-z]+", (meta.get("description") or "").lower())) - STOP
            jac = len(qt & dt) / max(1, len(qt | dt))
            ttrows.append(dict(
                task_id=int(tid), tool=name, category=CATEGORY.get(name, "other"),
                # label-free
                query_tool_jaccard=jac, query_len=len(q.split()),
                desc_len=len((meta.get("description") or "").split()),
                # analysis-only (never enters the deployable predictor)
                annot_relevant=int(name in tf["tools"]),
                annot_in_ref_chain=int(name in tf["chain_tools"]),
                task_category=tf["category"], ref_chain_len=tf["ref_chain_len"],
            ))
    pd.DataFrame(ttrows).to_parquet(f"{FT}/task_tool_features.parquet")
    print(f"[effects] task_tool_features: {len(ttrows)} rows")

    # ---------------- subset features ----------------
    srows = []
    for r in runs:
        menu = r.get("menu") or {}
        ts = menu.get("tools")
        if menu.get("mode") != "forced" or not ts:
            continue
        cats = [CATEGORY.get(t, "other") for t in ts]
        srows.append(dict(
            run_id=r["run_id"], model=r["model"], menu_size=len(ts),
            tools=",".join(sorted(ts)),
            n_categories=len(set(cats)),
            max_same_category=max([cats.count(c) for c in set(cats)] or [0]),
            desc_tokens=sum(len((json.load(open(f"{BIG}/data/gta_dataset/toolmeta.json"))
                                 .get(t, {}).get("description") or "").split()) for t in ts),
            n_image_output=sum(1 for t in ts if CATEGORY.get(t) == "generation"),
            n_unavailable=sum(1 for t in ts if t in ("GoogleSearch", "MathOCR")),
            answer_acc=r["answer_acc"], n_scored=r["n_scored"],
            visible_tool_tokens=r["visible_tool_tokens"],
            tool_calls=r["tool_calls"], n_engaged=r["n_engaged"],
        ))
    pd.DataFrame(srows).to_parquet(f"{FT}/subset_features.parquet")
    print(f"[effects] subset_features: {len(srows)} rows")
    return pe


if __name__ == "__main__":
    build()
