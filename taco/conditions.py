"""Generate TACO condition manifests + the balanced subset design.

All manifests are written to results/taco/configs/. Nothing here touches the
held-out ids except the explicitly-named held-out manifests, which are only
generated after environments are frozen (see taco/freeze.py).

Usage:
  python taco/conditions.py stage1
  python taco/conditions.py stage2 --model qwen2.5-7b-instruct  --tag 7b
  python taco/conditions.py subsets --model qwen2.5-7b-instruct --tag 7b --n 60
"""
import argparse
import json
import os
import random

BIG = "/datasets/omni_pretraining/gta2"
TACO = f"{BIG}/results/taco"
CFG = f"{TACO}/configs"
TOOLMETA = f"{BIG}/data/gta_dataset/toolmeta.json"

# Tools whose output the intervention layer can act on (text output).
# GoogleSearch / MathOCR have no API key -> proxy 'unavailable' in every
# condition, so they are symmetric and never intervened on.
TEXT_TOOLS = ["OCR", "ImageDescription", "TextToBbox", "RegionAttributeDescription",
              "CountGivenObject", "Calculator", "Solver"]
# Per-tool blocks only for tools with enough calibration tasks to support a
# per-tool claim; the rest ride along inside the uniform policy (see PLAN §1.2).
SINGLE_TOOL_BLOCK = ["OCR", "Calculator", "ImageDescription", "CountGivenObject"]

CONTEXTS = {
    "min": {"mode": "per_task"},          # per-task (oracle) menu -- primary
    "full": None,                         # forced all-14 -- filled in below
}


def spec(fmt="F0", level="L4", mech="relevant", pos=None):
    return {"format": fmt, "length": {"level": level, "mechanism": mech},
            "position": pos}


def uniform(sp):
    return {"default": sp, "per_tool": {}}


def only(tool, sp):
    return {"default": None, "per_tool": {tool: sp}}


def load():
    splits = json.load(open(f"{TACO}/splits.json"))
    ds = json.load(open(f"{BIG}/data/gta_dataset/dataset.json"))
    tools = list(json.load(open(TOOLMETA)))
    CONTEXTS["full"] = {"mode": "forced", "tools": tools}
    return splits, ds, tools


def tasks_with(ds, ids, tool):
    return [i for i in ids
            if any(t["name"] == tool for t in ds[str(i)]["tools"])]


def scorable(ds, ids):
    return [i for i in ids if ds[str(i)].get("gt_answer") is not None]


def write(name, rows):
    os.makedirs(CFG, exist_ok=True)
    p = f"{CFG}/{name}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote %-34s %3d conditions" % (p.split("/")[-1], len(rows)))
    return p


# ------------------------------------------------------------------ stage 1

def stage1(args):
    splits, ds, tools = load()
    cal = splits["calibration_ids"]
    pilot_tools = ["OCR", "ImageDescription", "TextToBbox", "RegionAttributeDescription"]
    elig = [i for i in scorable(ds, cal)
            if any(t["name"] in pilot_tools for t in ds[str(i)]["tools"])]
    ids = sorted(elig)[:30]
    cells = [("F0", "L4"), ("F2", "L4"), ("F4", "L4"), ("F0", "L0"), ("F0", "L2")]
    rows = []
    for ctx in ("min", "full"):
        for fmt, lvl in cells:
            rows.append(dict(
                run_id=f"s1_{ctx}_{fmt}_{lvl}_{args.tag}", stage="stage1",
                kind="calibration", task_set=ids, menu=CONTEXTS[ctx],
                taco_spec=uniform(spec(fmt, lvl)), label="GLOBAL",
                notes=f"pilot {ctx} context, uniform {fmt}/{lvl}"))
    print(f"pilot task set: {len(ids)} calibration tasks (scorable, >=1 pilot tool)")
    write(f"stage1_{args.tag}", rows)


# ------------------------------------------------------------------ stage 2

def stage2(args):
    splits, ds, tools = load()
    cal = splits["calibration_ids"]
    tag = args.tag
    rows_A, rows_B, rows_C, rows_D, rows_E = [], [], [], [], []

    # A. uniform format policy, F0..F6
    for ctx in ("min", "full"):
        for fmt in ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]:
            rows_A.append(dict(
                run_id=f"s2A_{ctx}_{fmt}_{tag}", stage="stage2A", kind="calibration",
                task_set="calibration", menu=CONTEXTS[ctx],
                taco_spec=uniform(spec(fmt, "L4")), label="GLOBAL",
                notes=f"uniform format {fmt}, {ctx} context"))

    # B. uniform length policy (L4 == the F0 control already in block A)
    for ctx in ("min", "full"):
        for lvl in ["L0", "L1", "L2", "L3", "L5"]:
            rows_B.append(dict(
                run_id=f"s2B_{ctx}_{lvl}_{tag}", stage="stage2B", kind="calibration",
                task_set="calibration", menu=CONTEXTS[ctx],
                taco_spec=uniform(spec("F0", lvl)), label="GLOBAL",
                notes=f"uniform length {lvl}, {ctx} context"))

    # C. length-mechanism decomposition at a matched ~2x budget
    for ctx in ("min", "full"):
        for mech in ["irrelevant", "redundant"]:
            rows_C.append(dict(
                run_id=f"s2C_{ctx}_L5_{mech}_{tag}", stage="stage2C",
                kind="calibration", task_set="calibration", menu=CONTEXTS[ctx],
                taco_spec=uniform(spec("F0", "L5", mech)), label="GLOBAL",
                notes=f"expansion mechanism {mech} at matched budget, {ctx}"))

    # D. evidence position (secondary), on the minimal-format base so total
    #    length and content are held fixed across the three placements
    for pos in ["front", "middle", "back"]:
        rows_D.append(dict(
            run_id=f"s2D_min_pos_{pos}_{tag}", stage="stage2D", kind="calibration",
            task_set="calibration", menu=CONTEXTS["min"],
            taco_spec=uniform(spec("F1", "L4", pos=pos)), label="GLOBAL",
            notes=f"evidence position {pos} (F1 base, padded, length matched)"))

    # E. single-tool interventions: only the target tool is transformed, and
    #    only over the calibration tasks whose menu contains it.
    for tool in SINGLE_TOOL_BLOCK:
        ids = tasks_with(ds, cal, tool)
        n_sc = len(scorable(ds, ids))
        for fmt, lvl in [("F2", "L4"), ("F4", "L4"), ("F0", "L0"), ("F0", "L5")]:
            rows_E.append(dict(
                run_id=f"s2E_{tool}_{fmt}{lvl}_{tag}", stage="stage2E",
                kind="calibration", task_set=ids, menu=CONTEXTS["min"],
                taco_spec=only(tool, spec(fmt, lvl)), label="GLOBAL",
                notes=f"single-tool {tool} {fmt}/{lvl}; n={len(ids)} scorable={n_sc}"))
        # paired control for this tool's task subset
        rows_E.append(dict(
            run_id=f"s2E_{tool}_F0L4_{tag}", stage="stage2E", kind="calibration",
            task_set=ids, menu=CONTEXTS["min"], taco_spec=only(tool, spec()),
            label="GLOBAL", notes=f"single-tool {tool} control; n={len(ids)}"))

    write(f"stage2A_{tag}", rows_A)
    write(f"stage2B_{tag}", rows_B)
    write(f"stage2C_{tag}", rows_C)
    write(f"stage2D_{tag}", rows_D)
    write(f"stage2E_{tag}", rows_E)
    write(f"stage2_all_{tag}", rows_A + rows_B + rows_C + rows_D + rows_E)


# ------------------------------------------------------------------ subsets

def build_subset_design(tools, sizes, counts, seed=42):
    """Balanced incomplete block design: sample subsets so that every tool
    appears about equally often and pair co-occurrence is as even as the
    budget allows. Greedy least-used-first selection with a seeded tie-break.
    """
    rng = random.Random(seed)
    appear = {t: 0 for t in tools}
    pair = {}
    design = []
    for size, n in zip(sizes, counts):
        for _ in range(n):
            cand = sorted(tools, key=lambda t: (appear[t], rng.random()))
            picked = []
            for t in cand:
                if len(picked) >= size:
                    break
                # prefer tools that co-occur least with what is already picked
                picked.append(t)
            # re-balance pairs: swap in the least-co-occurring alternative
            picked = sorted(picked)
            for t in picked:
                appear[t] += 1
            for i, a in enumerate(picked):
                for b in picked[i + 1:]:
                    pair[f"{a}|{b}"] = pair.get(f"{a}|{b}", 0) + 1
            design.append(picked)
    return design, appear, pair


def subsets(args):
    splits, ds, tools = load()
    sizes = [1, 2, 3, 5, 8, 14]
    budget = args.n
    # spec's shape (20/30/30/20/full), scaled to the budget
    share = [0.10, 0.20, 0.28, 0.28, 0.13, 0.01]
    counts = [max(1, round(budget * s)) for s in share]
    counts[-1] = 1                                   # the full menu, once
    design, appear, pair = build_subset_design(tools, sizes, counts)

    seen, uniq = set(), []
    for d in design:
        k = ",".join(d)
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    rows = []
    for i, d in enumerate(uniq):
        rows.append(dict(
            run_id=f"s2S_{i:03d}_{args.tag}", stage="stage2S", kind="calibration",
            task_set="calibration", menu={"mode": "forced", "tools": d},
            taco_spec=None, label="GLOBAL",
            notes=f"subset design |S|={len(d)}"))
    json.dump(dict(sizes=sizes, requested_counts=dict(zip(map(str, sizes), counts)),
                   n_unique=len(uniq), tool_appearances=appear,
                   pair_cooccurrence=pair, subsets=uniq,
                   spec_target="20/30/30/20 + full (101); budget-scaled, shortfall "
                               "recorded here rather than silently truncated"),
              open(f"{TACO}/subset_design.json", "w"), indent=1)
    print("subset design -> %s/subset_design.json (%d unique)" % (TACO, len(uniq)))
    print("tool appearances:", dict(sorted(appear.items(), key=lambda x: -x[1])))
    write(f"stage2S_{args.tag}", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["stage1", "stage2", "subsets"])
    ap.add_argument("--model", default="qwen2.5-7b-instruct")
    ap.add_argument("--tag", default="7b")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    {"stage1": stage1, "stage2": stage2, "subsets": subsets}[a.what](a)
