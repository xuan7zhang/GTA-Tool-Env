"""Stage 4 -> Stage 5 handoff: freeze environments, emit the held-out manifest.

Writes frozen_environments/frozen_envs.json with a content hash, then builds
heldout manifests that the runner will only accept because they are marked
kind="heldout". Stage 5 asserts the hash before evaluating, so an environment
cannot be quietly re-tuned after seeing held-out data.

Environments (spec §21 global baselines + §20 query-level baselines):

  GLOBAL
    keep_all              all 14 tools, native output          (the default)
    keep_none             tools on the menu, every call returns 'unavailable'
    random_same_size_1..3 random masks of |M*|                 (seeded)
    topk_individual       global top-k by individual utility
    greedy_empirical      best subset observed on calibration
    coverage_facility     facility-location / category-coverage subset
    ground_plus           image-output declutter (this repo's prior baseline)
    uniform_concise       all 14 tools, F2 output
    uniform_json          all 14 tools, F4 output
    uniform_truncated     all 14 tools, L1 output
    taco_intrinsic        TACO-Intrinsic search result (mask + format + length)
    taco_conditional      TACO-Conditional averaged over calibration tasks

  QUERY-LEVEL TOOL SELECTION   (labelled; a different optimisation problem)
    ql_oracle_relevant    per-task annotated tools (upper bound, cheats)
    ql_embedding_topk     per-task top-k by query/description similarity
    ql_taco_conditional   per-task top-k by TACO-Conditional predicted utility
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taco.paths import TACO, FT, CFG, SEARCH, FROZEN, TOOLMETA, DS_PATH   # noqa: E402

IMAGE_OUT = ["AddText", "DrawBox", "ImageStylization", "Plot", "TextToImage"]
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


def spec(fmt="F0", level="L4"):
    return {"default": {"format": fmt, "length": {"level": level,
                                                  "mechanism": "relevant"},
                        "position": None}, "per_tool": {}}


def coverage_subset(tools, k):
    """Facility location on categories: cover as many categories as possible,
    then fill by within-category order. Coverage-only, utility-blind."""
    picked, seen = [], set()
    for t in tools:
        c = CATEGORY.get(t, "other")
        if c not in seen:
            picked.append(t)
            seen.add(c)
        if len(picked) >= k:
            break
    for t in tools:
        if len(picked) >= k:
            break
        if t not in picked:
            picked.append(t)
    return sorted(picked)


def query_level_masks(splits, ds, tt, k=3):
    """Per-task keep-sets for the query-level baselines (labelled as such)."""
    out = {}
    # oracle: the annotated per-task tools (cheats; upper bound only)
    oracle = {str(i): splits["task_features"][str(i)]["tools"]
              for i in splits["heldout_ids"]}
    # embedding/description similarity top-k, label-free
    emb = {}
    for i in splits["heldout_ids"]:
        d = tt[tt.task_id == i].sort_values("query_tool_jaccard", ascending=False)
        emb[str(i)] = sorted(d.head(k)["tool"].tolist())
    out["ql_oracle_relevant"] = oracle
    out["ql_embedding_topk"] = emb
    return out


def main():
    os.makedirs(FROZEN, exist_ok=True)
    tools = list(json.load(open(TOOLMETA)))
    splits = json.load(open(f"{TACO}/splits.json"))
    ds = json.load(open(DS_PATH))
    tt = pd.read_parquet(f"{FT}/task_tool_features.parquet")
    search = json.load(open(f"{SEARCH}/search_results.json"))
    runs = pd.read_parquet(f"{FT}/runs.parquet")

    rng = np.random.default_rng(42)
    envs = {}
    for model, res in search.items():
        sel = res["selected"]
        M = sorted(sel["tools"])
        k = len(M)
        e = {}
        e["keep_all"] = dict(tools=tools, taco_spec=None)
        e["keep_none"] = dict(tools=tools, taco_spec=None,
                              proxy_modes={t: "tool_free" for t in tools})
        for j in range(1, 4):
            e[f"random_same_size_{j}"] = dict(
                tools=sorted(rng.choice(tools, size=k, replace=False).tolist()),
                taco_spec=None)
        e["topk_individual"] = dict(
            tools=sorted(res["topk_individual"][str(k)]["tools"])
            if str(k) in res["topk_individual"] else M, taco_spec=None)
        emp = res.get("empirical_best_observed")
        e["greedy_empirical"] = dict(tools=sorted(emp["tools"]) if emp else M,
                                     taco_spec=None)
        e["coverage_facility"] = dict(tools=coverage_subset(tools, k), taco_spec=None)
        e["ground_plus"] = dict(tools=sorted(t for t in tools if t not in IMAGE_OUT),
                                taco_spec=None)
        e["uniform_concise"] = dict(tools=tools, taco_spec=spec("F2", "L4"))
        e["uniform_json"] = dict(tools=tools, taco_spec=spec("F4", "L4"))
        e["uniform_truncated"] = dict(tools=tools, taco_spec=spec("F0", "L1"))
        e["taco_intrinsic"] = dict(tools=M,
                                   taco_spec=spec(sel["format"], sel["length"]))
        cond = res.get("coordinate_descent", {}).get("best", sel)
        e["taco_conditional"] = dict(tools=sorted(cond["tools"]),
                                     taco_spec=spec(cond["format"], cond["length"]))
        envs[model] = e

    ql = query_level_masks(splits, ds, tt)
    for name, masks in ql.items():
        p = f"{FROZEN}/{name}_masks.json"
        json.dump(masks, open(p, "w"))

    payload = dict(
        environments=envs,
        query_level_masks={k: f"{FROZEN}/{k}_masks.json" for k in ql},
        note=("frozen before any held-out evaluation; Stage 5 asserts this file's "
              "hash. Query-level entries solve a different optimisation problem "
              "and are labelled QUERY-LEVEL TOOL SELECTION everywhere."),
        n_calibration_runs=int((runs.kind == "calibration").sum()),
    )
    blob = json.dumps(payload, sort_keys=True, default=str)
    payload["frozen_hash"] = hashlib.sha256(blob.encode()).hexdigest()
    json.dump(payload, open(f"{FROZEN}/frozen_envs.json", "w"), indent=1, default=str)
    print("[freeze] hash", payload["frozen_hash"][:16], "->", f"{FROZEN}/frozen_envs.json")

    # ---- held-out manifests -------------------------------------------
    for model, e in envs.items():
        tag = "7b" if "7b" in model else ("14b" if "14b" in model else model)
        rows = []
        for name, cfg in e.items():
            rows.append(dict(
                run_id=f"ho_{name}_{tag}", stage="heldout", kind="heldout",
                task_set="heldout",
                menu={"mode": "forced", "tools": cfg["tools"]},
                taco_spec=cfg.get("taco_spec"),
                proxy_modes=cfg.get("proxy_modes"),
                label="GLOBAL", notes=f"frozen environment {name}"))
        for name in ql:
            rows.append(dict(
                run_id=f"ho_{name}_{tag}", stage="heldout", kind="heldout",
                task_set="heldout", menu={"mode": "forced", "tools": tools},
                taco_spec=None, predicted_mask=f"{FROZEN}/{name}_masks.json",
                label="QUERY-LEVEL TOOL SELECTION",
                notes=f"query-level selector {name} (different problem)"))
        p = f"{CFG}/heldout_{tag}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[freeze] wrote {p} ({len(rows)} environments)")


if __name__ == "__main__":
    main()
