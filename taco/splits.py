"""TACO data split (spec §4) -> results/taco/splits.json

Reuses the repository's existing frozen 80/149 split (seed 42,
`results/tool_graph_facility/split.json`) rather than drawing a new one.

Why reuse rather than re-draw:
  * the spec explicitly permits it ("if compatibility with the existing 80/149
    split is required: calibration 80, held-out 149, CV within the 80");
  * prior experiments in this repository already consumed those 80 tasks as
    calibration, so re-drawing would move previously-seen tasks into TACO's
    held-out set -- a real contamination risk, in exchange for nothing;
  * the split's `dataset_sha256` is recorded, so the identity of the 229 tasks
    is verifiable.

The spec's preferred 80/49/100 three-way split is obtained *inside* the
calibration 80 by grouped cross-validation (§17), not by moving held-out ids.

Stratification variables the spec asks for do not exist as fields in
GTA-Atomic, so they are *derived* and the achieved balance is measured and
reported (below) instead of being imposed by re-drawing:

  n_relevant_tools     len(task["tools"])                     1..4
  ref_chain_len        # assistant tool_calls in the reference dialog
  category             derived from the reference tool signature

Usage:  python taco/splits.py
"""
import hashlib
import json
import os
from collections import Counter

BIG = "/datasets/omni_pretraining/gta2"
DS = f"{BIG}/data/gta_dataset/dataset.json"
PRIOR = f"{BIG}/results/tool_graph_facility/split.json"
OUT = f"{BIG}/results/taco/splits.json"

# Derived coarse category: which perception/compute modality the reference
# solution leans on. Analysis / stratification only -- never a model feature
# in the deployable predictor.
CATEGORY_RULES = [
    ("ocr_text", {"OCR", "MathOCR"}),
    ("localization", {"TextToBbox", "DrawBox", "RegionAttributeDescription"}),
    ("counting", {"CountGivenObject"}),
    ("generation", {"TextToImage", "ImageStylization", "AddText", "Plot"}),
    ("knowledge", {"GoogleSearch"}),
    ("compute", {"Calculator", "Solver"}),
    ("description", {"ImageDescription"}),
]


def task_features(task):
    tools = [t["name"] for t in task.get("tools", [])]
    chain = [m for m in task.get("dialogs", []) if m.get("tool_calls")]
    chain_tools = [m["tool_calls"][0]["function"]["name"] for m in chain]
    cat = "other"
    for name, members in CATEGORY_RULES:
        if members & set(chain_tools or tools):
            cat = name
            break
    return dict(n_relevant_tools=len(tools), ref_chain_len=len(chain),
                category=cat, tools=tools, chain_tools=chain_tools)


def balance_table(feats, ids_a, ids_b, key):
    ca, cb = Counter(feats[i][key] for i in ids_a), Counter(feats[i][key] for i in ids_b)
    keys = sorted(set(ca) | set(cb), key=str)
    return {str(k): dict(calibration=ca.get(k, 0), calibration_pct=round(100 * ca.get(k, 0) / len(ids_a), 1),
                         heldout=cb.get(k, 0), heldout_pct=round(100 * cb.get(k, 0) / len(ids_b), 1))
            for k in keys}


def main():
    raw = open(DS, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    ds = json.loads(raw)
    prior = json.load(open(PRIOR))

    assert sha == prior["dataset_sha256"], (
        "dataset.json changed since the prior split was drawn (%s != %s)"
        % (sha[:12], prior["dataset_sha256"][:12]))

    cal, held = sorted(prior["calibration_ids"]), sorted(prior["heldout_ids"])
    assert not set(cal) & set(held)
    assert sorted(cal + held) == list(range(len(ds)))

    feats = {i: task_features(ds[str(i)]) for i in range(len(ds))}

    out = dict(
        source="reused from results/tool_graph_facility/split.json (seed 42)",
        rationale=("prior repository experiments already consumed these 80 tasks as "
                   "calibration; re-drawing would move previously-seen tasks into "
                   "TACO's held-out set. The spec's 80/49/100 three-way split is "
                   "realised as grouped CV inside the calibration 80."),
        dataset_sha256=sha,
        n_tasks=len(ds),
        primary_seed=42,
        calibration_ids=cal,
        heldout_ids=held,
        calibration_size=len(cal),
        heldout_size=len(held),
        # CV folds inside calibration ONLY (grouped by task; §17).
        cv_folds=5,
        cv_assignment={str(t): i % 5 for i, t in enumerate(cal)},
        stratification_achieved={
            k: balance_table(feats, cal, held, k)
            for k in ("n_relevant_tools", "ref_chain_len", "category")
        },
        task_features={str(i): feats[i] for i in range(len(ds))},
        leakage_rule=("held-out ids must not appear in any feature fit, hyperparameter "
                      "choice, subset search, threshold choice or environment freeze; "
                      "taco/runner.py enforces this mechanically"),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    print("splits ->", OUT)
    print("calibration %d / heldout %d / dataset sha %s" % (len(cal), len(held), sha[:12]))
    for k, tab in out["stratification_achieved"].items():
        print("\n%-18s %8s %8s" % (k, "cal%", "held%"))
        for kk, v in tab.items():
            print("  %-16s %7.1f %8.1f" % (kk, v["calibration_pct"], v["heldout_pct"]))


if __name__ == "__main__":
    main()
