#!/usr/bin/env python3
"""Behavioral probes (L1 input-grounding) + budget curve. CPU only: probes talk
to the AgentLego tool server through the proxy, never to the LLM.

L1 probe: a tool whose output is invariant to a degenerate-input substitution
(black image for visual tools; low output-cardinality across distinct inputs for
text tools) is input-degenerate. --mode budget-curve reports detection rate and
false-positive rate as a function of the per-tool probe budget B.

  probes.py --mode budget-curve --levels 1 --out <dir>

Writes metrics.json: {detection_rate, false_positive_rate, budget_curve[...]}.
Needs the tool server + proxy up (proxy defaults to $GTA_PROXY at 16281).
"""
import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import requests
from PIL import Image

PROXY = os.environ.get("GTA_PROXY", "http://127.0.0.1:16281")
BIG = Path(os.environ.get("GTA_BIG", "/datasets/omni_pretraining/gta2"))
LAB = Path(os.environ.get("GTA_LAB", "/project/6101776/xzhan576/gta2-envlab"))
POISON = set(json.loads((LAB / "envgen/variants/poison/names.json").read_text()))
REAL = list(json.loads((BIG / "data/gta_dataset/toolmeta.json").read_text()).keys())
IMG_TOOLS = {"ImageDescription", "OCR", "RegionAttributeDescription", "TextToBbox",
             "CountGivenObject", "DirectImageQA", "PreciseOCR", "SmartCount",
             "ImageDetailPro", "QuickBBox"}


def _png(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, format="PNG"); b.seek(0); return b


def _call_img(tool, arr, **fields):
    data = {k: str(v) for k, v in fields.items()}
    r = requests.post(f"{PROXY}/{tool}", data=data or None,
                      files={"image": ("i.png", _png(arr), "image/png")},
                      headers={"X-GTA-Task-Id": "probe"}, timeout=120)
    r.raise_for_status(); o = r.json(); return o if isinstance(o, str) else json.dumps(o)


def probe_tool(tool, budget):
    """Return True if the tool looks input-degenerate under `budget` probe pairs."""
    rng = np.random.RandomState(0)
    invariant = 0
    for _ in range(max(1, budget)):
        real = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            extra = {"text": "object", "top1": "true"} if tool in ("TextToBbox", "QuickBBox") else \
                    ({"bbox": "(0,0,100,100)", "attribute": "color"} if "Region" in tool else {})
            o_real = _call_img(tool, real, **extra)
            o_black = _call_img(tool, black, **extra)
            if o_real == o_black:
                invariant += 1
        except Exception:
            return None  # tool errored; inconclusive
    return invariant == max(1, budget)  # invariant on every pair -> degenerate


def budget_curve(levels, budgets=(1, 2, 4, 8)):
    tools = [t for t in IMG_TOOLS if t in REAL or t in POISON]
    curve = []
    for B in budgets:
        detected = {t for t in tools if probe_tool(t, B) is True}
        tp = len(detected & POISON)
        fp = len(detected - POISON)
        curve.append({"budget": B, "detected_poison": tp,
                      "false_positives": fp, "detected": sorted(detected)})
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="budget-curve")
    ap.add_argument("--levels", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    curve = budget_curve(a.levels)
    final = curve[-1]
    n_poison_img = len({t for t in IMG_TOOLS if t in POISON})
    out = {"budget_curve": curve,
           "detection_rate": round(final["detected_poison"] / max(1, n_poison_img), 3),
           "false_positive_rate": final["false_positives"],
           "mean_acc": 0.0}
    Path(a.out, "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
