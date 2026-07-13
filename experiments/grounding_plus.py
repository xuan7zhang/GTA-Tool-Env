#!/usr/bin/env python3
"""ours_grounding_plus = input-grounding poison removal  UNION  declutter.

Two deterministic, principled prune signals (no accuracy, no frequency, no GT):
  1. input-grounding (from grounding_selector): tools whose output ignores their
     input -> injected poison. (8/8 poison, 0 real, both regimes.)
  2. declutter: tools whose 200-response is an IMAGE/FILE (base64), not text.
     On answer-type tasks these cannot produce the final text answer and only
     burn turns/context -- the natural distractors call_frequency exploited
     (DrawBox/AddText/Plot/TextToImage/ImageStylization). Identified from the
     openapi response schema, deterministically.

prune = grounding_invariant ∪ image_output_tools ; keep = the rest.
This gives ours the de-cluttering bonus call_frequency got, on top of exact
poison removal, so it should match/beat call_freq on attractive poison while
staying regime-robust (it still never prunes a real perception/logic tool).
"""
import argparse
import json

import requests


def image_output_tools(spec):
    out = []
    for p, ops in spec["paths"].items():
        op = next(iter(ops.values()), {})
        resp = json.dumps(op.get("responses", {}).get("200", {})).lower()
        # agentlego image/file tools declare base64 string or binary content
        if "base64" in resp or "binary" in resp:
            out.append(p.strip("/"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--grounding", required=True, help="selectors_grounding.json")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--injected", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = args.pool.split(",")
    injected = set(args.injected.split(","))
    g = json.load(open(args.grounding))["ours_grounding"]["pruned"]
    spec = requests.get(args.proxy + "/openapi.json", timeout=30).json()
    img = [t for t in image_output_tools(spec) if t in pool]

    prune = sorted(set(g) | set(img), key=pool.index)
    keep = [t for t in pool if t not in prune]

    json.dump({"ours_grounding_plus": {"keep": keep, "pruned": prune},
               "from_grounding": g, "image_output": img},
              open(args.out, "w"), indent=1)
    print(f"grounding poison pruned ({len(g)}): {g}")
    print(f"image-output declutter ({len(img)}): {img}")
    print(f"ours_grounding_plus prunes {len(prune)} -> keeps {len(keep)}: {keep}")
    print(f"  poison caught: {len(set(prune)&injected)}/{len(injected)}; "
          f"real pruned: {sorted(set(prune)-injected)}")


if __name__ == "__main__":
    main()
