#!/usr/bin/env python3
"""Generate mask (m) environment variants: toollist + pruned toolmeta files.

Variants (all seeded, never modifying the source dataset files):
  - leave-one-out over every tool in the pool
  - random masks keeping |m| in {25%, 50%, 75%} of the pool, N seeds each

Each variant k produces:
  variants/masks/<variant_id>/toollist.txt   (for agentlego-server, or proxy mask)
  variants/masks/<variant_id>/toolmeta.json  (pruned; point GTA_TOOLMETA here)
  variants/masks/<variant_id>/meta.json      (variant params for aggregation)

Usage:
  python gen_masks.py --toolmeta .../gta_dataset/toolmeta.json \
      --out variants/masks --fracs 0.25 0.5 0.75 --seeds 0 1 2
"""
import argparse
import json
import random
from pathlib import Path


def write_variant(out_root: Path, variant_id: str, kept: list, full_meta: dict, params: dict):
    d = out_root / variant_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "toollist.txt").write_text("\n".join(kept) + "\n")
    pruned = {k: v for k, v in full_meta.items() if k in kept}
    (d / "toolmeta.json").write_text(json.dumps(pruned, indent=1))
    (d / "meta.json").write_text(json.dumps({
        "variant_id": variant_id, "family": "mask", "kept": kept,
        "dropped": [t for t in full_meta if t not in kept], **params}, indent=1))
    return variant_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolmeta", required=True)
    ap.add_argument("--out", default="variants/masks")
    ap.add_argument("--fracs", nargs="*", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    meta = json.load(open(args.toolmeta))
    pool = list(meta.keys())
    out = Path(args.out)
    made = []

    # leave-one-out
    for t in pool:
        kept = [x for x in pool if x != t]
        made.append(write_variant(out, f"loo_{t}", kept, meta,
                                  {"kind": "loo", "left_out": t, "seed": None}))
    # random masks
    for frac in args.fracs:
        k = max(1, round(len(pool) * frac))
        for seed in args.seeds:
            rng = random.Random(f"{frac}:{seed}")
            kept = sorted(rng.sample(pool, k))
            made.append(write_variant(out, f"rand{int(frac*100)}_s{seed}", kept, meta,
                                      {"kind": "random", "frac": frac, "seed": seed}))
    # full pool (identity mask, for controls)
    made.append(write_variant(out, "full", pool, meta, {"kind": "full", "seed": None}))
    print(f"wrote {len(made)} mask variants under {out}")
    for m in made:
        print(" ", m)


if __name__ == "__main__":
    main()
