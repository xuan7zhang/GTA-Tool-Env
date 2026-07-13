#!/usr/bin/env python3
"""Generate Φ (schema-rewrite) toolmeta variants.

Three rewrite families over tool descriptions (inputs/outputs untouched so the
call protocol still works):

  paraphrase  meaning-preserving template paraphrase (deterministic), or
              --rewrites FILE to inject externally generated (e.g. LLM)
              paraphrases from {"tool": "new description"} JSON
  degraded    vague/uninformative descriptions ("A tool that processes input.")
  misleading  seeded derangement: every tool gets another tool's description

Each variant produces variants/phi/<variant_id>/{toolmeta.json, meta.json}.
Point GTA_TOOLMETA (and OPENCOMPASS_TOOLMETA_PATH if used) at the variant file.
"""
import argparse
import json
import random
from pathlib import Path

VAGUE = "A general-purpose tool that processes the given input and returns a result."

PARA_TEMPLATES = [
    ("This tool can", "Use this utility to"),
    ("The tool can", "This capability lets the agent"),
    ("A tool that", "A utility which"),
    ("input image", "provided picture"),
    ("the image", "the picture"),
    ("recognize", "identify"),
    ("count", "tally"),
    ("generate", "produce"),
    ("search", "look up"),
    ("calculate", "compute"),
    ("description", "caption"),
]


def paraphrase(desc: str) -> str:
    out = desc
    for a, b in PARA_TEMPLATES:
        out = out.replace(a, b)
    if out == desc:  # ensure surface change even if no template hit
        out = "Utility: " + desc[0].lower() + desc[1:] if desc else desc
    return out


def derange(keys, rng):
    """Seeded derangement (no fixed point) of keys."""
    while True:
        perm = keys[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(keys, perm)):
            return dict(zip(keys, perm))


def write_variant(out_root: Path, variant_id: str, meta: dict, params: dict):
    d = out_root / variant_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "toolmeta.json").write_text(json.dumps(meta, indent=1))
    (d / "meta.json").write_text(json.dumps(
        {"variant_id": variant_id, "family": "phi", **params}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolmeta", required=True)
    ap.add_argument("--out", default="variants/phi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rewrites", help="optional JSON {tool: paraphrased_desc} "
                                       "(e.g. LLM-generated) used for 'paraphrase'")
    args = ap.parse_args()

    base = json.load(open(args.toolmeta))
    out = Path(args.out)

    # paraphrase
    para = json.loads(json.dumps(base))
    ext = json.load(open(args.rewrites)) if args.rewrites else {}
    for k, v in para.items():
        v["description"] = ext.get(k) or paraphrase(v["description"] or "")
    write_variant(out, f"paraphrase_s{args.seed}", para,
                  {"kind": "paraphrase", "seed": args.seed, "external": bool(ext)})

    # degraded
    deg = json.loads(json.dumps(base))
    for v in deg.values():
        v["description"] = VAGUE
    write_variant(out, "degraded", deg, {"kind": "degraded", "seed": None})

    # misleading (description derangement)
    mis = json.loads(json.dumps(base))
    mapping = derange(list(mis.keys()), random.Random(args.seed))
    for k in mis:
        mis[k]["description"] = base[mapping[k]]["description"]
    write_variant(out, f"misleading_s{args.seed}", mis,
                  {"kind": "misleading", "seed": args.seed, "derangement": mapping})

    print(f"wrote 3 phi variants under {out}")


if __name__ == "__main__":
    main()
