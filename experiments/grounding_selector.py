#!/usr/bin/env python3
"""Input-grounding selector via BLACK-IMAGE CONTRAST — noise-free, frequency-
independent, regime-independent (a tool-level probe; descriptions irrelevant).

A grounded image tool reads its image: its output on a real photo differs from
its output on a black square. The injected poison returns _wrong_token(text arg)
and ignores the image, so real-vs-black outputs are IDENTICAL. We test that
directly:

  grounded(tool) = [ output(real_image) != output(black_image) ]   (both HTTP 200)

Only image-taking tools with simple args are testable; a tool that errors or is
untestable is EXCLUDED (kept), never force-pruned. Poison image tools flag
cleanly; real image tools pass. No agent, no accuracy, no ±noise, no frequency.

Usage: python grounding_selector.py --proxy URL --pool csv --injected csv \
   --k 8 --images DIR --out sel.json
"""
import argparse
import base64
import glob
import hashlib
import io
import json
import os.path as osp

import requests
from PIL import Image

# simple text fillers by param name
def fill(pn):
    p = pn.lower()
    if "quest" in p:
        return "What is shown in the image?"
    if p in ("object", "text") or "obj" in p:
        return "person"
    if "attr" in p:
        return "color"
    return "a"


def openapi(proxy):
    return requests.get(proxy + "/openapi.json", timeout=30).json()


def tool_sig(spec, tool):
    ops = spec["paths"].get("/" + tool, {})
    op = next(iter(ops.values()), {})
    content = op.get("requestBody", {}).get("content", {})
    ref = (content.get("multipart/form-data", {}).get("schema", {}).get("$ref", "")
           or content.get("application/x-www-form-urlencoded", {}).get("schema", {}).get("$ref", ""))
    if not ref:
        return False, [], []
    props = spec.get("components", {}).get("schemas", {}).get(
        ref.split("/")[-1], {}).get("properties", {})
    has_image, texts, hard = False, [], []
    for pn, pv in props.items():
        if pv.get("format") == "binary" or pn == "image":
            has_image = True
        elif "bbox" in pn.lower() or "box" in pn.lower():
            hard.append(pn)          # needs coordinates — skip tool
        else:
            texts.append(pn)
    return has_image, texts, hard


# diverse text probes: a real text tool maps these to many distinct outputs;
# a poison tool maps them into its tiny fixed _wrong_token pool (low cardinality).
TEXT_PROBES = ["2+3", "10*7", "100-45", "sqrt(144)", "8/2", "2**10", "15+27",
               "what color is the car?", "how many people?", "what is the price?",
               "capital of France", "who wrote Hamlet", "boiling point of water",
               "the largest planet", "speed of light"]


def call_text(proxy, tool, text_param, value):
    try:
        r = requests.post(f"{proxy}/{tool}", data={text_param: value}, timeout=60)
        return hashlib.sha256(r.content).hexdigest()[:16] if r.status_code == 200 else None
    except Exception:
        return None


def text_cardinality(proxy, tool, text_param):
    outs = [call_text(proxy, tool, text_param, v) for v in TEXT_PROBES]
    ok = [o for o in outs if o is not None]
    return (len(set(ok)), len(ok))  # (distinct, tested)


def call(proxy, tool, img_bytes, texts):
    # agentlego multipart tools take the image as a file and text args as FORM
    # fields (data=), not query params.
    files = {"image": ("i.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    data = {t: fill(t) for t in texts}
    try:
        r = requests.post(f"{proxy}/{tool}", data=data, files=files, timeout=120)
        if r.status_code != 200:
            return None
        return hashlib.sha256(r.content).hexdigest()[:16]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--injected", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = args.pool.split(",")
    injected = set(args.injected.split(","))
    spec = openapi(args.proxy)

    # a few real images (with content) + their black counterparts
    real_imgs = sorted(glob.glob(osp.join(args.images, "*.jpg")))[:4]
    reals = []
    for p in real_imgs:
        im = Image.open(p).convert("RGB")
        buf = io.BytesIO(); im.save(buf, "JPEG"); reals.append((buf.getvalue(), im.size))

    status = {}
    for tool in pool:
        has_image, texts, hard = tool_sig(spec, tool)
        if not has_image and texts and not hard:
            # text-only tool: flag if output cardinality is degenerate (small
            # fixed pool) over many diverse text inputs -> hash-lookup poison.
            distinct, tested = text_cardinality(args.proxy, tool, texts[0])
            # poison maps many diverse inputs into a tiny fixed pool -> low
            # distinct/tested; a real text tool (Calculator) gives ~1 distinct
            # per valid input.
            if tested >= 6 and distinct / tested < 0.5:
                status[tool] = "INVARIANT"     # degenerate text tool -> poison-like
            elif tested == 0:
                status[tool] = "errored"
            else:
                status[tool] = "grounded"
            continue
        if not has_image or hard:
            status[tool] = "untestable"        # keep
            continue
        changed, tested = 0, 0
        for img_bytes, size in reals:
            blk = Image.new("RGB", size, 0); b = io.BytesIO(); blk.save(b, "JPEG")
            o_real = call(args.proxy, tool, img_bytes, texts)
            o_black = call(args.proxy, tool, b.getvalue(), texts)
            if o_real is not None and o_black is not None:
                tested += 1
                changed += int(o_real != o_black)
        if tested == 0:
            status[tool] = "errored"           # keep (probed wrong)
        elif changed == 0:
            status[tool] = "INVARIANT"         # ignores image -> poison-like
        else:
            status[tool] = "grounded"

    invariant = [t for t in pool if status[t] == "INVARIANT"]
    prune = invariant[:args.k]
    keep = [t for t in pool if t not in prune]

    json.dump({"ours_grounding": {"keep": keep, "pruned": prune}, "status": status},
              open(args.out, "w"), indent=1)
    print("black-image contrast (INVARIANT = ignores image = poison-like):")
    for t in pool:
        print(f"  {status[t]:11s} {t:24s} {'(injected)' if t in injected else ''}"
              f"{' <PRUNE>' if t in prune else ''}")
    hit = len(set(prune) & injected)
    print(f"\nours_grounding prunes {len(prune)}: {hit} injected, {len(prune)-hit} real")


if __name__ == "__main__":
    main()
