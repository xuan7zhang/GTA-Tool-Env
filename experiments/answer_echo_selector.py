#!/usr/bin/env python3
"""Answer-echo tool selector — a noise-free behavioral signal for poison tools.

Injected poison tools are authoritative-wrong answer providers: they return a
confident final-answer-shaped string that the agent tends to SUBMIT verbatim.
Real GTA tools are perception primitives (OCR text, counts, bboxes, captions)
whose output the agent processes rather than echoes as the final answer.

Signal (computed from the degraded trajectory only — NO extra evals, so it does
not inherit the +-subset accuracy-noise that sank corruption-attribution):
  echo(tool) = mean over the tool's calls of token-overlap(final_answer,
               tool_output) = |A intersect T| / |T| on content word sets.
High echo -> the agent submits this tool's output -> poison-like -> prune.
"""
import argparse
import glob
import json
import os.path as osp
import re
from collections import defaultdict

STOP = set("the a an of to in on at is are was were be been and or for with "
           "as by from this that it its his her their our your my no not "
           "there here what which who how many much number count image based "
           "answer result verified precise analysis final".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", str(s).lower()) if w not in STOP and len(w) > 1}


def final_answer(rec):
    pred = rec.get("prediction") or []
    rounds = pred if (pred and isinstance(pred[0], list)) else [pred]
    last = rounds[0][-1] if rounds[0] else None
    if isinstance(last, dict) and "tool_calls" not in last and last.get("role") == "assistant":
        return last.get("content") or ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degraded-run", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--injected", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = args.pool.split(",")
    injected = set(args.injected.split(","))
    f = sorted(glob.glob(osp.join(args.degraded_run, "*", "predictions", "*", "gta_bench_end.json")))
    data = json.load(open(f[-1]))

    echo_sum, echo_n = defaultdict(float), defaultdict(int)
    for k, v in data.items():
        if not k.isdigit():
            continue
        ans = toks(final_answer(v))
        pred = v.get("prediction") or []
        rounds = pred if (pred and isinstance(pred[0], list)) else [pred]
        steps = rounds[0] or []
        for i, step in enumerate(steps):
            if isinstance(step, dict) and "tool_calls" in step:
                nm = step["tool_calls"][0]["function"]["name"]
                out = ""
                if i + 1 < len(steps) and steps[i + 1].get("role") == "tool":
                    c = steps[i + 1].get("content")
                    out = c.get("content") if isinstance(c, dict) else c
                ot = toks(out)
                if ot:
                    echo_sum[nm] += len(ans & ot) / len(ot)
                    echo_n[nm] += 1

    echo = {t: (echo_sum[t] / echo_n[t]) if echo_n[t] else 0.0 for t in pool}
    ranked = sorted(pool, key=lambda t: -echo[t])
    prune = ranked[:args.k]
    keep = [t for t in pool if t not in prune]

    json.dump({"ours_echo": {"keep": keep, "pruned": prune},
               "echo": echo}, open(args.out, "w"), indent=1)

    print("echo ranking (high=poison-like, prune top-k):")
    for t in ranked:
        print(f"  {echo[t]:.3f}  {t:22s} {'(injected)' if t in injected else ''}")
    hit = len(set(prune) & injected)
    print(f"\nours_echo prunes {args.k}: {hit}/{len(injected)} poison caught")
    print(f"  pruned: {prune}")


if __name__ == "__main__":
    main()
