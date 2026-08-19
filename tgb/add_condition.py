"""Add one extra executed condition to an existing scored file.

Needed because the per-task selector must be offered the same options the
global search can reach. The greedy environment search returned the union mask
{OCR, Calculator, CountGivenObject, GoogleSearch}; that set is not any task's
`useful`, `full`, or degraded variant, so the closed-loop comparison was not
over a common action space until it is added.

    GD_TAG=7b python -m tgb.add_condition --name union \
        --tools OCR,Calculator,CountGivenObject,GoogleSearch

Writes the new condition into `tgb_tasks.json` (context + outputs) and into
`tgb_scored_<tag>.json` (L, dL, acc), leaving everything else untouched.
"""
import argparse
import json
import os
import random
import zlib

from vllm import LLM, SamplingParams

from . import scenes
from .generate import TPL, context
from .score_dl import correct
from .tools import run_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tools", required=True)
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")
    tools = a.tools.split(",")
    tpath = os.path.join(a.dir, "tgb_tasks.json")
    tasks = json.load(open(tpath))

    import tempfile
    tmp = tempfile.mkdtemp(prefix="tgb_add_")
    ctxs = {}
    for t in tasks:
        boxes = scenes.render(t["scene"], os.path.join(tmp, "s.png"))
        rng = random.Random(zlib.crc32(f"{t['id']}/{a.name}".encode()))
        outs, trace = run_chain(t, t["scene"], boxes, set(tools), rng,
                                corrupt_tools=())
        ctx = context(outs, [s["tool"] for s in t["plan"]])
        ctxs[t["id"]] = ctx
        t["conditions"][a.name] = dict(tools=tools, label=a.name, corrupt=[],
                                       context=ctx, outputs=outs, trace=trace)
    json.dump(tasks, open(tpath, "w"), indent=1)
    print(f"added condition '{a.name}' to {len(tasks)} tasks in {tpath}")

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sp_gen = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                            temperature=0)
    prompts = [TPL.format(q=t["question"], o=ctxs[t["id"]]) for t in tasks]

    jobs = []
    for p, t in zip(prompts, tasks):
        gi = tok(t["gold"], add_special_tokens=False)["input_ids"]
        pi = tok(p, add_special_tokens=False)["input_ids"][:4096 - len(gi) - 1]
        jobs.append((pi + gi, len(pi), len(gi)))
    outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
    LL = []
    for (full, plen, ng), o in zip(jobs, outs):
        lps = []
        for k in range(plen, plen + ng):
            d = o.prompt_logprobs[k]
            if d is None:
                continue
            lp = d.get(full[k])
            if lp is not None:
                lps.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
        LL.append(sum(lps) / len(lps) if lps else None)
    gouts = llm.generate([{"prompt": p} for p in prompts], sp_gen)

    spath = os.path.join(a.dir, f"tgb_scored_{tag}.json")
    rows = json.load(open(spath))
    by = {r["id"]: r for r in rows}
    for t, ll, go in zip(tasks, LL, gouts):
        r = by[t["id"]]
        base = r["conds"]["none"]["L"]
        r["conds"][a.name] = dict(
            L=ll, acc=correct(go.outputs[0].text, t["gold"]),
            gen=go.outputs[0].text.strip()[:80],
            dL=(ll - base) if (ll is not None and base is not None) else None)
        r["n_tools"][a.name] = len(tools)
    json.dump(rows, open(spath, "w"), indent=1)
    acc = sum(by[t["id"]]["conds"][a.name]["acc"] for t in tasks) / len(tasks)
    print(f"[{tag}] '{a.name}' acc over all {len(tasks)} tasks = {acc:.3f}")


if __name__ == "__main__":
    main()
