"""Score every family's chain on every task, so the selector's candidate set is
constructible without the gold.

Why this exists. The v2 comparison "per-task dL vs greedy search" was not
apples-to-apples: dL chose among conditions that are *defined relative to each
task's ground truth* (`useful` is that task's own chain, `partial_1of2` and
`wrong` are derived from it), while greedy may only pick one global mask. Even
though the selector reads nothing but dL, the candidate *set* was built with
the answer in hand.

The deployable candidate set needs no gold: enumerate the tool chains the
environment is known to support -- one per family -- plus the union and the
empty mask. Every task gets the same ten options, and per-task selection then
means exactly what it should mean: *which of the known chains fits this task*.

    GD_TAG=7b GD_MODEL=... python -m tgb.add_chains_v2

Adds `chain_<family>` conditions to tgb_tasks.json and to tgb_scored_<tag>.json.
"""
import argparse
import collections
import json
import os
import random
import tempfile
import zlib

from vllm import LLM, SamplingParams

from . import scenes, tools_v2
from .families_v2 import FAMILIES_V2
from .generate import TPL, context
from .score_dl import correct
from .tools import run_chain

tools_v2.register()

# the chain each family runs, keyed by family -- known to the environment
# designer, not to the gold of any particular task
CHAINS = {
    "f1_extract_compute": ["OCR", "Calculator"],
    "f2_visual_reason": ["CountGivenObject", "OCR", "Calculator"],
    "f3_retrieve_reason": ["OCR", "GoogleSearch", "Calculator"],
    "gauge_over": ["OCR", "UnitConvert"],
    "temp_over": ["OCR", "TempConvert"],
    "equation_short": ["OCR", "Solver"],
    "timetable_over": ["OCR", "DurationCalc"],
    "invoice_currency": ["OCR", "CurrencyConvert"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb2")
    ap.add_argument("--v1", action="store_true",
                    help="score TGB-v1's three chains instead, so the v1 and "
                         "v2 answers rest on the same gold-free protocol")
    a = ap.parse_args()
    if a.v1:
        CHAINS.clear()
        CHAINS.update({
            "f1_extract_compute": ["OCR", "Calculator"],
            "f2_visual_reason": ["CountGivenObject", "OCR", "Calculator"],
            "f3_retrieve_reason": ["OCR", "GoogleSearch", "Calculator"],
        })
    tag = os.environ.get("GD_TAG", "7b")
    tpath = os.path.join(a.dir, "tgb_tasks.json")
    tasks = json.load(open(tpath))
    tmp = tempfile.mkdtemp(prefix="tgb2_chains_")
    if not a.v1:
        assert set(CHAINS) == set(FAMILIES_V2), "chain table out of sync"

    names = [f"chain_{f}" for f in CHAINS]
    ctxs = {}
    for t in tasks:
        boxes = scenes.render(t["scene"], os.path.join(tmp, "s.png"))
        for fam, tools in CHAINS.items():
            nm = f"chain_{fam}"
            rng = random.Random(zlib.crc32(f"{t['id']}/{nm}".encode()))
            outs, trace = run_chain(t, t["scene"], boxes, set(tools), rng,
                                    corrupt_tools=())
            c = context(outs, [s["tool"] for s in t["plan"]])
            ctxs[(t["id"], nm)] = c
            t["conditions"][nm] = dict(tools=tools, label="chain", corrupt=[],
                                       context=c, outputs=outs, trace=trace)
    json.dump(tasks, open(tpath, "w"), indent=1)
    print(f"added {len(names)} chain conditions to {len(tasks)} tasks")

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sp_gen = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                            temperature=0)

    prompts, meta = [], []
    for t in tasks:
        for nm in names:
            prompts.append(TPL.format(q=t["question"], o=ctxs[(t["id"], nm)]))
            meta.append((t["id"], nm, t["gold"]))

    jobs = []
    for p, (_, _, g) in zip(prompts, meta):
        gi = tok(g, add_special_tokens=False)["input_ids"]
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
    for (tid, nm, g), ll, go in zip(meta, LL, gouts):
        r = by[tid]
        base = r["conds"]["none"]["L"]
        r["conds"][nm] = dict(
            L=ll, acc=correct(go.outputs[0].text, g),
            gen=go.outputs[0].text.strip()[:80],
            dL=(ll - base) if (ll is not None and base is not None) else None)
        r["n_tools"][nm] = len(CHAINS[nm[len("chain_"):]])
    json.dump(rows, open(spath, "w"), indent=1)

    acc = collections.defaultdict(list)
    for t in tasks:
        for nm in names:
            acc[nm].append(by[t["id"]]["conds"][nm]["acc"])
    print(f"[{tag}] accuracy of each fixed chain over all {len(tasks)} tasks")
    for nm in names:
        print(f"    {nm:<32} {sum(acc[nm])/len(acc[nm]):.3f}")


if __name__ == "__main__":
    main()
