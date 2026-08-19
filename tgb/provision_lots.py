"""Full LOTS loop on TGB: provisioning cold-start, then attribution refinement.

TGB is the one testbed where both operating points run on the same data: a
family is a task with traffic (train questions) and the tools carry docs
(their registry docstrings), so the provisioning form can select a family
set before anything executes; the attribution form (existing self_S LOO
masks) then refines per instance inside it.

Arms evaluated on the 3,600 test tasks:
  prov_K{3,5,7}   family set = top-K tools by family-mean doc-conditioned
                  question likelihood (provisioning, cold, no executions)
  loop_K7         provisioning K=7 menu, then per-instance self_S LOO mask
                  intersected with it (argmax fallback) -- the full loop
  (references printed: none / full / union / attribution-only from stored
   artifacts)

    GD_TAG=7b GD_MODEL=... python -m tgb.provision_lots
"""
import collections
import json
import os
import random
import statistics as st
import tempfile
import zlib

from vllm import LLM, SamplingParams

from . import scenes, tools_v2, tools_v4
from .coalitions_v4 import FULL_MENU_V4
from .generate import TPL, context
from .score_dl import correct, span
from .tools import TOOLS, run_chain

D = "/datasets/omni_pretraining/gta2/results/taco/tgb4"


def doc_of(name):
    fn = TOOLS.get(name)
    doc = (fn.__doc__ or "").strip().split("\n")[0] if fn else ""
    return f"{name}: {doc}" if doc else name


def main():
    tag = os.environ.get("GD_TAG", "7b")
    tools_v2.register()
    tools_v4.register()
    MENU = list(FULL_MENU_V4)

    tasks = json.load(open(os.path.join(D, "tgb_tasks.json")))
    step = max(1, len(tasks) // 400)
    tr_ids = {t["id"] for t in tasks[::step][:400]}
    train = [t for t in tasks if t["id"] in tr_ids]
    test = [t for t in tasks if t["id"] not in tr_ids]
    fam = lambda tid: tid.rsplit("_", 1)[0]
    tmp = tempfile.mkdtemp(prefix="tgb_prov_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    llm = LLM(model=os.environ["GD_MODEL"], tensor_parallel_size=1,
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sp_gen = SamplingParams(max_tokens=64, temperature=0)

    # ---- provisioning scores: question likelihood under each tool's doc ----
    docs = {n: doc_of(n) for n in MENU}
    jobs, meta = [], []
    for t in train:
        q = t["question"]
        jobs.append(span(tok, "Answer the question.\n\nQuestion:", " " + q, 4096))
        meta.append((t["id"], "__none__"))
        for n in MENU:
            pre = f"Available tool:\n{docs[n]}\n\nQuestion:"
            jobs.append(span(tok, pre, " " + q, 4096))
            meta.append((t["id"], n))
    outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
    L = {}
    for (tid, name), (full, pl, ng), o in zip(meta, jobs, outs):
        lps = []
        for i in range(pl, pl + ng):
            d = o.prompt_logprobs[i]
            if d is None:
                continue
            v = d.get(full[i])
            if v is not None:
                lps.append(v.logprob if hasattr(v, "logprob") else float(v))
        L.setdefault(tid, {})[name] = (sum(lps) / len(lps)) if lps else None

    byfam = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in train:
        base = L[t["id"]].get("__none__")
        if base is None:
            continue
        for n in MENU:
            v = L[t["id"]].get(n)
            if v is not None:
                byfam[fam(t["id"])][n].append(v - base)
    prov = {}
    for f, d in byfam.items():
        rank = sorted(d, key=lambda n: -st.mean(d[n]))
        prov[f] = {K: sorted(rank[:K]) for K in (3, 5, 7)}
        print(f"[{tag}] {f}: top7 = {rank[:7]}", flush=True)

    # ---- attribution masks from the stored self_S sweep (tau*) ----
    loo = json.load(open(os.path.join(D, f"tgb_loo_{tag}_self_S.json")))
    sw = {k: v for k, v in loo["sweep"].items() if k != "S"}
    ts = max(sw, key=lambda k: sw[k]["train"])
    inst_mask = sw[ts]["masks"]
    print(f"[{tag}] attribution masks at tau*={ts}", flush=True)

    # ---- evaluation ----
    def ctx_of(t, mask):
        rng = random.Random(zlib.crc32(
            f"{t['id']}/{'+'.join(sorted(mask))}".encode()))
        outs, _ = run_chain(t, t["scene"], t["boxes"], set(mask), rng,
                            corrupt_tools=())
        return context(outs, [s["tool"] for s in t["plan"]])

    def mask_for(t, arm):
        f = fam(t["id"])
        if arm.startswith("prov_K"):
            return prov[f][int(arm[-1])]
        if arm == "loop_K7":
            base = set(prov[f][7])
            m = set(inst_mask.get(t["id"], [])) & base
            return sorted(m) if m else sorted(base)
        raise KeyError(arm)

    arms = ["prov_K3", "prov_K5", "prov_K7", "loop_K7"]
    prompts, meta2 = [], []
    for arm in arms:
        for t in test:
            prompts.append(TPL.format(q=t["question"], o=ctx_of(t, mask_for(t, arm))))
            meta2.append((arm, t["id"], t["gold"]))
    gouts = llm.generate([{"prompt": p} for p in prompts], sp_gen)
    acc = collections.defaultdict(list)
    ntools = collections.defaultdict(list)
    for (arm, tid, g), o in zip(meta2, gouts):
        acc[arm].append(correct(o.outputs[0].text, g))
    for arm in arms:
        for t in test:
            ntools[arm].append(len(mask_for(t, arm)))

    print(f"\n[{tag}] TGB full-loop results (test n={len(test)})")
    for arm in arms:
        print(f"  {arm:<10} acc {st.mean(acc[arm]):.3f}   tools {st.mean(ntools[arm]):.2f}")
    out = {"tag": tag, "tau_star": ts,
           "prov": {f: prov[f] for f in prov},
           "acc": {a: st.mean(acc[a]) for a in arms},
           "tools": {a: st.mean(ntools[a]) for a in arms}}
    json.dump(out, open(os.path.join(D, f"tgb_provision_{tag}.json"), "w"), indent=1)
    print("wrote", os.path.join(D, f"tgb_provision_{tag}.json"))


if __name__ == "__main__":
    main()
