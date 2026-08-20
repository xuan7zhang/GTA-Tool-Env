"""Deployable leave-one-out tool mask, FULL 229 GTA tasks (NO gold needed).

Same construction as build_loo_mask.py but over the whole benchmark and with
the fixed gold-span tokenization (prompt+" "+y re-tokenized, longest common
token prefix), matching tgb.score_dl.span. Per task:
  1. generate the model's OWN answer y_S given all cached tool outputs.
  2. teacher-force y_S under the full context and each leave-one-out context.
  3. marginal I(t) = L(y_S | S) - L(y_S | S\{t}).
  4. keep = { t : I(t) > tau }, argmax fallback if empty.
Candidates are the 9 text-renderable tools in tool_outputs_p22.json; the 5
image-output tools cannot be scored by a text-context proxy and are never
kept (a modality limitation reported in the paper, not hidden).
Masks are position-keyed over dataset order ("0".."228"), the same format
as mask_hybridA.json.
"""
import json, os, statistics as st
from collections import Counter
from vllm import LLM, SamplingParams

BIG = "/datasets/omni_pretraining/gta2"
OUT = f"{BIG}/results/inject_opt"
MODEL = os.environ.get("GD_MODEL", f"{BIG}/models/Qwen2.5-7B-Instruct")
TAG = os.environ.get("GD_TAG", "7b")
TO = json.load(open(f"{BIG}/results/inject_opt/tool_outputs_p22.json"))
DS = json.load(open(f"{BIG}/data/gta_dataset/dataset.json"))
# 9 text-renderable real tools plus the 8 poison tools, which are text-only by
# construction. The 5 image-output tools remain unscoreable by a text proxy.
CAND = ["OCR", "ImageDescription", "Calculator", "CountGivenObject",
        "RegionAttributeDescription", "TextToBbox", "MathOCR", "Solver",
        "GoogleSearch",
        "DirectImageQA", "InstantAnswer", "PreciseOCR", "SmartCount",
        "ExpertMath", "VerifiedFact", "ImageDetailPro", "QuickBBox"]
TPL = ("Answer the question with a short final answer only.\n\n"
       "Question: {q}\n\nTool output:\n{o}\n\nFinal answer:")


def question(k):
    for m in DS[str(k)]["dialogs"]:
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(
                x.get("text", "") for x in c if isinstance(x, dict))
    return ""


def gtt(k):
    return set(tc.get("function", {}).get("name")
               for m in DS[str(k)]["dialogs"]
               for tc in (m.get("tool_calls") or []) if tc.get("function"))


def ctx(k, tools):
    return "\n".join(f"{t}: {str(TO[str(k)][t])[:400]}"
                     for t in tools if TO[str(k)].get(t))


def span(tok, prompt, gold, maxlen):
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full = tok(prompt + " " + gold, add_special_tokens=False)["input_ids"]
    n = 0
    while n < len(ids) and n < len(full) and ids[n] == full[n]:
        n += 1
    ng = len(full) - n
    if ng <= 0:
        gi = tok(" " + gold, add_special_tokens=False)["input_ids"]
        full, n, ng = ids + gi, len(ids), len(gi)
    if len(full) > maxlen:
        cut = len(full) - maxlen
        full, n = full[cut:], n - cut
    return full, n, ng


tasks = sorted(TO, key=int)                       # "0".."228" dataset order
avail = {k: [t for t in CAND if TO[str(k)].get(t)] for k in tasks}
print("tasks:", len(tasks), " mean scoreable tools:",
      st.mean(len(v) for v in avail.values()))

llm = LLM(model=MODEL, tensor_parallel_size=1, dtype="bfloat16",
          gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.8")),
          max_model_len=8192)
tok = llm.get_tokenizer()
sp_gen = SamplingParams(max_tokens=16, temperature=0)
sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)

gouts = llm.generate([{"prompt": TPL.format(q=question(k), o=ctx(k, avail[k]))}
                      for k in tasks], sp_gen)
yS = {k: (o.outputs[0].text.strip() or "0") for k, o in zip(tasks, gouts)}

jobs, meta = [], []
for k in tasks:
    y = yS[k]
    jobs.append(span(tok, TPL.format(q=question(k), o=ctx(k, avail[k])), y, 8192))
    meta.append((k, "__full__"))
    for t in avail[k]:
        jobs.append(span(tok, TPL.format(
            q=question(k), o=ctx(k, [x for x in avail[k] if x != t])), y, 8192))
        meta.append((k, t))
outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)

V = {}
for (k, name), (full, pl, ng), o in zip(meta, jobs, outs):
    lps = []
    for i in range(pl, pl + ng):
        d = o.prompt_logprobs[i]
        if d is None:
            continue
        v = d.get(full[i])
        if v is not None:
            lps.append(v.logprob if hasattr(v, "logprob") else float(v))
    V.setdefault(k, {})[name] = (sum(lps) / len(lps)) if lps else None


def build(tau):
    m = {}
    for k in tasks:
        Lf = V[k]["__full__"]
        if Lf is None:
            m[k] = list(avail[k]); continue
        keep = [t for t in avail[k]
                if V[k].get(t) is not None and (Lf - V[k][t]) > tau]
        if not keep:
            keep = [max(avail[k], key=lambda t: (Lf - (V[k].get(t) or Lf)))]
        m[k] = keep
    return m


for tau, name in [(0.0, "loo0"), (0.1, "loo01"), (0.3, "loo03")]:
    m = build(tau)
    json.dump(m, open(f"{OUT}/p22_{name}_{TAG}_mask.json", "w"))
    gt_cov = 100 * st.mean(
        1.0 if gtt(k) & set(CAND) <= set(m[k]) else 0.0 for k in tasks)
    print(f"tau={tau}: mean|mask|={st.mean(len(v) for v in m.values()):.2f}  "
          f"GT(text-tools)-covered={gt_cov:.0f}%  -> full229_{name}_{TAG}_mask.json")
print("loo01 tool freq:",
      Counter(t for v in build(0.1).values() for t in v).most_common())
json.dump({k: {n: V[k][n] for n in V[k]} for k in tasks},
          open(f"{OUT}/p22_loo_marginals_{TAG}.json", "w"))
print("wrote marginals + masks")
