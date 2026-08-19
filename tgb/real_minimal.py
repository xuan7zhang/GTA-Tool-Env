"""The minimal viable experiment, on the REAL GTA tasks.

Everything TGB establishes is on synthetic data. This runs the same protocol on
the 156 scorable GTA-Atomic tasks, whose tool outputs are already cached, and
answers the question the synthetic work cannot: **does gold likelihood track
real accuracy gains on real tool outputs?**

Three environments per task, fixed model / prompt / temperature:

    E_none      no tool output
    E_OCR       OCR only
    E_all       every cached text tool

recorded as token-normalized gold log-likelihood and binary accuracy, then

    dL(x,E) = L(x,E) - L(x,E_none)
    dA(x,E) = A(x,E) - A(x,E_none)

Three levels of correlation are reported, because they can disagree:
  global    over all (task, environment) pairs
  within    per task, does the dL ordering of environments match the dA ordering
  gain      does dL > 0 predict dA > 0, and with what precision

Unlike TGB, `E_none` here is *not* uniformly wrong -- real GTA has tasks the
model answers from the question alone -- so dA is a genuine three-valued
quantity and the correlation is not degenerate.

Finally each task gets a response profile

    z_x = [A_none, A_OCR, A_all, dL_OCR, dL_all]

which is clustered into the task types the meeting plan asks for
(tool-free / OCR-dependent / tool-helped / tool-harmed / belief-behaviour
mismatch), so the categories come out of measurement rather than hand labels.

    GD_TAG=7b GD_MODEL=... python -m tgb.real_minimal
"""
import argparse
import collections
import json
import os
import re
import statistics as st

from vllm import LLM, SamplingParams

BIG = "/datasets/omni_pretraining/gta2"
TEXT_TOOLS = ["OCR", "ImageDescription", "CountGivenObject", "TextToBbox",
              "RegionAttributeDescription", "GoogleSearch", "Calculator",
              "Solver", "MathOCR"]
TPL = ("Answer the question with a short final answer only.\n\n"
       "Question: {q}\n\nTool output:\n{o}\n\nFinal answer:")


def gold_of(ds, k):
    ga = ds[str(k)].get("gt_answer")
    if not isinstance(ga, dict) or not ga.get("whitelist"):
        return None
    g = ga["whitelist"][0]
    return str(g[0] if isinstance(g, list) else g).strip()


def question_of(ds, k):
    for m in ds[str(k)]["dialogs"]:
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(
                x.get("text", "") for x in c if isinstance(x, dict))
    return ""


def gt_tools(ds, k):
    return {tc.get("function", {}).get("name")
            for m in ds[str(k)]["dialogs"]
            for tc in (m.get("tool_calls") or []) if tc.get("function")}


def correct(text, gold):
    g = re.escape(gold.strip().lower())
    return int(re.search(g, text.lower().replace(",", "")) is not None)


def spearman(x, y):
    def rank(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
                j += 1
            av = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[o[k]] = av
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** .5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{BIG}/results/taco/real_minimal")
    ap.add_argument("--cap", type=int, default=400,
                    help="chars of each tool output kept, as in the §7 scorer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = os.environ.get("GD_TAG", "7b")
    TO = json.load(open(f"{BIG}/results/inject_opt/tool_outputs.json"))
    DS = json.load(open(f"{BIG}/data/gta_dataset/dataset.json"))

    tasks = [k for k in TO if gold_of(DS, k)]
    print(f"[{tag}] {len(tasks)} scorable GTA tasks")

    def ctx(k, tools):
        return "\n".join(f"{t}: {str(TO[k][t])[:a.cap]}" for t in tools
                         if TO[k].get(t))

    ENVS = {"none": [], "OCR": ["OCR"], "all": TEXT_TOOLS}

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=8192)
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sp_gen = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                            temperature=0)

    prompts, meta = [], []
    for k in tasks:
        q = question_of(DS, k)
        for e, tools in ENVS.items():
            prompts.append(TPL.format(q=q, o=ctx(k, tools)))
            meta.append((k, e, gold_of(DS, k)))

    jobs = []
    for p, (_, _, g) in zip(prompts, meta):
        gi = tok(g, add_special_tokens=False)["input_ids"]
        pi = tok(p, add_special_tokens=False)["input_ids"][:8192 - len(gi) - 1]
        jobs.append((pi + gi, len(pi), len(gi)))
    outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
    LL = []
    for (full, plen, ng), o in zip(jobs, outs):
        lps = []
        for i in range(plen, plen + ng):
            d = o.prompt_logprobs[i]
            if d is None:
                continue
            lp = d.get(full[i])
            if lp is not None:
                lps.append(lp.logprob if hasattr(lp, "logprob") else float(lp))
        LL.append(sum(lps) / len(lps) if lps else None)
    gouts = llm.generate([{"prompt": p} for p in prompts], sp_gen)

    V = collections.defaultdict(dict)
    for (k, e, g), ll, go in zip(meta, LL, gouts):
        V[k][e] = dict(L=ll, acc=correct(go.outputs[0].text, g),
                       gen=go.outputs[0].text.strip()[:60])
    rows = []
    for k in tasks:
        d = V[k]
        if d["none"]["L"] is None:
            continue
        rows.append(dict(
            task=k, gold=gold_of(DS, k), gt_tools=sorted(gt_tools(DS, k)),
            A_none=d["none"]["acc"], A_OCR=d["OCR"]["acc"], A_all=d["all"]["acc"],
            dL_OCR=(d["OCR"]["L"] - d["none"]["L"]),
            dL_all=(d["all"]["L"] - d["none"]["L"]),
            dA_OCR=d["OCR"]["acc"] - d["none"]["acc"],
            dA_all=d["all"]["acc"] - d["none"]["acc"]))
    json.dump(rows, open(os.path.join(a.out, f"real_minimal_{tag}.json"), "w"),
              indent=1)

    print(f"\n[{tag}] accuracy by environment (n={len(rows)})")
    for e, key in [("none", "A_none"), ("OCR", "A_OCR"), ("all", "A_all")]:
        print(f"    E_{e:<5} {st.mean([r[key] for r in rows]):.3f}")

    pairs = [(r[f"dL_{e}"], r[f"dA_{e}"]) for r in rows for e in ("OCR", "all")]
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    print(f"\n[{tag}] correlation, {len(pairs)} task-environment pairs")
    print(f"    global   spearman(dL, dA)      {spearman(x, y):+.3f}")
    pos = [b for aa, b in pairs if aa > 0]
    neg = [b for aa, b in pairs if aa <= 0]
    print(f"    gain     P(dA>0 | dL>0)        "
          f"{st.mean([1 if b > 0 else 0 for b in pos]):.3f}  (n={len(pos)})")
    print(f"             P(dA>0 | dL<=0)       "
          f"{st.mean([1 if b > 0 else 0 for b in neg]):.3f}  (n={len(neg)})")
    print(f"             P(dA<0 | dL<0)        "
          f"{st.mean([1 if b < 0 else 0 for b in [c for aa, c in pairs if aa < 0]]):.3f}")
    # within-task: does the dL ordering of the three envs match the dA ordering
    agree = tie = 0
    for r in rows:
        dl = {"none": 0.0, "OCR": r["dL_OCR"], "all": r["dL_all"]}
        da = {"none": 0, "OCR": r["dA_OCR"], "all": r["dA_all"]}
        best_l = max(dl, key=dl.get)
        if len(set(da.values())) == 1:
            tie += 1
        elif da[best_l] == max(da.values()):
            agree += 1
    scored = len(rows) - tie
    print(f"    within   argmax dL is also argmax dA: "
          f"{agree}/{scored} = {agree / max(1, scored):.3f}   "
          f"({tie} tasks where every environment ties)")

    # ---- response-profile clustering, rule-based on the same z_x
    def label(r):
        if r["A_none"] == 1 and r["A_all"] == 1:
            return "tool-free (correct without tools)"
        if r["A_none"] == 0 and r["A_OCR"] == 1:
            return "OCR-dependent"
        if r["A_none"] == 0 and r["A_all"] == 1 and r["A_OCR"] == 0:
            return "needs tools beyond OCR"
        if r["A_none"] == 1 and r["A_all"] == 0:
            return "tool-harmed"
        if max(r["dL_OCR"], r["dL_all"]) > 0.5 and r["A_all"] == 0 \
                and r["A_OCR"] == 0:
            return "belief-behaviour mismatch (dL up, still wrong)"
        return "tool-insensitive (wrong everywhere)"
    cl = collections.Counter(label(r) for r in rows)
    print(f"\n[{tag}] task types from the measured response profile")
    for nm, c in cl.most_common():
        sub = [r for r in rows if label(r) == nm]
        print(f"    {nm:<46} {c:>4}   mean dL_all {st.mean([r['dL_all'] for r in sub]):+.2f}")
    for r in rows:
        r["cluster"] = label(r)
    json.dump(rows, open(os.path.join(a.out, f"real_minimal_{tag}.json"), "w"),
              indent=1)
    print(f"wrote {a.out}/real_minimal_{tag}.json")


if __name__ == "__main__":
    main()
