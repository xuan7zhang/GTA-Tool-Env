"""OpRoute -- unsupervised tool-set selection by operation decomposition and
typed dataflow search, evaluated on TGB-v2.

The question this answers is not "which chain matches this task" -- a
hand-written unit-keyword router already routes 75% of TGB-v2 correctly and
still scores *below* the do-nothing union, because in this environment the
binding constraint is per-task solvability, not routing. OpRoute is the
strongest available test of the opposite hypothesis: that a *structural*
selector, whose discriminative signal is operation-level rather than
task-level and whose composition is symbolic rather than model-judged, can
convert routing accuracy into task accuracy.

Pipeline

    x --g--> operations O(x) --match--> margin matrix M --typed search--> S*(x)

1. `decompose`  one generation per task. The model abstracts the question into
   atomic operations with type signatures, under an explicit instruction not to
   reproduce surface words (unit names, currency names) -- delexicalization is
   inside the method, not left to a robustness ablation. Never sees the answer,
   never sees the tool menu.

2. `margins`    for each (operation, tool) a binary Yes/No margin

       m_it = log p(Yes | o_i, c_t) - log p(No | o_i, c_t)

   calibrated by a content-free baseline b_t obtained by replacing the
   operation with "perform an unspecified operation":  m~_it = m_it - b_t.
   One extra forward per tool, no log p(c_t) estimate. Cached by
   (operation text, tool), so repeated operations across tasks are free.

3. `search`     S* = argmax_S  sum_i max_{t in S} m~_it  -  lambda |S|
   subject to a hard typed-dataflow constraint: every operation must be
   coverable by some tool in S whose declared input/output types admit it, in
   an order that is reachable from the available inputs and ends at the target
   type. Because the objective decomposes per operation, the optimum is a union
   of per-operation assignments, so the search enumerates assignments rather
   than the 2^20 subsets.

lambda is fitted on a train split and reported on held-out tasks, so the
headline number is not a sweep maximum.

    GD_TAG=7b GD_MODEL=... python -m tgb.oproute --n 1200
"""
import argparse
import collections
import itertools
import json
import os
import random
import re
import statistics as st
import tempfile
import zlib

from vllm import LLM, SamplingParams

from . import scenes, tools_v2
from .add_chains_v2 import CHAINS
from .families_v2 import NUISANCE, UNION_TOOLS
from .generate import TPL, context
from .score_dl import correct
from .tools import IMAGE_TOOLS, run_chain

tools_v2.register()

MENU = UNION_TOOLS + NUISANCE + IMAGE_TOOLS
SEP = "\x1f"

# ---------------------------------------------------------------- tool cards
# Capability statements are what a tool registry ships: what the tool does, not
# which task it belongs to. Type signatures are coarse on purpose -- they carry
# *bridging* structure (only some tools consume an image) and leave the choice
# among same-signature tools to the margin, which is exactly the division of
# labour the method claims.
CARD = {
    "OCR": ("Reads the text printed in an image and returns the transcribed "
            "lines together with any figures they contain.",
            ["image"], ["text", "number"]),
    "CountGivenObject": ("Counts how many instances of a named object appear "
                         "in an image and returns the count.",
                         ["image"], ["number"]),
    "GoogleSearch": ("Looks up a text query in an external web index and "
                     "returns matching records with their listed values.",
                     ["text"], ["text", "number"]),
    "KnowledgeBase": ("Looks up a text query in an internal product "
                      "catalogue and returns matching records with their "
                      "listed values.", ["text"], ["text", "number"]),
    "Calculator": ("Evaluates an arithmetic expression over numbers and "
                   "returns its numeric value.", ["number"], ["number"]),
    "UnitConvert": ("Converts a quantity between two measurement systems of "
                    "length and returns the converted magnitude.",
                    ["number"], ["number"]),
    "TempConvert": ("Converts a temperature reading between two temperature "
                    "scales and returns the converted magnitude.",
                    ["number"], ["number"]),
    "CurrencyConvert": ("Converts a monetary amount into another currency at "
                        "a given conversion rate and returns the converted "
                        "amount.", ["number"], ["number"]),
    "DurationCalc": ("Computes the elapsed time between two clock readings "
                     "and returns it as a number of minutes.",
                     ["number"], ["number"]),
    "Solver": ("Solves a linear equation for its unknown given its "
               "coefficients and returns the root.", ["number"], ["number"]),
    "ImageDescription": ("Describes the visual content of an image in free "
                         "text.", ["image"], ["text"]),
    "TextToBbox": ("Locates a described object in an image and returns its "
                   "bounding box coordinates.", ["image", "text"], ["text"]),
    "Summarize": ("Restates a passage of text more briefly.",
                  ["text"], ["text"]),
    "Translate": ("Rewrites a passage of text in another language.",
                  ["text"], ["text"]),
    "Barcode": ("Decodes a barcode present in an image into its identifier "
                "string.", ["image"], ["text"]),
    "TextToImage": ("Generates an image from a textual description.",
                    ["text"], ["image"]),
    "DrawBox": ("Draws a rectangle onto an image and returns the edited "
                "image.", ["image"], ["image"]),
    "AddText": ("Writes a caption onto an image and returns the edited "
                "image.", ["image"], ["image"]),
    "ImageStylization": ("Restyles an image and returns the edited image.",
                         ["image"], ["image"]),
    "Plot": ("Renders a chart from data and returns it as an image.",
             ["text"], ["image"]),
}
assert set(CARD) >= set(MENU), sorted(set(MENU) - set(CARD))

TYPES = {"image", "text", "number"}
TYPEMAP = {"unit": "text", "string": "text", "str": "text", "label": "text",
           "record": "text", "table": "text", "list": "text", "query": "text",
           "bbox": "text", "boolean": "text", "date": "number",
           "quantity": "number", "integer": "number", "int": "number",
           "float": "number", "amount": "number", "currency": "number",
           "duration": "number", "time": "number", "count": "number",
           "price": "number", "money": "number", "value": "number",
           "picture": "image", "photo": "image", "img": "image"}


def norm_type(s):
    s = re.sub(r"[^a-z]", "", str(s).lower())
    if s in TYPES:
        return s
    for k, v in TYPEMAP.items():
        if k in s:
            return v
    return "text"


# ------------------------------------------------------------ step 1 prompts

DECOMP = """Break a task down into the minimal sequence of atomic operations \
needed to answer it.

Rules:
- Describe each operation by its GENERAL CLASS. Never name the specific \
content of this question: no unit name, no currency name, no product, store \
or document name, no number. Write "convert a quantity between two systems of \
length measurement", not "convert kilometres to miles".
- `in` and `out` are type lists drawn from exactly: image, text, number. A \
step that reads figures off a picture is in=["image"], out=["number"]. A step \
that computes with figures is in=["number"], out=["number"].
- Do not solve the task and do not state any answer.
- If answering requires turning a reading into a different scale, system, \
currency or representation, that transformation is its own operation. Name \
the CLASS of transformation ("convert a quantity between two systems of \
length measurement", "restate a temperature on a different scale", "solve a \
linear relation for its unknown", "compute elapsed time between two clock \
readings", "restate a monetary amount in another currency"). Do not describe \
it as generic arithmetic and do not skip it.

Here are three worked examples.

Question: The image shows a fuel gauge in litres. Our tank holds 60.00 \
gallons. How many gallons short of full is it?
{{"available_inputs": [{{"name": "img", "type": "image"}}, {{"name": "k", "type": "number"}}],
 "operations": [{{"id": "o1", "desc": "read a numeric quantity off a rendered document", "in": ["image"], "out": ["number"]}},
                {{"id": "o2", "desc": "convert a quantity between two systems of volume measurement", "in": ["number"], "out": ["number"]}},
                {{"id": "o3", "desc": "take the difference between two numeric quantities", "in": ["number"], "out": ["number"]}}],
 "target_type": "number"}}

Question: The image shows a shelf of items and a price tag. I have a 5.00 \
voucher. What do I pay?
{{"available_inputs": [{{"name": "img", "type": "image"}}, {{"name": "k", "type": "number"}}],
 "operations": [{{"id": "o1", "desc": "count the instances of an object in a picture", "in": ["image"], "out": ["number"]}},
                {{"id": "o2", "desc": "read a numeric quantity off a rendered document", "in": ["image"], "out": ["number"]}},
                {{"id": "o3", "desc": "combine numeric quantities by arithmetic", "in": ["number"], "out": ["number"]}},
                {{"id": "o4", "desc": "take the difference between two numeric quantities", "in": ["number"], "out": ["number"]}}],
 "target_type": "number"}}

Question: The image shows a parts list. Each part's list price is published \
externally. What is the total after a 3.00 rebate?
{{"available_inputs": [{{"name": "img", "type": "image"}}, {{"name": "k", "type": "number"}}],
 "operations": [{{"id": "o1", "desc": "read the item names off a rendered document", "in": ["image"], "out": ["text"]}},
                {{"id": "o2", "desc": "look up an external record to obtain a value not present in the document", "in": ["text"], "out": ["number"]}},
                {{"id": "o3", "desc": "combine numeric quantities by arithmetic", "in": ["number"], "out": ["number"]}},
                {{"id": "o4", "desc": "take the difference between two numeric quantities", "in": ["number"], "out": ["number"]}}],
 "target_type": "number"}}

Now do the same for this question. Reply with JSON only, same form.

Question: {q}"""

MATCH = """Tool: {name}
Capability: {cap}

Operation: {op}

Can this tool, on its own, perform that operation? Answer Yes or No."""

CONTENT_FREE = "perform an unspecified operation"

PICKSET = """You have access to these tools:

{menu}

Question: {q}

Select the smallest set of tools that is sufficient to answer this question. \
Reply with JSON only: {{"tools": ["...", "..."]}}"""


_OPOBJ = re.compile(
    r'\{\s*"id"\s*:\s*"[^"]*"\s*,\s*"desc"\s*:\s*"[^"]*"\s*,\s*'
    r'"in"\s*:\s*\[[^\]]*\]\s*,\s*"out"\s*:\s*\[[^\]]*\]\s*\}')


def parse_json(text, want="operations"):
    """Parse, and if the generation was truncated mid-array, salvage the
    operation objects that did complete. A repetitive model (Llama emits the
    same read step eight times) otherwise loses the whole decomposition to the
    token budget, which would be scored as a failure of the method rather than
    of the budget."""
    m = re.search(r"\{.*\}", text, re.S)
    for cand in ([m.group(0), m.group(0).rsplit("}", 1)[0] + "}"] if m else []):
        try:
            d = json.loads(cand)
            if want in d:
                return d
        except Exception:
            continue
    if want == "operations":
        ops = [json.loads(x) for x in _OPOBJ.findall(text)]
        if ops:
            av = re.search(r'"available_inputs"\s*:\s*(\[.*?\])\s*,', text, re.S)
            try:
                av = json.loads(av.group(1)) if av else None
            except Exception:
                av = None
            tt = re.search(r'"target_type"\s*:\s*"([^"]*)"', text)
            return {"available_inputs": av or [{"name": "img", "type": "image"}],
                    "operations": ops,
                    "target_type": tt.group(1) if tt else "number"}
    if want == "tools":
        got = re.findall(r'"([A-Za-z][A-Za-z0-9]*)"', text)
        if got:
            return {"tools": got}
    return None


# ------------------------------------------------------------ typed dataflow

# Subtype lattice: a number can stand in wherever text is wanted (it prints as
# text); the converse is false. Without this the type filter is not a filter but
# a bug -- decompositions routinely type a numeric readout as `text`, and a
# strict set-inclusion test then discards every number-emitting tool, i.e. the
# right answer, before the margin is ever consulted.
SUB = {"number": {"number", "text"}, "text": {"text"}, "image": {"image"}}
# The question is always in the prompt, so its text and its stated figures are
# available to any tool call without a tool having to produce them.
AMBIENT = {"text", "number"}
IMAGE_READERS = [t for t, c in CARD.items() if "image" in c[1]]


def compatible(op, tool):
    """Can `tool` cover `op` on types alone?"""
    tin, tout = set(CARD[tool][1]), set(CARD[tool][2])
    need_in = {norm_type(x) for x in op["in"]}
    need_out = {norm_type(x) for x in op["out"]}
    if need_in - (tin | AMBIENT):
        return False          # e.g. only an image reader can consume an image
    return all(any(n in SUB[o] for o in tout) for n in need_out)


def reachable(ops, avail0):
    """Dataflow check: run the operations in order, each one's inputs must be
    available when it fires. Returns the final type set, or False."""
    avail = set(avail0) | {"text"}
    for op in ops:
        if {norm_type(x) for x in op["in"]} - avail:
            return False
        avail |= {norm_type(x) for x in op["out"]}
    return avail


_SEARCH_MEMO = {}


def search(ops, mtil, lam, avail0, target, topk=5):
    """S* = argmax_S sum_i max_{t in S} m~_it - lam |S|, subject to typed
    coverage + reachability. The objective decomposes per operation, so the
    optimum is a union of per-operation assignments; enumerate those."""
    key = (tuple((o["desc"], tuple(o["in"]), tuple(o["out"])) for o in ops),
           tuple(sorted(set(avail0))), norm_type(target), lam)
    if key in _SEARCH_MEMO:
        return _SEARCH_MEMO[key]
    av = reachable(ops, avail0)
    if not av or norm_type(target) not in av:
        return _SEARCH_MEMO.setdefault(key, None)
    cands = []
    for op in ops:
        c = [t for t in MENU if compatible(op, t)]
        c.sort(key=lambda t: -mtil.get((op["desc"], t), 0.0))
        cands.append(c[:topk])
    if any(not c for c in cands):
        return _SEARCH_MEMO.setdefault(key, None)
    # The image bridge is a property of the *set*, not of any one operation: if
    # the evidence starts as an image, S must contain something that can read
    # one. Stating it here rather than per-operation makes it robust to a
    # decomposition that mistypes the visual step as a text step.
    need_bridge = "image" in {norm_type(x) for x in avail0}
    best, bestv, relaxed, relaxedv = None, -1e9, None, -1e9
    for combo in itertools.product(*cands):
        S = set(combo)
        v = sum(mtil.get((op["desc"], t), 0.0)
                for op, t in zip(ops, combo)) - lam * len(S)
        if v > relaxedv:
            relaxed, relaxedv = S, v
        if need_bridge and not (S & set(IMAGE_READERS)):
            continue
        if v > bestv:
            best, bestv = S, v
    return _SEARCH_MEMO.setdefault(key, best if best else relaxed)


# ------------------------------------------------------------------ baselines

KEYWORD = [("mile", "gauge_over"), ("fahrenheit", "temp_over"),
           ("minute", "timetable_over"), ("euro", "invoice_currency"),
           ("coefficient", "equation_short"), ("voucher", "f2_visual_reason"),
           ("coupon", "f3_retrieve_reason"), ("change", "f1_extract_compute")]


# The environment's own operation list per family, written in the same
# delexicalized vocabulary the decomposer uses. This is an ORACLE for step 1
# only -- steps 2 and 3 are untouched -- so the gap between `oproute` and
# `oproute_oracledecomp` is exactly the loss attributable to decomposition.
RD = "read a numeric quantity off a rendered document"
RN = "read the item names off a rendered document"
AR = "combine numeric quantities by arithmetic"
DF = "take the difference between two numeric quantities"


def _op(i, desc, tin, tout):
    return {"id": f"o{i}", "desc": desc, "in": tin, "out": tout}


def _ops(*specs):
    return [_op(i + 1, d, a, b) for i, (d, a, b) in enumerate(specs)]


IMG_N = (["image"], ["number"])
NUM_N = (["number"], ["number"])
ORACLE_OPS = {
    "f1_extract_compute": _ops((RD, *IMG_N), (AR, *NUM_N), (DF, *NUM_N)),
    "f2_visual_reason": _ops(("count the instances of an object in a picture",
                              *IMG_N), (RD, *IMG_N), (AR, *NUM_N), (DF, *NUM_N)),
    "f3_retrieve_reason": _ops((RN, ["image"], ["text"]),
                               ("look up an external record to obtain a value "
                                "not present in the document", ["text"],
                                ["number"]), (AR, *NUM_N), (DF, *NUM_N)),
    "gauge_over": _ops((RD, *IMG_N),
                       ("convert a quantity between two systems of length "
                        "measurement", *NUM_N), (DF, *NUM_N)),
    "temp_over": _ops((RD, *IMG_N),
                      ("restate a temperature on a different scale", *NUM_N),
                      (DF, *NUM_N)),
    "equation_short": _ops((RD, *IMG_N), (RD, *IMG_N), (RD, *IMG_N),
                           ("solve a linear relation for its unknown", *NUM_N),
                           (DF, *NUM_N)),
    "timetable_over": _ops((RD, *IMG_N),
                           ("compute elapsed time between two clock readings",
                            *NUM_N), (DF, *NUM_N)),
    "invoice_currency": _ops((RD, *IMG_N),
                             ("convert a quantity between two systems of "
                              "currency", *NUM_N), (DF, *NUM_N)),
}
ORACLE_DESCS = sorted({o["desc"] for v in ORACLE_OPS.values() for o in v})


def keyword_route(q):
    ql = q.lower()
    for w, fam in KEYWORD:
        if w in ql:
            return set(CHAINS[fam])
    return set(UNION_TOOLS)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/"
                                     "results/taco/tgb2")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--lams", default="0,0.25,0.5,1.0,2.0,4.0")
    ap.add_argument("--anon", action="store_true",
                    help="robustness: replace tool names with opaque ids")
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b") + ("_anon" if a.anon else "")
    lams = [float(x) for x in a.lams.split(",")]

    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    by_fam = collections.defaultdict(list)
    for t in tasks:
        by_fam[t["family"]].append(t)
    per = a.n // len(by_fam)
    tasks = [t for f in sorted(by_fam) for t in by_fam[f][:per]]
    print(f"[{tag}] {len(tasks)} tasks, {len(by_fam)} families")

    tmp = tempfile.mkdtemp(prefix="tgb_or_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    tok = llm.get_tokenizer()
    sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
    sp_json = SamplingParams(max_tokens=800, temperature=0)
    sp_gen = SamplingParams(max_tokens=64, temperature=0)

    def chat(msgs):
        return tok.apply_chat_template([{"role": "user", "content": m}
                                        for m in msgs], tokenize=False,
                                       add_generation_prompt=True)

    def gen_chat(prompts, sp):
        ps = [chat([p]) for p in prompts]
        return [o.outputs[0].text for o in
                llm.generate([{"prompt": p} for p in ps], sp)]

    def logp_cont(prompts, conts):
        """sum log p(cont | prompt) with cont a short literal string."""
        jobs = []
        for p, c in zip(prompts, conts):
            ci = tok(c, add_special_tokens=False)["input_ids"]
            pi = tok(p, add_special_tokens=False)["input_ids"][:4096 - len(ci) - 1]
            jobs.append((pi + ci, len(pi), len(ci)))
        outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
        res = []
        for (full, plen, nc), o in zip(jobs, outs):
            s = 0.0
            for k in range(plen, plen + nc):
                d = o.prompt_logprobs[k]
                lp = d.get(full[k]) if d else None
                if lp is not None:
                    s += lp.logprob if hasattr(lp, "logprob") else float(lp)
            res.append(s)
        return res

    # ---------------- step 1: operation decomposition
    p1 = os.path.join(a.dir, f"oproute_ops_{tag}.json")
    if os.path.exists(p1):
        OPS = json.load(open(p1))
    else:
        raw = gen_chat([DECOMP.format(q=t["question"]) for t in tasks], sp_json)
        OPS, nfail = {}, 0
        for t, r in zip(tasks, raw):
            d = parse_json(r, "operations")
            if not d or not d.get("operations"):
                nfail += 1
                d = {"available_inputs": [{"name": "img", "type": "image"}],
                     "operations": [{"id": "o1",
                                     "desc": "answer a question about an image",
                                     "in": ["image"], "out": ["number"]}],
                     "target_type": "number"}
            ops = []
            for i, o in enumerate(d["operations"][:5]):
                ops.append({"id": o.get("id", f"o{i+1}"),
                            "desc": str(o.get("desc", ""))[:200],
                            "in": [norm_type(x) for x in (o.get("in") or ["text"])],
                            "out": [norm_type(x) for x in (o.get("out") or ["number"])]})
            OPS[t["id"]] = {
                "ops": ops,
                "avail": [norm_type(x.get("type", "text"))
                          for x in (d.get("available_inputs") or [])] or ["image"],
                "target": norm_type(d.get("target_type", "number")),
                "raw": r[:600]}
        json.dump(OPS, open(p1, "w"), indent=1)
        print(f"  decomposition: {nfail} parse failures / {len(tasks)}")

    nops = [len(OPS[t["id"]]["ops"]) for t in tasks]
    print(f"  operations per task: mean {st.mean(nops):.2f}, "
          f"dist {collections.Counter(nops).most_common()}")
    # leakage audit: did delexicalization hold?
    SURFACE = ["mile", "km", "kilomet", "fahrenheit", "celsius", "euro", "usd",
               "dollar", "minute", "receipt", "voucher", "coupon"]
    leak = st.mean(any(w in o["desc"].lower() for o in OPS[t["id"]]["ops"]
                       for w in SURFACE) for t in tasks)
    print(f"  surface-word leakage into operation text: {leak:.1%}")

    # ---------------- step 2: calibrated operation x tool margins
    p2 = os.path.join(a.dir, f"oproute_margins_{tag}.json")
    names = {t: (f"T{i:02d}" if a.anon else t) for i, t in enumerate(MENU)}
    if os.path.exists(p2):
        raw = json.load(open(p2))
        MTIL = {tuple(k.split(SEP)): v for k, v in raw.items()}
    else:
        uniq = sorted({o["desc"] for t in tasks for o in OPS[t["id"]]["ops"]}
                      | set(ORACLE_DESCS))
        print(f"  unique operations: {len(uniq)}  -> "
              f"{len(uniq) * len(MENU) * 2} scored forwards")
        pairs = [(o, t) for o in uniq + [CONTENT_FREE] for t in MENU]
        pr = [MATCH.format(name=names[t], cap=CARD[t][0], op=o)
              for o, t in pairs]
        pr = [chat([p]) for p in pr]
        yes = logp_cont(pr, ["Yes"] * len(pr))
        no = logp_cont(pr, ["No"] * len(pr))
        M = {(o, t): y - n for (o, t), y, n in zip(pairs, yes, no)}
        BIAS = {t: M[(CONTENT_FREE, t)] for t in MENU}
        MTIL = {(o, t): M[(o, t)] - BIAS[t] for (o, t) in pairs}
        json.dump({SEP.join(k): v for k, v in MTIL.items()},
                  open(p2, "w"), indent=1)
        print("  content-free bias b_t: " + ", ".join(
            f"{t}={BIAS[t]:+.2f}" for t in MENU[:9]))

    # ---------------- step 3: typed dataflow search, per lambda
    SEL = {}
    for lam in lams:
        for t in tasks:
            o = OPS[t["id"]]
            S = search(o["ops"], MTIL, lam, o["avail"], o["target"])
            SEL.setdefault(lam, {})[t["id"]] = sorted(S) if S else \
                sorted(UNION_TOOLS)

    best_lam_hint = lams[len(lams) // 2]

    # ---------------- baselines that need a generation
    p4 = os.path.join(a.dir, f"oproute_pick_{tag}.json")
    if os.path.exists(p4):
        PICK = json.load(open(p4))
    else:
        menu_txt = "\n".join(f"- {names[t]}: {CARD[t][0]}" for t in MENU)
        raw = gen_chat([PICKSET.format(menu=menu_txt, q=t["question"])
                        for t in tasks], sp_json)
        inv = {v: k for k, v in names.items()}
        PICK, nf = {}, 0
        for t, r in zip(tasks, raw):
            d = parse_json(r, "tools")
            got = [inv[x] for x in (d or {}).get("tools", []) if x in inv]
            if not got:
                nf += 1
                got = list(UNION_TOOLS)
            PICK[t["id"]] = sorted(set(got))
        json.dump(PICK, open(p4, "w"), indent=1)
        print(f"  one-shot tool pick: {nf} parse failures")

    # ---------------- evaluation: execute each S and score the answer
    METHODS = {f"oproute_lam{lam:g}": SEL[lam] for lam in lams}
    ORA = {}
    for t in tasks:
        S = search(ORACLE_OPS[t["family"]], MTIL, best_lam_hint, ["image", "number"],
                   "number")
        ORA[t["id"]] = sorted(S) if S else sorted(UNION_TOOLS)
    METHODS["oproute_oracledecomp"] = ORA
    METHODS["llm_pickset"] = PICK
    METHODS["keyword_router"] = {t["id"]: sorted(keyword_route(t["question"]))
                                 for t in tasks}
    METHODS["union"] = {t["id"]: sorted(UNION_TOOLS) for t in tasks}
    METHODS["oracle_chain"] = {t["id"]: sorted(CHAINS[t["family"]])
                               for t in tasks}

    need, T = {}, {t["id"]: t for t in tasks}
    for m, sel in METHODS.items():
        for tid, S in sel.items():
            need[(tid, tuple(sorted(S)))] = None
    keys = list(need)
    print(f"  distinct (task, set) to execute: {len(keys)}")
    prompts, ctxlen = [], {}
    for tid, S in keys:
        t = T[tid]
        rng = random.Random(zlib.crc32(f"{tid}/{'+'.join(S)}".encode()))
        outs, _ = run_chain(t, t["scene"], t["boxes"], set(S), rng,
                            corrupt_tools=())
        ctx = context(outs, [s["tool"] for s in t["plan"]])
        ctxlen[(tid, S)] = len(tok(ctx, add_special_tokens=False)["input_ids"])
        prompts.append(TPL.format(q=t["question"], o=ctx))
    gens = llm.generate([{"prompt": p} for p in prompts], sp_gen)
    ACC = {k: correct(g.outputs[0].text, T[k[0]]["gold"])
           for k, g in zip(keys, gens)}

    # ---------------- report
    ids = [t["id"] for t in tasks]
    train = set(ids[i] for i in range(len(ids)) if i % 5 < 2)   # 40% train
    test = [i for i in ids if i not in train]

    def stats(sel, subset):
        acc = st.mean(ACC[(i, tuple(sel[i]))] for i in subset)
        ntool = st.mean(len(sel[i]) for i in subset)
        ctx = st.mean(ctxlen[(i, tuple(sel[i]))] for i in subset)
        exact = st.mean(set(sel[i]) == set(CHAINS[T[i]["family"]])
                        for i in subset)
        f1 = []
        for i in subset:
            g, s = set(CHAINS[T[i]["family"]]), set(sel[i])
            f1.append(2 * len(g & s) / (len(g) + len(s)))
        return acc, st.mean(f1), exact, ntool, ctx

    print(f"\n[{tag}] TEST split n={len(test)}  "
          f"(lambda picked on the disjoint 40% train split)")
    print(f"  {'method':<22} {'acc':>6} {'setF1':>6} {'exact':>6} "
          f"{'|S|':>5} {'ctxtok':>7}")
    best_lam, best_tr = None, -1
    for lam in lams:
        tr = st.mean(ACC[(i, tuple(SEL[lam][i]))] for i in train)
        if tr > best_tr:
            best_lam, best_tr = lam, tr
    for m in list(METHODS):
        acc, f1, ex, nt, ct = stats(METHODS[m], test)
        star = "  <- lambda* (train)" if m == f"oproute_lam{best_lam:g}" else ""
        print(f"  {m:<22} {acc:6.3f} {f1:6.3f} {ex:6.2f} {nt:5.1f} "
              f"{ct:7.1f}{star}")

    # per-family breakdown for the selected lambda
    sel = SEL[best_lam]
    print(f"\n  per-family (lambda={best_lam:g}, test split)")
    for fam in sorted(by_fam):
        sub = [i for i in test if T[i]["family"] == fam]
        if not sub:
            continue
        acc, f1, ex, nt, _ = stats(sel, sub)
        u = st.mean(ACC[(i, tuple(sorted(UNION_TOOLS)))] for i in sub)
        o = st.mean(ACC[(i, tuple(sorted(CHAINS[fam])))] for i in sub)
        print(f"    {fam:<22} oproute {acc:.3f}  union {u:.3f}  "
              f"oracle {o:.3f}  exact {ex:.2f}  |S| {nt:.1f}")

    # where the picks go
    cnt = collections.Counter()
    for i in test:
        g, s = set(CHAINS[T[i]["family"]]), set(sel[i])
        cnt["exact" if g == s else "superset" if g < s
            else "missing_gt_tool" if g - s else "other"] += 1
    print(f"\n  pick anatomy (lambda={best_lam:g}): "
          + ", ".join(f"{k} {v} ({v/len(test):.0%})"
                      for k, v in cnt.most_common()))
    for k in ["exact", "superset", "missing_gt_tool", "other"]:
        sub = [i for i in test
               if (lambda g, s: "exact" if g == s else "superset" if g < s
                   else "missing_gt_tool" if g - s else "other")(
                       set(CHAINS[T[i]["family"]]), set(sel[i])) == k]
        if sub:
            print(f"    acc | {k:<16} {st.mean(ACC[(i, tuple(sel[i]))] for i in sub):.3f}"
                  f"   (n={len(sub)})")

    json.dump({"sel": {str(k): v for k, v in SEL.items()},
               "pick": PICK, "acc": {f"{k[0]}|{'+'.join(k[1])}": v
                                     for k, v in ACC.items()},
               "best_lam": best_lam},
              open(os.path.join(a.dir, f"oproute_{tag}.json"), "w"), indent=1)
    print(f"\n  wrote oproute_{tag}.json")


if __name__ == "__main__":
    main()
