"""TGB tool executors.

Every tool is a pure function of the *scene* (and, for perception tools, of the
recorded render boxes) -- no executor ever sees the gold answer or the task's
final arithmetic. Output strings are byte-format-compatible with the real
AgentLego tools used by GTA-Atomic, sampled from the GTA dialogs:

    OCR                   "(x1, y1, x2, y2) TEXT" per line
    CountGivenObject      bare integer
    ImageDescription      free prose
    TextToBbox            "(x1, y1, x2, y2), score NN" per line
    GoogleSearch          "1 - Title: snippet\\n\\n2 - ..."
    Calculator            bare number, or a Python error string
    Solver                "[value]"

so a context built here is surface-indistinguishable from a real GTA context.

`run_chain` is the executable part of the benchmark: it takes a *set* of tools
and runs them in dependency order, feeding each downstream tool arguments
**parsed out of upstream tool outputs**. That is what makes the coalition
conditions mechanically different rather than hand-written:

  * drop the upstream tool  -> the downstream tool receives an unresolved
    placeholder and returns an error, exactly the "no marginal without its
    upstream input" structure that condition (3) is about;
  * corrupt the upstream tool -> the downstream tool silently computes the
    wrong number, with no other change to the context.
"""
import re

# --------------------------------------------------------------------- noise

_DIGIT = {"0": "8", "1": "7", "3": "8", "5": "6", "6": "5", "7": "1",
          "8": "3", "9": "0", "2": "7", "4": "1"}
# "S" and "O" are deliberately absent: they occur in the field markers ("USD",
# "QTY") that a real engine reads reliably, and garbling them would corrupt the
# *schema* rather than the value -- which again would collapse `corrupt` into
# `no_upstream`.
_ALPHA = {"A": "H", "N": "M", "M": "N", "l": "1",
          "e": "c", "o": "c", "r": "n", "n": "m"}


def garble(text, rng, p=0.35):
    """Character-level OCR corruption in the style of the real GTA OCR
    ("MAGNA" -> "HAGNN").

    Digits are confused with *digits*, never with letters. That is deliberate:
    a corruption that broke number parsing outright would collapse the
    "corrupted coalition" condition into the "missing upstream" condition
    (both would end in a downstream error). Keeping the output parseable but
    wrong is what isolates *output quality* as its own failure mode.
    """
    out, hit = [], False
    for ch in text:
        m = _DIGIT if ch.isdigit() else _ALPHA
        if ch in m and rng.random() < p:
            out.append(m[ch])
            hit = True
        else:
            out.append(ch)
    if not hit:  # guarantee the corruption is real, not a no-op
        for i, ch in enumerate(out):
            if ch.isdigit():
                out[i] = _DIGIT[ch]
                hit = True
                break
    return "".join(out)


# ------------------------------------------------------------------ perception

def ocr(scene, boxes, rng, corrupt=False):
    if not boxes:
        return "No text detected."
    lines = []
    for b in boxes:
        t = garble(b["text"], rng) if corrupt else b["text"]
        x1, y1, x2, y2 = b["box"]
        lines.append(f"({x1}, {y1}, {x2}, {y2}) {t}")
    return "\n".join(lines)


def count_given_object(scene, boxes, rng, corrupt=False, text=None):
    if scene["kind"] != "shelf":
        return "0"
    n = scene["n_target"]
    if text and scene["target"] not in text.lower():
        for d in scene["distractors"]:
            if d["name"] in text.lower():
                n = d["count"]
                break
    if corrupt:
        n = max(1, n + rng.choice([-2, -1, 1, 2]))
    return str(n)


_VAGUE = ["several", "a number of", "multiple", "a group of", "various"]


def image_description(scene, boxes, rng, corrupt=False):
    """Prose. Deliberately faithful but *imprecise*: it names the objects and
    says a price tag is present, and never states the exact count or the
    digits on the tag. This is the "right modality, insufficient content"
    control -- the reason a coalition can be wrong without being noisy."""
    if scene["kind"] == "shelf":
        v = rng.choice(_VAGUE)
        others = ", ".join(d["name"] + "s" for d in scene["distractors"])
        return (f"The image shows a store shelf ({scene['shop']}) holding "
                f"{v} {scene['target']}s arranged in rows. Alongside them "
                f"there are also {others}. A white price tag is attached to "
                f"the front of the shelf, but the writing on it is small. "
                f"The overall scene appears to be a retail display.")
    if scene["kind"] == "receipt":
        v = rng.choice(_VAGUE)
        return (f"The image shows a printed paper receipt from a shop. It "
                f"lists {v} purchased items, each on its own line with a "
                f"quantity and a price, followed by a horizontal rule and a "
                f"line giving a tax rate. The text is small but legible.")
    if scene["kind"] == "form":
        v = rng.choice(_VAGUE)
        return (f"The image shows a printed order form. Under a heading it "
                f"lists {v} product codes, each followed by a handwritten-"
                f"looking quantity marker. No prices appear anywhere on the "
                f"form. A signature line runs along the bottom.")
    return "The input does not contain an image."


def text_to_bbox(scene, boxes, rng, corrupt=False, text=None):
    if not boxes:
        return "No matching region found."
    out = []
    for b in boxes[:4]:
        x1, y1, x2, y2 = b["box"]
        out.append(f"({x1}, {y1}, {x2}, {y2}), score {rng.randint(60, 90)}")
    return "\n".join(out)


# ------------------------------------------------------------------- retrieval

def google_search(scene, boxes, rng, corrupt=False, queries=()):
    """Deterministic retrieval over the task's own almanac corpus, rendered in
    the real GoogleSearch answer-box format. Under corruption the retriever
    returns the *near-miss* entity (a plausible neighbour in the corpus), which
    is the retrieval analogue of a misread digit."""
    ents = scene.get("entities")
    if not ents:
        return "No results found."
    blocks, i = [], 1
    for q in queries:
        hit = None
        for e in ents:
            if e["name"].lower() in q.lower():
                hit = e
                break
        if hit is None:
            hit = ents[0]
        if corrupt:
            # a near-miss in the same product category if one exists, else any
            # other entity -- the retrieval analogue of a misread digit
            alts = [e for e in ents if e is not hit
                    and e["category"] == hit["category"]] or \
                   [e for e in ents if e is not hit]
            if alts:
                hit = alts[rng.randrange(len(alts))]
        blocks.append(f"{i} - {hit['name']} | {hit['category']}: {hit['blurb']} "
                      f"The listed {hit['attr']} is {hit['value']}.")
        i += 1
        # one distractor result per query, as the real tool returns
        d = ents[(ents.index(hit) + 3) % len(ents)]
        blocks.append(f"{i} - {d['name']} product page: general information "
                      f"about {d['category']} models and availability.")
        i += 1
    return "\n\n".join(blocks)


# ------------------------------------------------------------------ computation

_SAFE = re.compile(r"^[0-9+\-*/(). ]+$")


def calculator(scene, boxes, rng, corrupt=False, expression=None):
    """Real evaluation of a real expression string. If the expression still
    contains an unresolved placeholder (because its upstream tool was not in
    the coalition) it fails the way the real tool fails."""
    if not expression:
        return "Error: no expression provided."
    if not _SAFE.match(expression):
        bad = sorted(set(re.findall(r"[A-Za-z_]\w*", expression)))
        return (f"NameError: name '{bad[0]}' is not defined"
                if bad else "SyntaxError: invalid expression")
    try:
        v = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - guarded
    except ZeroDivisionError:
        return "ZeroDivisionError: division by zero"
    except Exception as e:                                # pragma: no cover
        return f"Error: {type(e).__name__}"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def solver(scene, boxes, rng, corrupt=False, equation=None):
    if not equation:
        return "Error: no equation provided."
    m = re.match(r"^x\s*-\s*\(([^)]*)\)\s*=\s*0$", equation.strip())
    if not m:
        return "Error: could not parse the equation."
    v = calculator(scene, boxes, rng, expression=m.group(1))
    return f"[{v}]" if not v.startswith(("Error", "NameError", "Syntax")) else v


TOOLS = {
    "OCR": ocr,
    "CountGivenObject": count_given_object,
    "ImageDescription": image_description,
    "TextToBbox": text_to_bbox,
    "GoogleSearch": google_search,
    "Calculator": calculator,
    "Solver": solver,
}

# Tools whose output is an image in real GTA -- they can appear in a menu as
# type-mismatched options but produce no readable text.
IMAGE_TOOLS = ["TextToImage", "DrawBox", "AddText", "ImageStylization", "Plot"]


def _image_tool_output(name):
    return "image/dummy_generated_image.jpg"


# ---------------------------------------------------------------- chain runner

def run_chain(task, scene, boxes, coalition, rng, corrupt_tools=()):
    """Execute `coalition` (a set of tool names) on `scene`, in dependency
    order, wiring downstream arguments from upstream *outputs*.

    Returns (outputs, trace) where outputs maps tool name -> output string in
    call order, and trace records the resolved arguments -- the audit trail
    that the chain really ran rather than being pasted in.
    """
    plan = task["plan"]           # ordered list of {tool, args, needs}
    outputs, trace = {}, []
    have = {}                     # symbol -> value parsed from an upstream output
    for step in plan:
        tool = step["tool"]
        if tool not in coalition:
            continue
        args = dict(step.get("args", {}))
        corrupt = tool in corrupt_tools
        if "template" in step:
            # Downstream arguments are *built from upstream output*. A symbol
            # that no upstream resolved is left as its own name, so the tool
            # receives an unresolved placeholder and fails -- which is what
            # gives a downstream tool no marginal without its upstream.
            expr = step["template"]
            for sym in step.get("needs", []):
                expr = expr.replace("{" + sym + "}", str(have[sym])
                                    if sym in have else sym)
            args[step.get("arg_key", "expression")] = expr
        fn = TOOLS[tool]
        out = fn(scene, boxes, rng, corrupt=corrupt, **args)
        outputs[tool] = out
        trace.append(dict(tool=tool, args=args, output=out, corrupt=int(corrupt)))
        for sym, how in step.get("emits", {}).items():
            v = _parse(out, how)
            if v is not None:
                have[sym] = v
    # Tools in the coalition that the task's chain has no use for still run --
    # that is the whole point of the `wrong` and `full` conditions. They get
    # the generic arguments an agent would guess, and are appended after the
    # chain, matching the call order a ReAct agent produces.
    for tool in coalition:
        if tool in outputs:
            continue
        if tool in IMAGE_TOOLS:
            outputs[tool] = _image_tool_output(tool)
            trace.append(dict(tool=tool, args={}, output=outputs[tool], corrupt=0))
            continue
        if tool not in TOOLS:
            continue
        args = dict(DEFAULT_ARGS.get(tool, {}))
        if tool == "GoogleSearch":
            args["queries"] = [task["question"][:80]]
        out = TOOLS[tool](scene, boxes, rng, corrupt=(tool in corrupt_tools),
                          **args)
        outputs[tool] = out
        trace.append(dict(tool=tool, args=args, output=out, corrupt=0))
    return outputs, trace


DEFAULT_ARGS = {
    "CountGivenObject": {"text": "the objects in the image"},
    "TextToBbox": {"text": "the main object in the image"},
}


# Field syntax follows what a real OCR engine returns on the TGB renders (see
# scenes.FSIZE and tgb/probe_render.py): "QTY 3", "USD 30.47" survive token by
# token, where "3 @ $30.47" did not.
_QTY_ONLY = re.compile(r"QTY\s*(\d+)")
_USD_ONLY = re.compile(r"USD\s*(\d+)\s*\.\s*(\d{2})")
_TAX_RE = re.compile(r"(\d+)\s*%")
_QTY_RE = _QTY_ONLY
# A real engine emits one detection per *word group*, so "QTY 3" and
# "USD 30.47" arrive as separate lines with their own bounding boxes -- and
# those boxes are full of digits. Pairing therefore has to happen after the
# boxes are stripped, in reading order, not with one regex spanning both.
_BBOX = re.compile(r"^\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)\s*", re.M)


def strip_boxes(out):
    return _BBOX.sub("", out)


def _parse(out, how):
    """Pull a symbol's value out of a tool output *string*.

    Everything downstream of a perception tool goes through here, so this is
    also where corruption becomes wrongness: a garbled "$3.40" -> "$8.40"
    still parses, and the bad number flows into the Calculator untouched.
    Parsing failures return None, which leaves the downstream placeholder
    unresolved and makes the Calculator error out.
    """
    kind = how["kind"]
    if kind == "int":
        m = re.search(r"-?\d+", out)
        return int(m.group()) if m else None
    if kind == "int_idx":
        vals = _QTY_RE.findall(out)
        i = how["idx"]
        return int(vals[i]) if i < len(vals) else None
    # money is kept as the literal string the tool printed, not a float: the
    # expression that reaches the Calculator must contain the same characters
    # the upstream tool emitted, so "$19.70" cannot silently become 19.7
    if kind == "money":
        m = re.search(how["pattern"], out)
        return re.sub(r"\s+", "", m.group(1)) if m else None
    if kind == "money_idx":
        vals = re.findall(how["pattern"], out)
        i = how["idx"]
        return re.sub(r"\s+", "", vals[i]) if i < len(vals) else None
    if kind == "float":
        m = re.search(r"-?\d+(?:\.\d+)?", out)
        return float(m.group()) if m else None
    if kind == "nums":
        # the i-th numbers of the output, bbox prefixes removed, joined for a
        # downstream tool's `value` argument
        flat = strip_boxes(out)
        vals = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)", flat)
        idx = how["idx"]
        if not vals or max(idx) >= len(vals):
            return None
        return ",".join(vals[i] for i in idx)
    if kind == "rows_expr":
        # every row of a multi-row table -> one product per row. Aggregating
        # six products is what makes the downstream Calculator load-bearing.
        rows = re.findall(r"qty\s+(\d+)\s*\|\s*unit\s+(\d+\.\d{2})", out)
        lo = how.get("min_qty")
        if lo is not None:
            rows = [(q, u) for q, u in rows if int(q) >= lo]
        return " + ".join(f"{q}*{u}" for q, u in rows) or None
    if kind == "mean_expr":
        vs = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?=[,\s])", out.split("\n")[-1])
        return f"({' + '.join(vs)}) / {len(vs)}" if vs else None
    if kind == "mean_f_expr":
        vs = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?=[,\s])", out.split("\n")[-1])
        return (f"(({' + '.join(vs)}) / {len(vs)}) * 1.8 + 32") if vs else None
    if kind == "gap_expr":
        d = re.search(r"depart\s*\|\s*(\d+)\s+(\d+)", out)
        a = re.search(r"arrive\s*\|\s*(\d+)\s+(\d+)", out)
        if not d or not a:
            return None
        return (f"({int(a.group(1))}*60 + {int(a.group(2))}) - "
                f"({int(d.group(1))}*60 + {int(d.group(2))})")
    if kind == "two_price_expr":
        ps = re.findall(r"price of (\d+\.\d{2})", out)
        if len(ps) < 2:
            return None
        return f"{how['qa']}*{ps[0]} + {how['qb']}*{ps[1]}"
    if kind == "cipher_words":
        m = re.search(r"recorded as ([a-z ]+?) in the supplier", out)
        return m.group(1).strip() if m else None
    if kind == "table_expr":
        m = re.search(r"quantity\s+(\d+)\s*\|\s*unit value\s+(\d+\.\d{2})",
                      out)
        return f"{m.group(1)} * {m.group(2)}" if m else None
    if kind == "receipt_expr":
        flat = strip_boxes(out)
        qtys = _QTY_ONLY.findall(flat)
        prices = _USD_ONLY.findall(flat)   # "USD 30 . 47" -> ("30", "47")
        if not qtys or len(qtys) != len(prices):
            return None
        sub = " + ".join(f"{q}*{a}.{b}" for q, (a, b) in zip(qtys, prices))
        taxes = _TAX_RE.findall(out)
        if not taxes:
            return None
        return f"({sub}) * {1 + int(taxes[-1]) / 100:.4f}"
    return None
