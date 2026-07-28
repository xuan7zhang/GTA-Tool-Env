"""TACO tool-output intervention layer (Phase 1).

Sits inside the probe-proxy, between the real AgentLego tool server and the
agent. Receives the *original* tool output and returns a transformed output
according to an intervention spec, preserving the underlying evidence unless
the condition is explicitly a corruption control.

Design
------
Every tool output is first parsed into a **canonical representation**

    {"tool": ..., "kind": ..., "units": [{"text": ..., "meta": {...}}, ...]}

and every rendered condition is produced from that representation by a
**deterministic renderer** (no LLM paraphraser anywhere). So F3 and F4 of the
same output carry exactly the same evidence units, by construction.

Axes
----
format   F0 original | F1 minimal | F2 concise NL | F3 key-value | F4 JSON
         F5 evidence-first structured | F6 verbose prose
length   L0 ~10-25 tok | L1 ~25-50 | L2 ~50-100 | L3 ~100-200 | L4 full
         L5 ~2x padded
         mechanism for expansion: relevant | irrelevant | redundant
position front | middle | back  (of the key evidence within a fixed length)

Honest note on the length axis, recorded here because it constrains what the
length coefficients may be read as:

* L0-L3 shorten by **dropping evidence units** (in native order, no gold
  signal used). They therefore confound "shorter" with "less information".
* The three L5 mechanisms at a **matched token budget** do not: relevant adds
  real units, irrelevant adds tool-generic boilerplate that asserts nothing
  about the task, redundant repeats units already present. The pure-length
  contrast is the mechanism contrast, not the truncation ladder.
* F1 vs F0 changes evidence *density* at a constant evidence set (bounding
  boxes / scores are wrapper, not evidence), so it is the clean H3 probe.

Nothing here imports the dataset, the gold answers, or any annotation: the
transform is a function of the tool output alone.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

VERSION = "taco-transform-v1"

FORMATS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
LENGTHS = ["L0", "L1", "L2", "L3", "L4", "L5"]
MECHANISMS = ["relevant", "irrelevant", "redundant"]
POSITIONS = ["front", "middle", "back"]

# Target token buckets (upper bound of the bucket; L4/L5 are relative).
LENGTH_BUDGET = {"L0": 25, "L1": 50, "L2": 100, "L3": 200}

# Tools whose output is a file path / base64 payload: never transformed.
IMAGE_OUTPUT_TOOLS = {
    "TextToImage", "ImageStylization", "DrawBox", "AddText", "Plot",
}

# Per-tool vocabulary for the renderers. `field` is the key-value/JSON key,
# `noun` the concise-NL noun phrase, `source` the F5 SOURCE line.
TOOL_VOCAB = {
    "OCR": dict(field="text", noun="visible text", source="image text",
                unit_noun="text line"),
    "ImageDescription": dict(field="description", noun="image description",
                             source="image caption", unit_noun="sentence"),
    "RegionAttributeDescription": dict(field="attribute", noun="region attribute",
                                       source="region description", unit_noun="sentence"),
    "TextToBbox": dict(field="bbox", noun="object location",
                       source="object detection", unit_noun="detection"),
    "CountGivenObject": dict(field="count", noun="object count",
                             source="object counting", unit_noun="count"),
    "Calculator": dict(field="result", noun="calculation result",
                       source="calculator", unit_noun="result"),
    "Solver": dict(field="solution", noun="equation solution",
                   source="equation solver", unit_noun="solution"),
    "GoogleSearch": dict(field="results", noun="search results",
                         source="web search", unit_noun="result"),
    "MathOCR": dict(field="latex", noun="math expression",
                    source="math OCR", unit_noun="expression"),
}
DEFAULT_VOCAB = dict(field="result", noun="tool result", source="tool output",
                     unit_noun="item")

# Task-irrelevant but semantically valid filler. Asserts nothing about the
# image, the query, or the answer — only about the tool module itself. Used
# for irrelevant expansion and for position padding.
FILLER_SENTENCES = [
    "This result was produced by the {tool} module of the tool server.",
    "The module accepted the supplied input and completed its processing routine.",
    "Processing finished without an internal error being raised.",
    "The module runs on the shared inference host and returns its output as text.",
    "Output is emitted once per call and is not cached between calls.",
    "The module exposes a fixed interface and does not take additional options.",
    "Standard input formats are accepted by this module.",
    "The routine completed within its normal time budget.",
    "No further post-processing was applied to the values above.",
    "The module reports its findings in the form shown here.",
]

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_OCR_LINE_RE = re.compile(r"^\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)\s*(.*)$")
_BBOX_SCORE_RE = re.compile(
    r"^\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)\s*,?\s*score\s*([\d.]+)\s*$", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def ntok(text: str) -> int:
    """Deterministic word/punct token approximation (proxy-side budgeting).

    Exact model-tokenizer counts are recomputed offline in the analysis stage
    from the logged strings; this only has to be monotone and reproducible.
    """
    return len(_TOKEN_RE.findall(text or ""))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# --------------------------------------------------------------------------
# canonical representation
# --------------------------------------------------------------------------

def to_canonical(tool: str, payload: Any) -> Dict[str, Any]:
    """Parse a tool payload into {tool, kind, units, native}.

    `units` are ordered as the tool emitted them (reading order for OCR,
    detection order for TextToBbox, sentence order for prose). No relevance
    signal is used, so truncation cannot leak gold information.
    """
    native = payload if isinstance(payload, str) else json.dumps(payload)
    text = native

    if not isinstance(payload, str):
        # int / float / bool scalar outputs (e.g. CountGivenObject)
        return dict(tool=tool, kind="scalar", native=native,
                    units=[dict(text=str(payload), meta={})])

    lines = [ln for ln in text.split("\n") if ln.strip()]

    # TextToBbox: "(x1, y1, x2, y2), score 88.5".  Checked BEFORE the OCR
    # pattern, which is the looser of the two and would otherwise swallow the
    # detection lines and mis-file ", score 88.5" as recognised text.
    if lines and all(_BBOX_SCORE_RE.match(ln) for ln in lines):
        units = []
        for ln in lines:
            m = _BBOX_SCORE_RE.match(ln)
            bbox = [int(m.group(i)) for i in range(1, 5)]
            units.append(dict(text="(%d, %d, %d, %d)" % tuple(bbox),
                              meta=dict(bbox=bbox, score=float(m.group(5)),
                                        score_raw=m.group(5))))
        return dict(tool=tool, kind="bbox_score", native=native, units=units)

    # OCR: "(x1, y1, x2, y2) recognised text"
    if lines and all(_OCR_LINE_RE.match(ln) for ln in lines):
        units = []
        for ln in lines:
            m = _OCR_LINE_RE.match(ln)
            bbox = [int(m.group(i)) for i in range(1, 5)]
            units.append(dict(text=m.group(5).strip(), meta=dict(bbox=bbox)))
        if any(u["text"] for u in units):
            return dict(tool=tool, kind="bbox_text", native=native, units=units)

    # short scalar-ish payloads (Calculator, Solver, counts as strings)
    if len(lines) <= 1 and ntok(text) <= 8:
        return dict(tool=tool, kind="scalar", native=native,
                    units=[dict(text=text.strip(), meta={})])

    # multi-line non-OCR (e.g. search results): one unit per line
    if len(lines) > 1 and all(ntok(ln) < 60 for ln in lines):
        return dict(tool=tool, kind="lines", native=native,
                    units=[dict(text=ln.strip(), meta={}) for ln in lines])

    # prose: one unit per sentence
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text.replace("\n", " ")) if s.strip()]
    return dict(tool=tool, kind="prose", native=native,
                units=[dict(text=s, meta={}) for s in sents] or
                      [dict(text=text.strip(), meta={})])


# --------------------------------------------------------------------------
# renderers  (canonical -> string).  Deterministic, no paraphraser.
# --------------------------------------------------------------------------

def _vocab(tool: str) -> Dict[str, str]:
    return TOOL_VOCAB.get(tool, DEFAULT_VOCAB)


def _unit_strings(canon: Dict[str, Any], units: List[dict], keep_wrapper: bool) -> List[str]:
    """Evidence strings for the units.

    keep_wrapper=True re-attaches tool metadata that is *wrapper*, not
    evidence (OCR bounding boxes, detector scores). F1 drops it; F0 keeps it.
    """
    out = []
    for u in units:
        t = u["text"]
        if keep_wrapper and canon["kind"] == "bbox_text" and u["meta"].get("bbox"):
            t = "(%d, %d, %d, %d) %s" % (*u["meta"]["bbox"], t)
        elif keep_wrapper and canon["kind"] == "bbox_score" and u["meta"].get("score") is not None:
            # score_raw keeps the tool's own literal ("76", not "76.0"): a
            # renderer must not silently restate a value in another notation.
            t = "%s, score %s" % (t, u["meta"].get("score_raw") or u["meta"]["score"])
        out.append(t)
    return out


def render(canon: Dict[str, Any], fmt: str, units: Optional[List[dict]] = None) -> str:
    """Render the given units of a canonical representation in format `fmt`."""
    units = canon["units"] if units is None else units
    v = _vocab(canon["tool"])
    kind = canon["kind"]

    if fmt == "F0":
        if units is canon["units"] or len(units) == len(canon["units"]):
            return canon["native"]
        return "\n".join(_unit_strings(canon, units, keep_wrapper=True))

    ev = _unit_strings(canon, units, keep_wrapper=False)
    ev_full = _unit_strings(canon, units, keep_wrapper=True)

    if fmt == "F1":                                        # minimal plain text
        return "\n".join(ev)

    if fmt == "F2":                                        # concise NL
        if kind in ("scalar",):
            return 'The %s is %s.' % (v["noun"], ev[0] if ev else "")
        if kind == "prose":
            return 'The %s is: %s' % (v["noun"], " ".join(ev))
        body = "; ".join('"%s"' % e if kind == "bbox_text" else e for e in ev)
        return 'The %s is: %s.' % (v["noun"], body)

    if fmt == "F3":                                        # key-value
        if len(ev) == 1:
            return "%s: %s" % (v["field"], ev[0])
        return "\n".join("%s_%d: %s" % (v["field"], i + 1, e) for i, e in enumerate(ev))

    if fmt == "F4":                                        # JSON
        if kind == "bbox_text":
            payload = {v["field"]: [dict(text=u["text"], bbox=u["meta"].get("bbox"))
                                    for u in units]}
        elif kind == "bbox_score":
            payload = {v["field"]: [dict(bbox=u["meta"].get("bbox"),
                                         score=u["meta"].get("score_raw",
                                                             u["meta"].get("score")))
                                    for u in units]}
        elif len(ev) == 1:
            payload = {v["field"]: ev[0]}
        else:
            payload = {v["field"]: ev}
        return json.dumps(payload, ensure_ascii=False)

    if fmt == "F5":                                        # evidence-first
        head = "EVIDENCE: " + ("\n          ".join(ev) if len(ev) > 1 else (ev[0] if ev else ""))
        return "%s\nSOURCE: %s" % (head, v["source"])

    if fmt == "F6":                                        # verbose prose
        lead = ("The %s module was run on the supplied input and returned the "
                "following %s. " % (canon["tool"], v["noun"]))
        body = (" ".join(ev) if kind == "prose"
                else "; ".join(ev_full) + ".")
        tail = (" The values above are reported exactly as the module produced them, "
                "with no additional interpretation applied. Each %s listed corresponds "
                "to one item that the module identified in the input." % v["unit_noun"])
        return lead + body + tail

    raise ValueError("unknown format %r" % fmt)


# --------------------------------------------------------------------------
# length + position
# --------------------------------------------------------------------------

def _filler(tool: str, n_tokens: int, start: int = 0) -> str:
    """Deterministic task-irrelevant filler of approximately n_tokens tokens."""
    out, used, i = [], 0, start
    while used < n_tokens and len(out) < 60:
        s = FILLER_SENTENCES[i % len(FILLER_SENTENCES)].format(tool=tool)
        out.append(s)
        used += ntok(s)
        i += 1
    return " ".join(out)


def _truncate_units(canon: Dict[str, Any], fmt: str, budget: int) -> List[dict]:
    """Keep units in native order while the rendered output fits `budget`."""
    units = canon["units"]
    if not units:
        return units
    keep: List[dict] = []
    for u in units:
        trial = keep + [u]
        if keep and ntok(render(canon, fmt, trial)) > budget:
            break
        keep = trial
    return keep or units[:1]


def apply_length(canon: Dict[str, Any], fmt: str, level: str,
                 mechanism: str = "relevant") -> Tuple[str, List[dict], Dict[str, Any]]:
    """Return (text, kept_units, meta) for the requested length level."""
    meta: Dict[str, Any] = dict(length_level=level, length_mechanism=mechanism)

    if level in LENGTH_BUDGET:
        kept = _truncate_units(canon, fmt, LENGTH_BUDGET[level])
        text = render(canon, fmt, kept)
        meta.update(units_kept=len(kept), units_total=len(canon["units"]),
                    truncated=len(kept) < len(canon["units"]), padded=False)
        return text, kept, meta

    if level == "L4":
        kept = canon["units"]
        text = render(canon, fmt, kept)
        meta.update(units_kept=len(kept), units_total=len(kept),
                    truncated=False, padded=False)
        return text, kept, meta

    if level == "L5":
        # ~2x the full-output length, capped, via the requested mechanism.
        base_units = canon["units"]
        base_text = render(canon, fmt, base_units)
        base_tok = ntok(base_text)
        target = min(2 * base_tok, base_tok + 400)
        need = max(0, target - base_tok)

        if mechanism == "irrelevant":
            text = base_text + "\n" + _filler(canon["tool"], need)
            kept = base_units
        elif mechanism == "redundant":
            # repeat the evidence units (paraphrase-free repetition) until the
            # budget is met; no new factual content is introduced.
            reps, cur = [], base_tok
            i = 0
            while cur < target and i < 40:
                u = base_units[i % len(base_units)] if base_units else None
                if u is None:
                    break
                reps.append(u)
                cur = ntok(render(canon, fmt, base_units + reps))
                i += 1
            kept = base_units + reps
            text = render(canon, fmt, kept)
        else:  # "relevant" -- all real units already present; nothing to add
            kept = base_units
            text = base_text
            meta["relevant_expansion_saturated"] = True

        meta.update(units_kept=len(base_units), units_total=len(base_units),
                    truncated=False, padded=(mechanism != "relevant"),
                    target_tokens=target)
        return text, kept, meta

    raise ValueError("unknown length level %r" % level)


def apply_position(text: str, tool: str, position: str) -> str:
    """Place the evidence block at front / middle / back of a padded output.

    Total length and semantic content are held fixed across the three levels:
    the same amount of filler is used, only its placement changes.
    """
    pad_tokens = max(20, ntok(text) // 2)
    a = _filler(tool, pad_tokens // 2, start=0)
    b = _filler(tool, pad_tokens - pad_tokens // 2, start=5)
    if position == "front":
        return "%s\n%s %s" % (text, a, b)
    if position == "back":
        return "%s %s\n%s" % (a, b, text)
    if position == "middle":
        return "%s\n%s\n%s" % (a, text, b)
    raise ValueError("unknown position %r" % position)


# --------------------------------------------------------------------------
# factual preservation
# --------------------------------------------------------------------------

def check_preservation(canon: Dict[str, Any], kept_units: List[dict],
                       out_text: str) -> Dict[str, Any]:
    """Did the transform keep every evidence unit it claimed to keep, and did
    it invent anything?

    * `units_present`  — each kept unit's evidence text occurs in the output.
    * `no_new_numbers` — the output introduces no numeric literal absent from
      the native output (guards against fabricated facts; filler is prose).
      Scaffold indices produced by the key-value renderer (`text_1:`,
      `text_2:` ...) are allowed and recorded as such: they are structural,
      carry no claim about the input, and are bounded by the unit count.
    """
    no = _norm(out_text)
    missing = [u["text"] for u in kept_units
               if _norm(u["text"]) and _norm(u["text"]) not in no]
    src_nums = set(re.findall(r"\d+(?:\.\d+)?", canon["native"]))
    scaffold = {str(i) for i in range(1, len(kept_units) + 2)}
    out_nums = set(re.findall(r"\d+(?:\.\d+)?", out_text))
    new_nums = sorted(out_nums - src_nums - scaffold)
    return dict(units_present=(len(missing) == 0),
                missing_units=missing[:5],
                n_missing=len(missing),
                no_new_numbers=(len(new_nums) == 0),
                new_numbers=new_nums[:5])


# --------------------------------------------------------------------------
# top-level entry point used by the proxy
# --------------------------------------------------------------------------

def transform(tool: str, payload: Any, spec: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Apply an intervention spec to one tool payload.

    spec: {"format": "F0".."F6",
           "length": {"level": "L0".."L5", "mechanism": "relevant"|...} | null,
           "position": "front"|"middle"|"back" | null}

    Returns (new_payload, log_record). `new_payload` is JSON-serialisable and
    replaces the upstream body. On any parse/render failure the original
    payload is returned unchanged and the failure recorded, so a transform bug
    degrades to passthrough rather than corrupting a condition silently.
    """
    rec: Dict[str, Any] = dict(version=VERSION, tool=tool, spec=spec)
    if tool in IMAGE_OUTPUT_TOOLS or not isinstance(payload, (str, int, float)):
        rec.update(applied=False, reason="non-text-output")
        return payload, rec
    if isinstance(payload, str) and len(payload) > 20000:
        rec.update(applied=False, reason="payload-too-large")
        return payload, rec

    try:
        canon = to_canonical(tool, payload)
        fmt = spec.get("format") or "F0"
        length = spec.get("length") or {}
        level = length.get("level") or "L4"
        mech = length.get("mechanism") or "relevant"

        text, kept, meta = apply_length(canon, fmt, level, mech)
        pos = spec.get("position")
        pre_pos_text = text
        if pos:
            text = apply_position(text, tool, pos)

        chk = check_preservation(canon, kept, text)
        rec.update(
            applied=True,
            kind=canon["kind"],
            format=fmt,
            n_units_total=len(canon["units"]),
            n_units_kept=len(kept),
            tokens_before=ntok(canon["native"]),
            tokens_after=ntok(text),
            evidence_tokens=sum(ntok(u["text"]) for u in kept),
            irrelevant_tokens=max(0, ntok(text) - ntok(pre_pos_text)) +
                              (meta.get("target_tokens", 0) - ntok(render(canon, fmt, kept))
                               if meta.get("padded") and mech == "irrelevant" else 0),
            repetition_ratio=(len(kept) / len(canon["units"]) - 1.0
                              if canon["units"] and len(kept) > len(canon["units"]) else 0.0),
            evidence_position=pos or "native",
            identical_to_native=(text == canon["native"]),
            length_meta=meta,
            preservation=chk,
            original=canon["native"][:4000],
            transformed=text[:4000],
        )
        rec["evidence_density"] = (rec["evidence_tokens"] / rec["tokens_after"]
                                   if rec["tokens_after"] else 0.0)
        return text, rec
    except Exception as e:                                  # never break a run
        rec.update(applied=False, reason="error:%s:%s" % (type(e).__name__, e))
        return payload, rec


def spec_for(tool: str, spec_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve the per-tool spec from a proxy `taco_spec` config block."""
    if not spec_cfg:
        return None
    per_tool = spec_cfg.get("per_tool") or {}
    if tool in per_tool:
        return per_tool[tool]
    if spec_cfg.get("default") and tool not in (spec_cfg.get("exclude") or []):
        return spec_cfg["default"]
    return None
