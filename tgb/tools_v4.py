"""TGB-v4 tools: text-only sources, and downstreams that are actually needed.

v3 showed the failure mode to avoid. Its `table_lookup` chain returned a single
row -- `quantity 13 | unit value 14.65` -- so the model could just multiply the
two numbers itself, and accuracy with the upstream *alone* (0.267) beat the
full chain (0.198). A downstream tool whose work fits in the model's head is
not load-bearing, whatever the chain diagram says.

Every upstream here therefore returns a **set** of records, readings or
passages. Aggregating six rows or five readings does not fit in the head, so
the downstream Calculator earns its place, and the "no single tool suffices"
property of the benchmark is real rather than declared.

No tool here touches an image. That is the point: v1/v2 put OCR in every chain
and v3 still had three visual families, so nothing so far separates "likelihood
works for tool use" from "likelihood works for reading text off a picture".
"""
import re

from .tools import DEFAULT_ARGS, TOOLS

_ROWS = 6


def table_query(scene, boxes, rng, corrupt=False, key=None, **kw):
    """Return every row of the requested table, not one row.

    Multi-row output is what makes the downstream necessary; it is also what a
    real query interface returns.
    """
    rows = scene.get("table")
    if not rows:
        return "TableQuery: no table matches that request."
    sel = rows
    if key:
        sel = [r for r in rows if str(key).lower() in str(r["group"]).lower()] \
            or rows
    if corrupt:
        sel = [dict(r, qty=max(1, r["qty"] + rng.choice([-3, 4])))
               for r in sel]
    head = "TableQuery: %d rows\n" % len(sel)
    return head + "\n".join(
        f"  {r['id']} | {r['group']} | qty {r['qty']} | unit {r['unit']:.2f}"
        for r in sel)


def sensor_api(scene, boxes, rng, corrupt=False, station=None, **kw):
    """Return the whole reading series for a station."""
    st = scene.get("stations")
    if not st:
        return "SensorAPI: no station in range."
    hit = None
    if station:
        hit = next((s for s in st if str(s["id"]).lower() in str(station).lower()),
                   None)
    if hit is None:
        hit = st[0]
    vals = hit["series"]
    if corrupt:
        vals = [round(v * rng.choice([0.85, 1.16]), 1) for v in vals]
    return (f"SensorAPI: station {hit['id']}, {len(vals)} samples\n  "
            + ", ".join(f"{v:g}" for v in vals) + f"  ({hit['unit']})")


def calendar_api(scene, boxes, rng, corrupt=False, ref=None, **kw):
    """Return the events on a booking reference, as clock times."""
    ev = scene.get("events")
    if not ev:
        return "CalendarAPI: no bookings found."
    sel = ev
    if ref:
        sel = [e for e in ev if str(ref).lower() in str(e["ref"]).lower()] or ev
    if corrupt:
        sel = [dict(e, m=(e["m"] + rng.choice([-17, 23])) % 60) for e in sel]
    return ("CalendarAPI: %d bookings\n" % len(sel)) + "\n".join(
        f"  {e['ref']} | {e['label']} | {e['h']:02d} {e['m']:02d}" for e in sel)


def doc_retrieve(scene, boxes, rng, corrupt=False, queries=(), **kw):
    """Passage retrieval over the task's corpus. Returns the matching passage
    plus one neighbour, the way a real retriever does."""
    docs = scene.get("docs")
    if not docs:
        return "DocRetrieve: no passage matched."
    out, i = [], 1
    for q in list(queries) or [""]:
        hit = next((d for d in docs if d["key"].lower() in str(q).lower()),
                   docs[0])
        if corrupt:
            alts = [d for d in docs if d is not hit]
            if alts:
                hit = alts[rng.randrange(len(alts))]
        out.append(f"[{i}] {hit['key']} — {hit['text']}")
        i += 1
        nb = docs[(docs.index(hit) + 2) % len(docs)]
        out.append(f"[{i}] {nb['key']} — general background, no figures given.")
        i += 1
    return "DocRetrieve:\n" + "\n".join(out)


# A deliberately opaque encoding: the corpus stores the number in words of a
# constructed language, so the passage is unusable until Translate runs. This
# is what makes Translate load-bearing here rather than the nuisance tool it
# was in v2.
_WORDS = ["nul", "ein", "dvo", "tri", "kvar", "kvin", "ses", "sep", "ok",
          "non"]


def encode_number(x):
    return " ".join(_WORDS[int(d)] for d in f"{x:.0f}")


def translate(scene, boxes, rng, corrupt=False, text=None, **kw):
    """Decode the constructed-language numerals in whatever passage it is
    given. Fired blind (no text) it has nothing to decode."""
    if not text:
        return "Translate: no source text supplied."
    idx = {w: str(i) for i, w in enumerate(_WORDS)}
    toks = str(text).split()
    digits = [idx[t] for t in toks if t in idx]
    if not digits:
        return "Translate: the source contains no recognised numerals."
    v = int("".join(digits))
    if corrupt:
        v = int(v * rng.choice([0.87, 1.14])) or 1
    return f"Translate: the passage states the value {v}"


def exchange_rate(scene, boxes, rng, corrupt=False, pair=None, **kw):
    r = scene.get("rate")
    if r is None:
        return "ExchangeRate: pair not quoted."
    if corrupt:
        r = round(r * rng.choice([0.9, 1.11]), 4)
    return f"ExchangeRate: 1 unit = {r:g} settlement units"


V4_TOOLS = {
    "TableQuery": table_query,
    "SensorAPI": sensor_api,
    "CalendarAPI": calendar_api,
    "DocRetrieve": doc_retrieve,
    "Translate": translate,
    "ExchangeRate": exchange_rate,
}


def register():
    TOOLS.update(V4_TOOLS)
    DEFAULT_ARGS.update({"TableQuery": {}, "SensorAPI": {},
                         "CalendarAPI": {}, "DocRetrieve": {"queries": []},
                         "ExchangeRate": {}})
