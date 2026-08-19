"""TGB-v2 tools: a menu where every hard tool is *needed by one family and
harmful to several others*.

This is the property TGB-v1 lacked and the reason its per-task masking could
not beat a global mask. In v1 an off-family tool was **inert**
(`CountGivenObject: 0`, `GoogleSearch: No results found`): keeping it cost
nothing, so the best constant policy was the union of every chain and there was
nothing left for per-task selection to win. Here each compute tool applies its
own transform to whatever it finds and reports the result in its own
authoritative format -- which is what a real tool does when an agent calls it
on the wrong input. On a receipt task `CurrencyConvert` therefore emits a
total-shaped number that is not the total:

    CurrencyConvert: SUBTOTAL EQUIVALENT 223.76 EUR (rate 0.9251)

TGB-v1 established that misleading numbers are lethal (the `corrupt` condition:
same tools, same context length, accuracy 0.001), so this makes the union of
all chains a genuinely bad policy and forces any fixed mask into a compromise.

Every tool has three behaviours, and all three are load-bearing:

  in-chain    `value` arrives from the upstream tool's parsed output -> correct
  blind call  no `value` (the tool is in the mask but not in this task's plan)
              -> it reads the panel anyway and emits a plausible wrong number
  orphaned    `value` is an unresolved symbol (its upstream was masked out)
              -> it errors, so a downstream tool still has no marginal alone

No executor ever sees the gold.
"""
import re

from .tools import DEFAULT_ARGS, TOOLS, garble

_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def panel_numbers(scene, boxes=None):
    """Every number the image shows, in reading order.

    Read off the *rendered glyph boxes* rather than the scene's semantic
    fields, so a tool called on a receipt or a shelf sees numbers too. Without
    this the v2 tools would be inert on the v1 families and v2 would silently
    degrade back into v1, where the union mask is already near-optimal.
    """
    if boxes:
        out = []
        for b in boxes:
            out += [float(x) for x in _NUM.findall(b["text"])]
        return out
    if scene.get("kind") != "panel":
        return []
    out = []
    for r in scene["rows"]:
        out += [float(x) for x in _NUM.findall(r)]
    return out


def _resolve(value, scene, n_expected=1, boxes=None):
    """(values, error). `value` is the chain argument; None means blind call."""
    if value is None:
        ns = panel_numbers(scene, boxes)
        # a blind call targets the *salient* figures: a real tool pointed at a
        # document goes for the decimal amounts, not the "QTY 1" integers.
        # This is also what makes the contamination dangerous rather than
        # obviously irrelevant -- the competing number is money-shaped.
        dec = [x for x in ns if x != int(x)]
        pick = dec if len(dec) >= n_expected else ns
        return (pick[:n_expected] if len(pick) >= n_expected else None), None
    parts = [p.strip() for p in str(value).split(",")]
    if any(not re.fullmatch(r"-?\d+(?:\.\d+)?", p) for p in parts):
        bad = next(p for p in parts
                   if not re.fullmatch(r"-?\d+(?:\.\d+)?", p))
        return None, f"NameError: name '{bad}' is not defined"
    return [float(p) for p in parts], None


def unit_convert(scene, boxes, rng, corrupt=False, value=None, **kw):
    """km -> miles. Needed by `gauge_over`; on any other panel it converts
    whatever number it finds and reports it as a distance."""
    ns, err = _resolve(value, scene, 1, boxes)
    if err:
        return err
    if not ns:
        return "UnitConvert: no convertible quantity found."
    mi = ns[0] * 0.621371
    if corrupt:
        mi *= rng.choice([0.87, 1.14])
    return f"UnitConvert: {ns[0]:g} KM = {mi:.2f} MI"


def temp_convert(scene, boxes, rng, corrupt=False, value=None, **kw):
    """Celsius -> Fahrenheit. Needed by `temp_over`."""
    ns, err = _resolve(value, scene, 1, boxes)
    if err:
        return err
    if not ns:
        return "TempConvert: no temperature found."
    f = ns[0] * 9 / 5 + 32
    if corrupt:
        f += rng.choice([-7.0, 9.0])
    return f"TempConvert: {ns[0]:g} C = {f:.2f} F"


def currency_convert(scene, boxes, rng, corrupt=False, value=None, **kw):
    """USD -> EUR. Needed by `invoice_currency`; elsewhere it emits a
    total-shaped competing number."""
    if value is None:
        ns = panel_numbers(scene, boxes)
        if not ns:
            return "CurrencyConvert: no monetary amount found."
        amt = max([n for n in ns if n != int(n)] or ns)
        rate = next((n for n in ns if 0.5 < n < 1.5 and n != int(n)),
                    0.9251)
    else:
        vals, err = _resolve(value, scene, 2, boxes)
        if err:
            return err
        amt, rate = vals[0], vals[1]
    eur = amt * rate
    if corrupt:
        eur *= rng.choice([0.91, 1.08])
    # rate first so the converted amount is the *last* number in the string:
    # the model-side final step reads the trailing value uniformly across tools
    return f"CurrencyConvert: at rate {rate:g}, SUBTOTAL EQUIVALENT {eur:.2f} EUR"


def duration_calc(scene, boxes, rng, corrupt=False, value=None, **kw):
    """Minutes between two clock readings. Needed by `timetable_over`;
    elsewhere it reads any four numbers as two times."""
    ns, err = _resolve(value, scene, 4, boxes)
    if err:
        return err
    if not ns:
        ns = panel_numbers(scene, boxes)
    if not ns:
        return "DurationCalc: no timestamps found."
    if len(ns) < 4:
        # fired blind on a document with no timetable: it still reports an
        # elapsed time, reading the largest figure as minutes. Returning an
        # error here instead would make the tool inert off-family, which is
        # exactly the v1 property v2 exists to remove.
        return f"DurationCalc: ELAPSED {int(round(max(ns)))} MIN"
    h1, m1, h2, m2 = ns[:4]
    mins = (h2 * 60 + m2) - (h1 * 60 + m1)
    if mins < 0:
        mins += 24 * 60
    if corrupt:
        mins += rng.choice([-23, 41])
    return f"DurationCalc: ELAPSED {int(round(mins))} MIN"


def linear_solve(scene, boxes, rng, corrupt=False, value=None,
                 expression=None, **kw):
    """Solves A*X + B = C. Needed by `equation_short`; elsewhere it takes the
    first three numbers and reports a root anyway."""
    ns, err = _resolve(value if value is not None else expression, scene, 3, boxes)
    if err:
        return err
    if not ns or len(ns) < 3 or ns[0] == 0:
        return "Solver: could not parse an equation."
    x = (ns[2] - ns[1]) / ns[0]
    if corrupt:
        x *= rng.choice([0.88, 1.13])
    return f"Solver: [X = {x:.2f}]"


def knowledge_base(scene, boxes, rng, corrupt=False, queries=(), **kw):
    """A second catalogue -- the conflicting-source control. Needed by nobody;
    it returns the same product codes at different prices, so a model that
    reads it instead of GoogleSearch derives a wrong subtotal."""
    ents = scene.get("entities")
    if not ents:
        return "KnowledgeBase: no matching record."
    blocks = []
    for i, q in enumerate(list(queries) or [""], 1):
        hit = next((e for e in ents if e["name"].lower() in str(q).lower()),
                   ents[0])
        v = float(str(hit["value"]).lstrip("$"))
        blocks.append(f"{i} - {hit['name']} internal catalogue: list price "
                      f"${v * 1.17 + 3.40:.2f} per unit (revision B).")
    return "\n\n".join(blocks)


def summarize(scene, boxes, rng, corrupt=False, **kw):
    """Lossy restatement with a rounded aggregate -- authoritative in tone and
    wrong in value. Harmful to every family."""
    ns = panel_numbers(scene, boxes)
    if not ns:
        return "Summarize: the image contains no readable figures."
    return (f"Summarize: the document lists {len(ns)} figures; the headline "
            f"amount is approximately {round(sum(ns) * 0.94, 1):g}.")


def translate(scene, boxes, rng, corrupt=False, **kw):
    """Restates whatever text is on the image, perturbing numerals the way MT
    systems do."""
    lines = [b["text"] for b in boxes] if boxes else scene.get("rows") or []
    if not lines:
        return "Translate: nothing to translate."
    return "Translate (normalised):\n" + "\n".join(
        garble(r, rng, p=0.4) for r in lines)


def barcode(scene, boxes, rng, corrupt=False, **kw):
    ns = panel_numbers(scene, boxes)
    return f"Barcode: 88{(int(sum(ns) * 100) if ns else 4711) % 100000000:08d}"


V2_TOOLS = {
    "UnitConvert": unit_convert,
    "TempConvert": temp_convert,
    "CurrencyConvert": currency_convert,
    "DurationCalc": duration_calc,
    "Solver": linear_solve,
    "KnowledgeBase": knowledge_base,
    "Summarize": summarize,
    "Translate": translate,
    "Barcode": barcode,
}


def register():
    """Install the v2 executors so `tools.run_chain` can drive them."""
    TOOLS.update(V2_TOOLS)
    DEFAULT_ARGS.update({"KnowledgeBase": {"queries": []}})
