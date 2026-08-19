"""TGB-v3 families: nine chains, OCR in exactly one of them.

v1 (3/3) and v2 (8/8) put OCR in every chain, so every likelihood result so far
is really a result about read-text-then-compute -- the same extraction-heavy
regime §4a blames for the real-GTA null. v3 sources the upstream elsewhere and
keeps one OCR family inside the same benchmark as a control, so the comparison
is within-benchmark rather than across papers.

| family            | upstream                     | image? | text on it? |
|-------------------|------------------------------|--------|-------------|
| count_price       | CountGivenObject             | yes    | **no**      |
| count_compare     | CountGivenObject (twice)     | yes    | **no**      |
| attr_size         | RegionAttributeDescription   | yes    | **no**      |
| retrieve_calc     | GoogleSearch                 | no     | -           |
| table_lookup      | TableQuery                   | no     | -           |
| sensor_convert    | SensorAPI                    | no     | -           |
| compute_only      | none (numbers in question)    | no    | -           |
| no_tool           | none                          | no    | -           |
| ocr_control       | OCR                          | yes    | yes         |

`compute_only` and `no_tool` are the first tool-optional and tool-free tasks in
any TGB version. Without them Acc(E_none) is identically zero and dA collapses
onto Acc, which makes the dL-vs-dA correlation the meeting plan asks for
degenerate. Here E_none is genuinely non-zero.

Invariant unchanged: gold = easy_final(hard_intermediate, k), k in the question.
"""
import random

from . import scenes
from .families import gold_in, money  # noqa: F401

OBJ = "cup box bottle plate book can mug crate jar tin bowl carton".split()
SHAPE = ["circle", "square", "triangle"]
COLOR = ["red", "blue", "green", "yellow", "purple"]
_S1 = "Zor Kal Mir Vex Tan Qua Bri Nol Fen Dax Lir Sev Pol Yun Cra Hib".split()
_S2 = "vex bin ell dan tro nix mar quel forn ic ath ule ond eryn".split()
LABELS = ["ballast unit", "filter cell", "drive spindle", "relay pack",
          "sensor mount", "valve seat", "rotor clip", "bearing shell"]


def _mk(fam, q, gold, scene, plan, gt, up, corrupt_tool, meta):
    return dict(family=fam, domain=fam.split("_")[0], question=q, gold=gold,
                scene=scene, plan=plan, gt_tools=gt, upstream=up,
                corrupt_tool=corrupt_tool, meta=meta)


def _nonce(rng):
    nm = (rng.choice(_S1) + rng.choice(_S2)).capitalize()
    return f"{nm} {nm[:3].upper()}-{rng.choice('QRTVX')}{rng.randint(2, 9)}"


# --------------------------------------------------- counting (no text at all)

def count_price(rng, idx):
    tgt = rng.choice(OBJ)
    n = rng.randint(3, 11)
    others = [o for o in OBJ if o != tgt]
    ds = [(o, rng.randint(1, 4)) for o in rng.sample(others, 2)]
    price = round(rng.uniform(2.5, 24.9), 2)
    sub = round(n * price, 2)
    voucher = float(rng.randrange(5, max(6, int(sub)), 5))
    gold = round(sub - voucher, 2)
    scene = dict(kind="objects", target=tgt, n_target=n,
                 distractors=[dict(name=o, count=c) for o, c in ds])
    q = (f"The image shows a tray of items. Each {tgt} costs "
         f"{price:.2f} dollars, and I hold a {voucher:.2f} dollar voucher. "
         f"If I buy every {tgt} in the image, how much do I still owe?")
    plan = [dict(tool="CountGivenObject", args={"text": f"the {tgt}s"},
                 emits={"N": {"kind": "int"}}),
            dict(tool="Calculator", template=f"{{N}} * {price:.2f}",
                 needs=["N"])]
    return _mk("count_price", q, money(gold), scene, plan,
               ["CountGivenObject", "Calculator"], ["CountGivenObject"],
               "CountGivenObject",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{voucher:.2f}", n_target=n))


def count_compare(rng, idx):
    a, b = rng.sample(OBJ, 2)
    na, nb = rng.randint(6, 14), rng.randint(1, 5)
    rate = round(rng.uniform(3.0, 19.9), 2)
    sub = round((na - nb) * rate, 2)
    fee = float(rng.randrange(5, max(6, int(sub)), 5))
    gold = round(sub - fee, 2)
    scene = dict(kind="objects", target=a, n_target=na,
                 distractors=[dict(name=b, count=nb)])
    q = (f"The image shows a tray. Each surplus {a} beyond the number of "
         f"{b}s is charged at {rate:.2f} dollars, and a {fee:.2f} dollar "
         f"credit applies. What is the net charge?")
    plan = [dict(tool="CountGivenObject", args={"text": f"the {a}s"},
                 emits={"NA": {"kind": "int"}}),
            dict(tool="Calculator",
                 template=f"({{NA}} - {nb}) * {rate:.2f}", needs=["NA"])]
    return _mk("count_compare", q, money(gold), scene, plan,
               ["CountGivenObject", "Calculator"], ["CountGivenObject"],
               "CountGivenObject",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{fee:.2f}"))


# ------------------------------------------------- attributes (no text at all)

def attr_size(rng, idx):
    shapes = []
    names = rng.sample(SHAPE, 3)
    cols = rng.sample(COLOR, 3)
    sizes = rng.sample(range(20, 95), 3)
    for n, c, s in zip(names, cols, sizes):
        shapes.append(dict(name=n, color=c, size=s))
    shapes.sort(key=lambda s: -s["size"])
    tgt = shapes[0]
    rate = round(rng.uniform(1.5, 9.9), 2)
    sub = round(tgt["size"] * rate, 2)
    allow = float(rng.randrange(10, max(11, int(sub)), 10))
    gold = round(sub - allow, 2)
    scene = dict(kind="shapes", shapes=shapes)
    q = (f"The image shows several coloured objects. Coating costs "
         f"{rate:.2f} dollars per unit of width, and {allow:.2f} dollars of "
         f"the bill is covered. For the largest object in the frame, how much "
         f"is left to pay?")
    plan = [dict(tool="RegionAttributeDescription",
                 args={"value": tgt["name"]},
                 emits={"W": {"kind": "int"}}),
            dict(tool="Calculator", template=f"{{W}} * {rate:.2f}",
                 needs=["W"])]
    return _mk("attr_size", q, money(gold), scene, plan,
               ["RegionAttributeDescription", "Calculator"],
               ["RegionAttributeDescription"], "RegionAttributeDescription",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{allow:.2f}"))


# ------------------------------------------------------- no image at all

def retrieve_calc(rng, idx):
    ents = []
    for _ in range(12):
        nm = _nonce(rng)
        ents.append(dict(name=nm, category=rng.choice(LABELS),
                         attr="unit price",
                         value=f"${round(rng.uniform(4.0, 89.9), 2):.2f}",
                         blurb=f"{nm} is stocked by authorised distributors."))
    tgt = ents[rng.randrange(len(ents))]
    qty = rng.randint(3, 9)
    price = float(tgt["value"][1:])
    sub = round(qty * price, 2)
    coupon = float(rng.randrange(10, max(11, int(sub)), 10))
    gold = round(sub - coupon, 2)
    scene = dict(kind="textless", entities=ents)
    q = (f"We are ordering {qty} units of {tgt['name']}. A {coupon:.2f} "
         f"dollar trade coupon applies to the order. What is the amount "
         f"payable? There is no document to read; the catalogue must be "
         f"looked up.")
    plan = [dict(tool="GoogleSearch",
                 args={"queries": [f"{tgt['name']} unit price"]},
                 emits={"P": {"kind": "money", "pattern": r"\$(\d+\.\d{2})"}}),
            dict(tool="Calculator", template=f"{qty} * {{P}}", needs=["P"])]
    return _mk("retrieve_calc", q, money(gold), scene, plan,
               ["GoogleSearch", "Calculator"], ["GoogleSearch"],
               "GoogleSearch",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{coupon:.2f}"))


def table_lookup(rng, idx):
    rows = []
    for i in range(8):
        rows.append(dict(id=f"R-{rng.randint(1000, 9999)}",
                         label=rng.choice(LABELS),
                         qty=rng.randint(2, 14),
                         unit=round(rng.uniform(3.0, 59.9), 2)))
    tgt = rows[rng.randrange(len(rows))]
    sub = round(tgt["qty"] * tgt["unit"], 2)
    rebate = float(rng.randrange(10, max(11, int(sub)), 10))
    gold = round(sub - rebate, 2)
    scene = dict(kind="textless", table=rows)
    q = (f"Record {tgt['id']} in the inventory system needs valuing. A "
         f"{rebate:.2f} dollar rebate applies to its line total. What is the "
         f"net line value?")
    plan = [dict(tool="TableQuery", args={"key": tgt["id"]},
                 emits={"E": {"kind": "table_expr"}}),
            dict(tool="Calculator", template="{E}", needs=["E"])]
    return _mk("table_lookup", q, money(gold), scene, plan,
               ["TableQuery", "Calculator"], ["TableQuery"], "TableQuery",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{rebate:.2f}"))


def sensor_convert(rng, idx):
    sts = []
    for _ in range(5):
        sts.append(dict(id=f"S{rng.randint(10, 99)}",
                        reading=round(rng.uniform(12.0, 95.0), 1),
                        unit="C"))
    tgt = sts[rng.randrange(len(sts))]
    f = round(tgt["reading"] * 9 / 5 + 32, 2)
    thr = float(rng.randrange(60, max(61, int(f)), 5))
    gold = round(f - thr, 2)
    scene = dict(kind="textless", stations=sts)
    q = (f"Station {tgt['id']} must be checked against an alarm threshold of "
         f"{thr:.2f} degrees Fahrenheit. By how many degrees Fahrenheit does "
         f"its current reading exceed the threshold?")
    plan = [dict(tool="SensorAPI", args={"station": tgt["id"]},
                 emits={"C": {"kind": "nums", "idx": [0]}}),
            dict(tool="TempConvert", template="{C}", needs=["C"],
                 arg_key="value")]
    return _mk("sensor_convert", q, money(gold), scene, plan,
               ["SensorAPI", "TempConvert"], ["SensorAPI"], "SensorAPI",
               dict(intermediate=f"{f:.2f}", final_op="v_minus_k",
                    k=f"{thr:.2f}"))


# ------------------------------------------- tool-optional and tool-free

def compute_only(rng, idx):
    """Every number is already in the question; the arithmetic is what is hard.
    A Calculator helps but nothing has to be perceived or retrieved -- this is
    the tool-OPTIONAL class, so Acc(E_none) is genuinely above zero."""
    a = round(rng.uniform(11.0, 89.9), 2)
    n = rng.randint(4, 19)
    sub = round(a * n, 2)
    disc = float(rng.randrange(20, max(21, int(sub)), 20))
    gold = round(sub - disc, 2)
    scene = dict(kind="textless")
    q = (f"A part costs {a:.2f} dollars and we need {n} of them. A "
         f"{disc:.2f} dollar bulk discount applies to the order total. What "
         f"is the amount payable?")
    plan = [dict(tool="Calculator", template=f"{a:.2f} * {n}", needs=[])]
    return _mk("compute_only", q, money(gold), scene, plan,
               ["Calculator"], [], "Calculator",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{disc:.2f}"))


def no_tool(rng, idx):
    """Answerable directly. The tool-FREE class: the best environment is the
    empty one, and any tool output is pure distraction."""
    a = rng.randint(12, 89)
    b = rng.randint(3, 9)
    gold = a - b
    scene = dict(kind="textless")
    q = (f"A shelf held {a} boxes this morning and {b} were removed. How many "
         f"boxes remain on the shelf?")
    return _mk("no_tool", q, str(gold), scene, [],
               [], [], None,
               dict(intermediate=str(gold), final_op="none", k="0"))


# ------------------------------------------------------------- OCR control

def ocr_control(rng, idx):
    from .families import f1_extract_compute
    t = f1_extract_compute(rng, idx)
    t["family"] = "ocr_control"
    return t


FAMILIES_V3 = {
    "count_price": count_price,
    "count_compare": count_compare,
    "attr_size": attr_size,
    "retrieve_calc": retrieve_calc,
    "table_lookup": table_lookup,
    "sensor_convert": sensor_convert,
    "compute_only": compute_only,
    "no_tool": no_tool,
    "ocr_control": ocr_control,
}

UNION_TOOLS_V3 = ["OCR", "Calculator", "CountGivenObject", "GoogleSearch",
                  "RegionAttributeDescription", "TableQuery", "SensorAPI",
                  "TempConvert"]
NUISANCE_V3 = ["ImageDescription", "TextToBbox", "Summarize", "Translate",
               "Barcode", "UnitConvert", "CurrencyConvert", "Solver",
               "DurationCalc"]
WRONG_PARTNER_V3 = {
    "count_price": ["RegionAttributeDescription", "Calculator"],
    "count_compare": ["OCR", "Calculator"],
    "attr_size": ["CountGivenObject", "Calculator"],
    "retrieve_calc": ["TableQuery", "Calculator"],
    "table_lookup": ["GoogleSearch", "Calculator"],
    "sensor_convert": ["OCR", "TempConvert"],
    "compute_only": ["OCR", "Summarize"],
    "no_tool": ["Summarize", "Translate"],
    "ocr_control": ["CountGivenObject", "Calculator"],
}
