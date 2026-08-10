"""TGB-v2 task families: eight chains over a menu where tools *conflict*.

v1's finding was that per-task masking cannot beat a global mask when the
surplus tools are inert, because then the union of every chain is already
near-optimal. v2 removes that property by construction:

    every hard tool is needed by exactly one family and damaging to several
    others, so no fixed mask can be right for all of them.

| family             | chain                        | needs        | damages       |
|--------------------|------------------------------|--------------|---------------|
| receipt_change     | OCR -> Calculator            | Calculator   | -             |
| shelf_voucher      | Count + OCR -> Calculator    | Count        | -             |
| order_corpus       | OCR + Search -> Calculator   | GoogleSearch | -             |
| gauge_over         | OCR -> UnitConvert           | UnitConvert  | money panels  |
| temp_over          | OCR -> TempConvert           | TempConvert  | most panels   |
| equation_short     | OCR -> Solver                | Solver       | most panels   |
| timetable_over     | OCR -> DurationCalc          | DurationCalc | most panels   |
| invoice_currency   | OCR -> CurrencyConvert       | CurrencyConv | money panels  |

`KnowledgeBase`, `Summarize`, `Translate`, `Barcode` are needed by nobody and
damaging to everybody -- the part a global search *can* fix, kept in the menu
so the search has something real to do.

The v1 invariant is unchanged: gold = easy_final(hard_intermediate, k), with k
in the question, so no tool output ever contains the answer.
"""
import random

from . import scenes
from .families import (f1_extract_compute, f2_visual_reason,  # noqa: F401
                       f3_retrieve_reason, gold_in, money)

CHAIN_TOOLS = ["Calculator", "UnitConvert", "TempConvert", "Solver",
               "DurationCalc", "CurrencyConvert"]
# needed by exactly one family each, so a fixed mask must compromise
UNION_TOOLS = ["OCR", "Calculator", "CountGivenObject", "GoogleSearch",
               "UnitConvert", "TempConvert", "Solver", "DurationCalc",
               "CurrencyConvert"]
NUISANCE = ["ImageDescription", "TextToBbox", "KnowledgeBase", "Summarize",
            "Translate", "Barcode"]


def _panel(title, rows):
    return dict(kind="panel", title=title, rows=rows)


def _mk(fam, question, gold, scene, plan, gt, up, corrupt_tool, meta):
    return dict(family=fam, domain=fam.split("_")[0], question=question,
                gold=gold, scene=scene, plan=plan, gt_tools=gt, upstream=up,
                corrupt_tool=corrupt_tool, meta=meta)


# ------------------------------------------------------------------ panel v2

def gauge_over(rng, idx):
    km = round(rng.uniform(12.0, 480.0), 1)
    mi = round(km * 0.621371, 2)
    allow = float(rng.randrange(5, max(6, int(mi)), 5))
    gold = round(mi - allow, 2)
    scene = _panel("TRIP LOG", [f"DISTANCE {km:g} KM"])
    q = (f"The image shows a trip log. Our travel allowance covers "
         f"{allow:.2f} miles. How many miles beyond the allowance is this "
         f"trip?")
    plan = [dict(tool="OCR", args={}, emits={"KM": {"kind": "nums", "idx": [0]}}),
            dict(tool="UnitConvert", template="{KM}", needs=["KM"],
                 arg_key="value")]
    return _mk("gauge_over", q, money(gold), scene, plan,
               ["OCR", "UnitConvert"], ["OCR"], "UnitConvert",
               dict(intermediate=f"{mi:.2f}", final_op="v_minus_k",
                    k=f"{allow:.2f}"))


def temp_over(rng, idx):
    c = round(rng.uniform(18.0, 95.0), 1)
    f = round(c * 9 / 5 + 32, 2)
    thr = float(rng.randrange(60, max(61, int(f)), 5))
    gold = round(f - thr, 2)
    scene = _panel("SENSOR PANEL", [f"READING {c:g} C"])
    q = (f"The image shows a sensor panel. The alarm threshold is "
         f"{thr:.2f} degrees Fahrenheit. How many degrees Fahrenheit above "
         f"the threshold is the reading?")
    plan = [dict(tool="OCR", args={}, emits={"C": {"kind": "nums", "idx": [0]}}),
            dict(tool="TempConvert", template="{C}", needs=["C"],
                 arg_key="value")]
    return _mk("temp_over", q, money(gold), scene, plan,
               ["OCR", "TempConvert"], ["OCR"], "TempConvert",
               dict(intermediate=f"{f:.2f}", final_op="v_minus_k",
                    k=f"{thr:.2f}"))


def equation_short(rng, idx):
    a = rng.randint(12, 49)
    x100 = rng.randint(1500, 9000)
    x = x100 / 100.0
    b = rng.randint(50, 900)
    c = round(a * x + b, 2)
    target = float(rng.randrange(int(x) + 5, int(x) + 60, 5))
    gold = round(target - x, 2)
    scene = _panel("CALIBRATION SHEET",
                   [f"COEFFICIENT {a}", f"OFFSET {b}", f"RESULT {c:g}"])
    q = (f"The image shows a calibration sheet giving a coefficient, an "
         f"offset and a result for the relation coefficient times X plus "
         f"offset equals result. We need X to reach {target:.2f}. By how much "
         f"is the current X short of that?")
    plan = [dict(tool="OCR", args={},
                 emits={"ABC": {"kind": "nums", "idx": [0, 1, 2]}}),
            dict(tool="Solver", template="{ABC}", needs=["ABC"],
                 arg_key="value")]
    return _mk("equation_short", q, money(gold), scene, plan,
               ["OCR", "Solver"], ["OCR"], "Solver",
               dict(intermediate=f"{x:.2f}", final_op="k_minus_v",
                    k=f"{target:.2f}"))


def timetable_over(rng, idx):
    h1, m1 = rng.randint(5, 12), rng.randint(0, 59)
    dur = rng.randint(95, 700)
    t2 = h1 * 60 + m1 + dur
    h2, m2 = (t2 // 60) % 24, t2 % 60
    permit = float(rng.randrange(30, max(31, dur), 30))
    gold = round(dur - permit, 2)
    scene = _panel("SERVICE TIMETABLE",
                   [f"DEPART {h1:02d} {m1:02d}", f"ARRIVE {h2:02d} {m2:02d}"])
    q = (f"The image shows a service timetable. Our permit covers "
         f"{permit:.2f} minutes of running time. By how many minutes does "
         f"this service exceed the permit?")
    plan = [dict(tool="OCR", args={},
                 emits={"T": {"kind": "nums", "idx": [0, 1, 2, 3]}}),
            dict(tool="DurationCalc", template="{T}", needs=["T"],
                 arg_key="value")]
    return _mk("timetable_over", q, money(gold), scene, plan,
               ["OCR", "DurationCalc"], ["OCR"], "DurationCalc",
               dict(intermediate=f"{dur}", final_op="v_minus_k",
                    k=f"{permit:.2f}"))


def invoice_currency(rng, idx):
    amt = round(rng.uniform(120.0, 990.0), 2)
    rate = round(rng.uniform(0.72, 1.19), 4)
    eur = round(amt * rate, 2)
    budget = float(rng.randrange(20, max(21, int(eur)), 20))
    gold = round(eur - budget, 2)
    scene = _panel("SUPPLIER INVOICE",
                   [f"SUBTOTAL USD {amt:.2f}", f"RATE {rate:g}"])
    q = (f"The image shows a supplier invoice with a subtotal and the "
         f"applicable conversion rate. Our euro budget for this order is "
         f"{budget:.2f} EUR. By how many euro does the converted subtotal "
         f"exceed the budget?")
    plan = [dict(tool="OCR", args={},
                 emits={"AR": {"kind": "nums", "idx": [0, 1]}}),
            dict(tool="CurrencyConvert", template="{AR}", needs=["AR"],
                 arg_key="value")]
    return _mk("invoice_currency", q, money(gold), scene, plan,
               ["OCR", "CurrencyConvert"], ["OCR"], "CurrencyConvert",
               dict(intermediate=f"{eur:.2f}", final_op="v_minus_k",
                    k=f"{budget:.2f}"))


FAMILIES_V2 = {
    "f1_extract_compute": f1_extract_compute,
    "f2_visual_reason": f2_visual_reason,
    "f3_retrieve_reason": f3_retrieve_reason,
    "gauge_over": gauge_over,
    "temp_over": temp_over,
    "equation_short": equation_short,
    "timetable_over": timetable_over,
    "invoice_currency": invoice_currency,
}

# which other family's chain to hand a task as the `wrong` control
WRONG_PARTNER = {
    "f1_extract_compute": ["OCR", "TempConvert"],
    "f2_visual_reason": ["OCR", "DurationCalc"],
    "f3_retrieve_reason": ["OCR", "UnitConvert"],
    "gauge_over": ["OCR", "CurrencyConvert"],
    "temp_over": ["OCR", "Solver"],
    "equation_short": ["OCR", "DurationCalc"],
    "timetable_over": ["OCR", "TempConvert"],
    "invoice_currency": ["OCR", "UnitConvert"],
}
