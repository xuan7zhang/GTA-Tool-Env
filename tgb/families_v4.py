"""TGB-v4: eight text-only families. No image anywhere, no OCR anywhere.

The point of the version. v1 and v2 put OCR in every chain; v3 cut it to 11%
but still had three visual families. So no result so far cleanly separates
"the likelihood signal works for tool use" from "it works for reading text off
a picture" -- the mechanism §4a blames for the real-GTA null lives exactly in
the second reading. Here there is no picture at all.

The other lesson v3 taught, the hard way: a downstream tool is only load-
bearing if its work does not fit in the model's head. v3's `table_lookup`
returned one row, so the upstream alone (0.267) beat the full chain (0.198).
Every upstream here returns a *set* -- six rows, five readings, two passages --
so the aggregation is real work.

| family          | chain                              | why the downstream is needed |
|-----------------|------------------------------------|------------------------------|
| table_total     | TableQuery -> Calculator           | sum of 6 qty*unit products   |
| table_filter    | TableQuery -> Calculator           | sum over a filtered subset   |
| sensor_mean     | SensorAPI -> Calculator            | mean of 5 readings           |
| sensor_convert  | SensorAPI -> Calculator            | mean, then a C->F conversion |
| schedule_gap    | CalendarAPI -> Calculator          | minutes between two clocks   |
| doc_two_facts   | DocRetrieve -> Calculator          | two retrieved prices combined|
| translate_fact  | DocRetrieve -> Translate           | the figure is in cipher words|
| fx_settle       | DocRetrieve + ExchangeRate -> Calc | price x qty x rate           |
| compute_only    | Calculator (optional)              | tool-OPTIONAL control        |
| no_tool         | (none)                             | tool-FREE control            |

Invariant unchanged: gold = easy_final(hard_intermediate, k), k in the question.
"""
import re

from .families import money  # noqa: F401


def gold_in(text, gold):
    """Answer-echo check at a *number boundary*, not as a bare substring.

    v1-v3 used a plain substring test, which rejects a task whose gold is the
    decimal tail of its own intermediate: mean 47.32 with gold 7.32 looks like
    an echo and is not one -- the model would have to slice a number out of the
    middle of another number. It also has to agree with how correctness is
    scored (score_dl.correct uses the same boundary rule); an echo test stricter
    than the correctness test rejects tasks that could never have been leaked.
    """
    g = re.escape(gold.strip().lstrip("$"))
    return re.search(r"(?<![\d.])" + g + r"(?![\d])",
                     text.replace(",", "")) is not None
from .tools_v4 import encode_number

# 36 groups, not 6. With six names and a coarse round-number amount the
# question text repeated on 326 of `table_total`'s 400 tasks: the contexts
# differed but the prompts did not, which reads as a broken generator and
# gives the model a genuinely ambiguous input.
GROUPS = ("ballast filter spindle relay mount valve gasket bearing coupling "
          "impeller manifold nozzle piston rotor seal shaft sleeve strut "
          "bushing collar damper flange grommet hub insert journal keyway "
          "liner pinion quill ratchet sprocket tappet union vane washer").split()
LABELS = ["intake", "return", "bypass", "primary", "standby"]
_S1 = "Zor Kal Mir Vex Tan Qua Bri Nol Fen Dax Lir Sev".split()
_S2 = "vex bin ell dan tro nix mar quel forn ath ule ond".split()


def _mk(fam, q, gold, scene, plan, gt, up, corrupt_tool, meta):
    return dict(family=fam, domain="text", question=q, gold=gold, scene=scene,
                plan=plan, gt_tools=gt, upstream=up, corrupt_tool=corrupt_tool,
                meta=meta)


def _nonce(rng):
    nm = (rng.choice(_S1) + rng.choice(_S2)).capitalize()
    return f"{nm}-{rng.choice('QRTVX')}{rng.randint(2, 9)}"


def _table(rng, n=6, group=None):
    return [dict(id=f"R{rng.randint(100, 999)}",
                 group=group or rng.choice(GROUPS),
                 qty=rng.randint(2, 12),
                 unit=round(rng.uniform(3.0, 39.9), 2)) for _ in range(n)]


# ------------------------------------------------------------------ tables

def table_total(rng, idx):
    grp = rng.choice(GROUPS)
    rows = _table(rng, 6, grp)
    sub = round(sum(r["qty"] * r["unit"] for r in rows), 2)
    paid = round(sub + rng.uniform(1.0, 99.9), 2)
    gold = round(paid - sub, 2)
    scene = dict(kind="textless", table=rows)
    q = (f"Batch {grp} has been invoiced and we transferred {paid:.2f} "
         f"against it. By how much does the transfer exceed the batch "
         f"total?")
    plan = [dict(tool="TableQuery", args={"key": grp},
                 emits={"E": {"kind": "rows_expr"}}),
            dict(tool="Calculator", template="{E}", needs=["E"])]
    return _mk("table_total", q, money(gold), scene, plan,
               ["TableQuery", "Calculator"], ["TableQuery"], "TableQuery",
               dict(intermediate=money(sub), final_op="k_minus_v",
                    k=f"{paid:.2f}"))


def table_filter(rng, idx):
    grp = rng.choice(GROUPS)
    rows = _table(rng, 6, grp)
    thr = rng.randint(5, 9)
    keep = [r for r in rows if r["qty"] >= thr]
    if not keep:
        keep = rows[:1]
    sub = round(sum(r["qty"] * r["unit"] for r in keep), 2)
    budget = round(sub - rng.uniform(0.5, 49.9), 2)
    gold = round(sub - budget, 2)
    scene = dict(kind="textless", table=rows)
    q = (f"For batch {grp}, only the lines with a quantity of {thr} or more "
         f"are chargeable. The approved budget is {budget:.2f}. By how much do "
         f"the chargeable lines exceed the budget?")
    plan = [dict(tool="TableQuery", args={"key": grp},
                 emits={"E": {"kind": "rows_expr", "min_qty": thr}}),
            dict(tool="Calculator", template="{E}", needs=["E"])]
    return _mk("table_filter", q, money(gold), scene, plan,
               ["TableQuery", "Calculator"], ["TableQuery"], "TableQuery",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{budget:.2f}"))


# ----------------------------------------------------------------- sensors

def sensor_mean(rng, idx):
    sts = [dict(id=f"S{rng.randint(100, 999)}", unit="kPa",
                series=[round(rng.uniform(20, 180), 1) for _ in range(5)])
           for _ in range(4)]
    tgt = sts[rng.randrange(len(sts))]
    mean = round(sum(tgt["series"]) / len(tgt["series"]), 2)
    limit = round(mean - rng.uniform(0.5, 19.9), 2)
    gold = round(mean - limit, 2)
    scene = dict(kind="textless", stations=sts)
    q = (f"Station {tgt['id']} is rated to {limit:.2f} kPa. By how much does "
         f"its mean recorded pressure exceed the rating?")
    plan = [dict(tool="SensorAPI", args={"station": tgt["id"]},
                 emits={"E": {"kind": "mean_expr"}}),
            dict(tool="Calculator", template="{E}", needs=["E"])]
    return _mk("sensor_mean", q, money(gold), scene, plan,
               ["SensorAPI", "Calculator"], ["SensorAPI"], "SensorAPI",
               dict(intermediate=money(mean), final_op="v_minus_k",
                    k=f"{limit:.2f}"))


def sensor_convert(rng, idx):
    sts = [dict(id=f"T{rng.randint(100, 999)}", unit="C",
                series=[round(rng.uniform(15, 90), 1) for _ in range(5)])
           for _ in range(4)]
    tgt = sts[rng.randrange(len(sts))]
    mean = sum(tgt["series"]) / len(tgt["series"])
    f = round(mean * 9 / 5 + 32, 2)
    thr = round(f - rng.uniform(0.5, 19.9), 2)
    gold = round(f - thr, 2)
    scene = dict(kind="textless", stations=sts)
    q = (f"Station {tgt['id']} logs in Celsius. Its alarm threshold is "
         f"{thr:.2f} degrees Fahrenheit. By how many degrees Fahrenheit does "
         f"the mean of its samples exceed that threshold?")
    plan = [dict(tool="SensorAPI", args={"station": tgt["id"]},
                 emits={"E": {"kind": "mean_f_expr"}}),
            dict(tool="Calculator", template="{E}", needs=["E"])]
    return _mk("sensor_convert", q, money(gold), scene, plan,
               ["SensorAPI", "Calculator"], ["SensorAPI"], "SensorAPI",
               dict(intermediate=f"{f:.2f}", final_op="v_minus_k",
                    k=f"{thr:.2f}"))


# ---------------------------------------------------------------- calendar

def schedule_gap(rng, idx):
    ref = f"BK-{rng.randint(1000, 9999)}"
    h1, m1 = rng.randint(5, 11), rng.randint(0, 59)
    dur = rng.randint(95, 640)
    t2 = h1 * 60 + m1 + dur
    ev = [dict(ref=ref, label="depart", h=h1, m=m1),
          dict(ref=ref, label="arrive", h=(t2 // 60) % 24, m=t2 % 60)]
    for _ in range(2):
        ev.append(dict(ref=f"BK-{rng.randint(1000, 9999)}", label="other",
                       h=rng.randint(0, 23), m=rng.randint(0, 59)))
    # not a multiple of 30: with permit = int(dur/30)*30 the answer was
    # dur mod 30, so all 400 tasks shared 29 distinct golds and guessing paid
    permit = dur - rng.randint(5, min(240, dur - 1))
    gold = dur - permit          # minutes are a count, not an amount
    scene = dict(kind="textless", events=ev)
    q = (f"Booking {ref} covers a departure and an arrival. Our permit allows "
         f"{permit} minutes of running time. By how many minutes does the "
         f"booking exceed the permit?")
    plan = [dict(tool="CalendarAPI", args={"ref": ref},
                 emits={"E": {"kind": "gap_expr"}}),
            dict(tool="Calculator", template="{E}", needs=["E"])]
    return _mk("schedule_gap", q, str(gold), scene, plan,
               ["CalendarAPI", "Calculator"], ["CalendarAPI"], "CalendarAPI",
               dict(intermediate=str(dur), final_op="v_minus_k",
                    k=str(permit)))


# --------------------------------------------------------------- retrieval

def _docs(rng, n=6):
    out = []
    for _ in range(n):
        k = _nonce(rng)
        out.append(dict(key=k, price=round(rng.uniform(6.0, 79.9), 2),
                        text=""))
    for d in out:
        d["text"] = (f"{d['key']} is supplied in single units at a listed "
                     f"price of {d['price']:.2f} per unit.")
    return out


def doc_two_facts(rng, idx):
    docs = _docs(rng)
    a, b = rng.sample(range(len(docs)), 2)
    qa, qb = rng.randint(3, 9), rng.randint(3, 9)
    sub = round(qa * docs[a]["price"] + qb * docs[b]["price"], 2)
    coupon = float(int(sub / 50) * 50)
    gold = round(sub - coupon, 2)
    scene = dict(kind="textless", docs=docs)
    # comparative phrasing on purpose. "A coupon applies, what is payable?"
    # let the model report the Calculator's subtotal verbatim on 86% of tasks
    # (7B acc 0.163, 14B 0.015); "by how much does X exceed Y" -- the form the
    # families that work already use -- makes the remaining step explicit.
    q = (f"We are ordering {qa} units of {docs[a]['key']} and {qb} units of "
         f"{docs[b]['key']}. Our approved budget is {coupon:.2f}. By how much "
         f"does the order total exceed the budget?")
    plan = [dict(tool="DocRetrieve",
                 args={"queries": [docs[a]["key"], docs[b]["key"]]},
                 emits={"E": {"kind": "two_price_expr", "qa": qa, "qb": qb}}),
            dict(tool="Calculator", template="{E}", needs=["E"])]
    return _mk("doc_two_facts", q, money(gold), scene, plan,
               ["DocRetrieve", "Calculator"], ["DocRetrieve"], "DocRetrieve",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{coupon:.2f}"))


def translate_fact(rng, idx):
    docs = _docs(rng)
    tgt = docs[rng.randrange(len(docs))]
    val = rng.randint(120, 980)
    tgt["text"] = (f"{tgt['key']}: the certified capacity is recorded as "
                   f"{encode_number(val)} in the supplier's own notation.")
    # not a multiple of 100: quota = int(val/100)*100 made gold = val mod 100,
    # collapsing 400 tasks onto 96 distinct answers
    quota = val - rng.randint(10, min(400, val - 1))
    gold = val - quota           # a capacity is a count
    scene = dict(kind="textless", docs=docs)
    q = (f"The certificate for {tgt['key']} records a capacity in the "
         f"supplier's own numeral words. Our quota is {quota:.0f}. By how much "
         f"does the certified capacity exceed the quota?")
    plan = [dict(tool="DocRetrieve", args={"queries": [tgt["key"]]},
                 emits={"W": {"kind": "cipher_words"}}),
            dict(tool="Translate", template="{W}", needs=["W"],
                 arg_key="text")]
    # money() format, not str(int(...)): final_step always emits two decimals,
    # so an integer-formatted gold can never match and every task is rejected
    return _mk("translate_fact", q, str(gold), scene, plan,
               ["DocRetrieve", "Translate"], ["DocRetrieve"], "DocRetrieve",
               dict(intermediate=str(val), final_op="v_minus_k",
                    k=str(quota)))


def fx_settle(rng, idx):
    docs = _docs(rng)
    tgt = docs[rng.randrange(len(docs))]
    qty = rng.randint(4, 14)
    rate = round(rng.uniform(0.65, 1.45), 3)
    sub = round(qty * tgt["price"] * rate, 2)
    prepaid = float(int(sub / 20) * 20)
    gold = round(sub - prepaid, 2)
    scene = dict(kind="textless", docs=docs, rate=rate)
    q = (f"We are settling {qty} units of {tgt['key']} in the settlement "
         f"currency, and {prepaid:.2f} has already been prepaid. By how much "
         f"does the settlement value exceed the amount already prepaid?")
    plan = [dict(tool="DocRetrieve", args={"queries": [tgt["key"]]},
                 emits={"P": {"kind": "money", "pattern":
                              r"price of (\d+\.\d{2})"}}),
            dict(tool="ExchangeRate", args={},
                 emits={"R": {"kind": "money", "pattern":
                              r"= (\d+\.\d+) settlement"}}),
            dict(tool="Calculator", template=f"{qty} * {{P}} * {{R}}",
                 needs=["P", "R"])]
    return _mk("fx_settle", q, money(gold), scene, plan,
               ["DocRetrieve", "ExchangeRate", "Calculator"],
               ["DocRetrieve", "ExchangeRate"], "DocRetrieve",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{prepaid:.2f}"))


# ------------------------------------------------------- optional and free

def compute_only(rng, idx):
    a = round(rng.uniform(11.0, 89.9), 2)
    n = rng.randint(4, 19)
    sub = round(a * n, 2)
    disc = float(int(sub / 20) * 20)
    gold = round(sub - disc, 2)
    q = (f"A part costs {a:.2f} and we need {n} of them. Our approved budget "
         f"is {disc:.2f}. By how much does the order total exceed the "
         f"budget?")
    return _mk("compute_only", q, money(gold), dict(kind="textless"),
               [dict(tool="Calculator", template=f"{a:.2f} * {n}", needs=[])],
               ["Calculator"], [], "Calculator",
               dict(intermediate=money(sub), final_op="v_minus_k",
                    k=f"{disc:.2f}"))


def no_tool(rng, idx):
    a, b = rng.randint(120, 989), rng.randint(3, 97)
    q = (f"A rack held {a} units this morning and {b} were issued. How many "
         f"units remain?")
    return _mk("no_tool", q, str(a - b), dict(kind="textless"), [], [], [],
               None, dict(intermediate=str(a - b), final_op="none", k="0"))


FAMILIES_V4 = {
    "table_total": table_total,
    "table_filter": table_filter,
    "sensor_mean": sensor_mean,
    "sensor_convert": sensor_convert,
    "schedule_gap": schedule_gap,
    "doc_two_facts": doc_two_facts,
    "translate_fact": translate_fact,
    "fx_settle": fx_settle,
    "compute_only": compute_only,
    "no_tool": no_tool,
}

UNION_TOOLS_V4 = ["TableQuery", "SensorAPI", "CalendarAPI", "DocRetrieve",
                  "Translate", "ExchangeRate", "Calculator"]
NUISANCE_V4 = ["Summarize", "Barcode", "Solver", "UnitConvert", "TempConvert",
               "CurrencyConvert", "DurationCalc", "GoogleSearch"]
WRONG_PARTNER_V4 = {
    "table_total": ["SensorAPI", "Calculator"],
    "table_filter": ["CalendarAPI", "Calculator"],
    "sensor_mean": ["TableQuery", "Calculator"],
    "sensor_convert": ["DocRetrieve", "Calculator"],
    "schedule_gap": ["SensorAPI", "Calculator"],
    "doc_two_facts": ["TableQuery", "Calculator"],
    "translate_fact": ["TableQuery", "Translate"],
    "fx_settle": ["CalendarAPI", "Calculator"],
    "compute_only": ["DocRetrieve", "Summarize"],
    "no_tool": ["Summarize", "Barcode"],
}
