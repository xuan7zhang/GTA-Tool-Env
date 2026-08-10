"""TGB task families.

Every task obeys one structural invariant, which is the whole reason this
benchmark can test *tool* utility rather than *evidence* utility:

    gold = easy_final( hard_intermediate , k )

`hard_intermediate` is what the tool chain computes (a multi-item subtotal, a
count times a price, a two-line order value). `k` is a constant that lives in
the **question** (cash paid, a voucher, a coupon). Consequences, all of them
enforced by assertion at generation time:

  * NON-ECHO -- no tool output contains gold, because every tool output stops
    one easy step short of it.
  * The upstream tool is necessary: the intermediate cannot be obtained from
    the question, only by reading/counting/retrieving.
  * The downstream tool has no marginal without its upstream: its argument is
    built by parsing the upstream output, so without it the Calculator gets an
    unresolved symbol and errors -- condition (3) made mechanical.
  * The final step is easy (one subtraction against a round number), so a
    model that has the intermediate can finish, and a model that only has raw
    numbers has to do the hard part in its head. That difference is exactly
    what the coalition ranking is supposed to detect.

Families:
  F1 extract_compute  receipt image      OCR -> Calculator
  F2 visual_reason    shelf image        CountGivenObject + OCR -> Calculator
  F3 retrieve_reason  order form + corpus  OCR + GoogleSearch -> Calculator
"""
import re

from . import scenes

# prices read off an image carry the USD marker the renderer draws; prices read
# out of a retrieval snippet carry the "$" the search tool prints
MONEY_OCR = r"USD\s*(\d+\s*\.\s*\d{2})"
MONEY_WEB = r"\$(\d+\.\d{2})"

ITEMS = {
    "grocery": "Milk Eggs Bread Butter Cheese Rice Beans Flour Sugar Oil Pasta Cereal".split(),
    "hardware": "Hammer Nails Screws Drill Wrench Bolt Tape Glue Brush Paint Sander Clamp".split(),
    "office": "Pen Notebook Stapler Folder Marker Eraser Ruler Clip Ink Paper Binder".split(),
    "cafe": "Coffee Sandwich Salad Water Bagel Chips Juice Muffin Soup Cookie Tea".split(),
}
SHELF_OBJ = "cup box bottle plate book can mug crate jar tin bowl carton".split()

_SYL1 = "Zor Kal Mir Vex Tan Qua Bri Nol Fen Dax Lir Sev Pol Yun Cra Hib".split()
_SYL2 = "vex bin ell dan tro nix mar quel forn ic ath ule ond eryn".split()
_CATS = ["industrial sensor", "bench tool", "lab consumable", "field module",
         "control unit", "test cartridge"]


def money(x):
    return f"{x:.2f}"


def _round_up(x, step):
    return float(int(x / step + 1) * step)


# --------------------------------------------------------------- F1: receipt

def f1_extract_compute(rng, idx):
    dom = rng.choice(list(ITEMS))
    n = rng.randint(3, 6)
    names = rng.sample(ITEMS[dom], n)
    items = [(nm, rng.randint(1, 4), round(rng.uniform(1.2, 39.9), 2))
             for nm in names]
    tax = rng.choice([5, 7, 8, 10])
    layout = rng.choice(["receipt", "table"])
    scene = scenes.receipt_scene(rng, items, tax, layout)
    sub = sum(q * u for _, q, u in items)
    total = round(sub * (1 + tax / 100), 2)
    paid = _round_up(total, rng.choice([20.0, 50.0, 100.0]))
    gold = round(paid - total, 2)
    q = (f"The image shows a receipt from {scene['store']}. I paid the full "
         f"amount, tax included, with ${money(paid)} in cash. How much change "
         f"should I get back?")
    plan = [
        dict(tool="OCR", args={}, emits={"TOTAL_EXPR": {"kind": "receipt_expr"}}),
        dict(tool="Calculator", template="{TOTAL_EXPR}", needs=["TOTAL_EXPR"]),
    ]
    return dict(family="f1_extract_compute", domain=dom, question=q,
                gold=money(gold), scene=scene, plan=plan,
                gt_tools=["OCR", "Calculator"], upstream=["OCR"],
                corrupt_tool="OCR",
                meta=dict(intermediate=money(total), final_op="paid_minus_total",
                          k=money(paid), n_items=n, tax=tax, layout=layout))


# ----------------------------------------------------------------- F2: shelf

def f2_visual_reason(rng, idx):
    tgt = rng.choice(SHELF_OBJ)
    others = [o for o in SHELF_OBJ if o != tgt]
    n_t = rng.randint(3, 11)
    ds = rng.sample(others, rng.randint(2, 3))
    budget = 30 - n_t
    counts = []
    for o in ds:
        c = rng.randint(1, max(1, min(5, budget - (len(ds) - len(counts) - 1))))
        counts.append((o, c))
        budget -= c
    price = round(rng.uniform(1.5, 24.9), 2)
    scene = scenes.shelf_scene(rng, tgt, n_t, counts, price)
    sub = round(n_t * price, 2)
    voucher = float(rng.randrange(5, max(6, int(sub)), 5)) if sub > 10 else 5.0
    voucher = min(voucher, float(int(sub)))
    gold = round(sub - voucher, 2)
    q = (f"The image shows shelf {scene['shop']}. I want to buy every "
         f"{tgt} on that shelf and I have a ${money(voucher)} voucher to use "
         f"against the purchase. How much do I still have to pay?")
    plan = [
        dict(tool="CountGivenObject", args={"text": f"the {tgt}s"},
             emits={"N": {"kind": "int"}}),
        dict(tool="OCR", args={}, emits={"P": {"kind": "money", "pattern": MONEY_OCR}}),
        dict(tool="Calculator", template="{N} * {P}", needs=["N", "P"]),
    ]
    return dict(family="f2_visual_reason", domain="shelf", question=q,
                gold=money(gold), scene=scene, plan=plan,
                gt_tools=["CountGivenObject", "OCR", "Calculator"],
                upstream=["CountGivenObject", "OCR"], corrupt_tool="OCR",
                meta=dict(intermediate=money(sub), final_op="subtotal_minus_voucher",
                          k=money(voucher), n_target=n_t, unit_price=money(price)))


# ------------------------------------------------------------------ F3: form

def _corpus(rng, n=14):
    ents, seen = [], set()
    while len(ents) < n:
        nm = (rng.choice(_SYL1) + rng.choice(_SYL2)).capitalize()
        code = f"{nm[:3].upper()}-{rng.choice('QRTVX')}{rng.randint(2, 9)}"
        if code in seen:
            continue
        seen.add(code)
        cat = rng.choice(_CATS)
        val = round(rng.uniform(2.5, 89.9), 2)
        ents.append((f"{nm} {code}", cat,
                     "unit price", f"${val:.2f}",
                     f"{nm} {code} is a {cat} sold through authorised "
                     f"distributors in single units and trade packs."))
    return ents


def f3_retrieve_reason(rng, idx):
    ents = _corpus(rng)
    a, b = rng.sample(range(len(ents)), 2)
    qa, qb = rng.randint(2, 9), rng.randint(2, 9)
    order = [dict(code=ents[a][0], qty=qa), dict(code=ents[b][0], qty=qb)]
    scene = dict(kind="form", order=order,
                 entities=[dict(name=n, category=c, attr=at, value=v, blurb=bl)
                           for n, c, at, v, bl in ents],
                 ref=f"PO-{rng.randint(10000, 99999)}")
    pa = float(ents[a][3][1:])
    pb = float(ents[b][3][1:])
    sub = round(qa * pa + qb * pb, 2)
    coupon = float(rng.randrange(10, max(11, int(sub)), 10))
    coupon = min(coupon, float(int(sub)))
    gold = round(sub - coupon, 2)
    q = (f"The image shows purchase order {scene['ref']}. The two product "
         f"codes on it are not priced on the form. Our supplier gave us a "
         f"${money(coupon)} trade coupon for this order. After the coupon, "
         f"what is the amount payable?")
    plan = [
        dict(tool="OCR", args={},
             emits={"QA": {"kind": "int_idx", "idx": 0},
                    "QB": {"kind": "int_idx", "idx": 1}}),
        dict(tool="GoogleSearch",
             args={"queries": [f"{ents[a][0]} unit price",
                               f"{ents[b][0]} unit price"]},
             emits={"PA": {"kind": "money_idx", "pattern": MONEY_WEB, "idx": 0},
                    "PB": {"kind": "money_idx", "pattern": MONEY_WEB, "idx": 1}}),
        dict(tool="Calculator", template="{QA} * {PA} + {QB} * {PB}",
             needs=["QA", "PA", "QB", "PB"]),
    ]
    return dict(family="f3_retrieve_reason", domain="procurement", question=q,
                gold=money(gold), scene=scene, plan=plan,
                gt_tools=["OCR", "GoogleSearch", "Calculator"],
                upstream=["OCR", "GoogleSearch"], corrupt_tool="GoogleSearch",
                meta=dict(intermediate=money(sub), final_op="subtotal_minus_coupon",
                          k=money(coupon), codes=[ents[a][0], ents[b][0]],
                          qtys=[qa, qb]))


FAMILIES = {
    "f1_extract_compute": f1_extract_compute,
    "f2_visual_reason": f2_visual_reason,
    "f3_retrieve_reason": f3_retrieve_reason,
}


def gold_in(text, gold):
    """Answer-echo check. Compares against the bare number too, so that
    '56.13' is caught inside '$56.13' and inside '156.130'."""
    g = gold.strip().lstrip("$")
    return re.search(re.escape(g), text.replace(",", "")) is not None
