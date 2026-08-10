"""TGB raw inputs: the *scene* is the only ground truth a tool may look at.

A scene is a plain dict describing a physical situation (a receipt lying on a
table, a shelf with objects on it, an almanac of facts). Nothing in a scene
mentions the gold answer -- the gold is computed by the task family from the
scene *plus* a number that lives in the question, and is therefore not
recoverable from any single tool output by construction (see families.py).

Scenes with a visual modality are rendered to a real PNG, and the renderer
records the exact bounding box of every glyph run it draws. The simulated OCR
in tools.py reads those recorded boxes -- it never reads the scene's semantic
fields -- so a "perfect OCR" output is exactly what a perfect OCR engine would
return on that image, and the same PNG can be fed to the real AgentLego OCR
(tools.py --backend real) for validation.
"""
import os

from PIL import Image, ImageDraw, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/liberation"
_FONT_CACHE = {}


def font(size, mono=True, bold=False):
    key = (size, mono, bold)
    if key not in _FONT_CACHE:
        name = ("LiberationMono" if mono else "LiberationSans") + (
            "-Bold" if bold else "-Regular") + ".ttf"
        _FONT_CACHE[key] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    return _FONT_CACHE[key]


# ---------------------------------------------------------------- text scenes

def receipt_scene(rng, items, tax_rate, layout="receipt"):
    """items: [(name, qty, unit_price)]. Returns a scene dict."""
    return dict(kind="receipt", layout=layout, items=[
        dict(name=n, qty=q, unit=u) for n, q, u in items], tax_rate=tax_rate,
        store=rng.choice(["Northgate Market", "Vale Hardware", "Copperline Cafe",
                          "Ridgeway Office", "Fenwick Grocers"]))


def shelf_scene(rng, target, n_target, distractors, unit_price):
    """A shelf holding n_target copies of `target` plus distractor objects,
    with a price tag giving the per-unit price of the target."""
    return dict(kind="shelf", target=target, n_target=n_target,
                distractors=[dict(name=n, count=c) for n, c in distractors],
                unit_price=unit_price,
                shop=rng.choice(["Aisle 4", "Aisle 7", "Bay 12", "Rack B"]))


def almanac_scene(entities):
    """entities: [(name, category, attr_name, value, blurb)] -- a fixed corpus
    the retrieval tool searches over. Entity names are nonce strings so no
    model can know the values parametrically."""
    return dict(kind="almanac", entities=[
        dict(name=n, category=c, attr=a, value=v, blurb=b)
        for n, c, a, v, b in entities])


# ------------------------------------------------------------------ rendering

def _draw_lines(img, draw, lines, x0, y0, dy, f, boxes, tag=None):
    y = y0
    for text in lines:
        draw.text((x0, y), text, fill=(20, 20, 20), font=f)
        l, t, r, b = draw.textbbox((x0, y), text, font=f)
        boxes.append(dict(text=text, box=[int(l), int(t), int(r), int(b)],
                          tag=tag))
        y += dy
    return y


# Field syntax and type size are not cosmetic. They were chosen by probing a
# real OCR engine (tgb/probe_render.py): at 19px with "3 @ $30.47" EasyOCR
# dropped the quantity outright and read "$15.04" as "S15" "04", which would
# have made the executable-tools claim false. "QTY 3" / "USD 30.47" at 32px
# bold comes back intact, token by token.
FSIZE = 32
LINE_H = 52


def render_receipt(scene, path):
    """Draw the receipt. Returns the OCR ground truth: ordered [{text, box}].

    The rendered lines carry the item names, quantities and unit prices and the
    tax rate -- everything needed to *derive* the total -- but never the total
    itself, so no perception tool can echo the intermediate, let alone the gold.
    """
    W = 760
    H = 150 + LINE_H * (len(scene["items"]) + 2)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    boxes = []
    fh, fb = font(FSIZE, mono=False, bold=True), font(FSIZE, mono=False, bold=True)
    d.text((30, 28), scene["store"], fill=(0, 0, 0), font=fh)
    l, t, r, b = d.textbbox((30, 28), scene["store"], font=fh)
    boxes.append(dict(text=scene["store"], box=[int(l), int(t), int(r), int(b)],
                      tag="store"))
    y = 28 + LINE_H + 12
    if scene["layout"] == "table":
        y = _draw_lines(img, d, ["ITEM   QTY   UNIT PRICE"], 30, y, LINE_H, fb,
                        boxes, "header")
    for it in scene["items"]:
        line = f"{it['name']}   QTY {it['qty']}   USD {it['unit']:.2f}"
        y = _draw_lines(img, d, [line], 30, y, LINE_H, fb, boxes, "item")
    d.line([(30, y + 8), (W - 30, y + 8)], fill=(90, 90, 90), width=2)
    _draw_lines(img, d, [f"TAX RATE {scene['tax_rate']}%"], 30, y + 22, LINE_H,
                fb, boxes, "tax")
    img.save(path)
    return boxes


_SHAPES = {
    "cup": "ellipse", "box": "rect", "bottle": "tall", "plate": "ellipse",
    "book": "rect", "can": "tall", "mug": "ellipse", "crate": "rect",
    "jar": "tall", "tin": "rect", "bowl": "ellipse", "carton": "rect",
}


def render_shelf(scene, path):
    """Draw n_target target objects plus distractors, and a price tag.

    The count is only obtainable by *looking* (counting glyphs); the price is
    only obtainable by *reading* the tag. Neither tool alone suffices.
    """
    W, H = 760, 500
    img = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(img)
    boxes = []
    objs = [(scene["target"], 1)] * scene["n_target"]
    for dd in scene["distractors"]:
        objs += [(dd["name"], 0)] * dd["count"]
    # deterministic placement on an 8x4 grid
    slots = [(46 + 86 * (i % 8), 90 + 88 * (i // 8)) for i in range(32)]
    for (name, is_tgt), (x, y) in zip(objs, slots):
        shape = _SHAPES.get(name, "rect")
        col = (70, 110, 170) if is_tgt else (150, 150, 145)
        if shape == "ellipse":
            d.ellipse([x, y, x + 52, y + 52], fill=col)
        elif shape == "tall":
            d.rectangle([x + 14, y - 8, x + 38, y + 58], fill=col)
        else:
            d.rectangle([x, y + 8, x + 54, y + 50], fill=col)
    tag = f"PRICE PER {scene['target'].upper()}   USD {scene['unit_price']:.2f}"
    fb = font(FSIZE, mono=False, bold=True)
    d.rectangle([26, H - 78, 26 + d.textlength(tag, font=fb) + 28, H - 20],
                fill=(255, 255, 255), outline=(60, 60, 60), width=2)
    _draw_lines(img, d, [tag], 40, H - 68, LINE_H, fb, boxes, "tag")
    _draw_lines(img, d, [scene["shop"]], 40, 24, LINE_H,
                font(FSIZE, mono=False, bold=True), boxes, "shop")
    img.save(path)
    return boxes


def render_form(scene, path):
    """An order form: product codes and quantities, and deliberately NO prices.

    The prices for those codes exist only in the retrieval corpus, so the form
    and the corpus are each necessary and neither is sufficient -- this is the
    3-tool coalition (OCR + GoogleSearch + Calculator) that the real GTA
    read-compute subset also contains.
    """
    W = 800
    H = 190 + LINE_H * len(scene["order"])
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    boxes = []
    fb = font(FSIZE, mono=False, bold=True)
    _draw_lines(img, d, [f"PURCHASE ORDER  {scene['ref']}"], 30, 28, LINE_H,
                fb, boxes, "header")
    y = 28 + LINE_H + 14
    for o in scene["order"]:
        y = _draw_lines(img, d, [f"{o['code']}   QTY {o['qty']}"], 30, y,
                        LINE_H, fb, boxes, "line")
    d.line([(30, y + 8), (W - 30, y + 8)], fill=(90, 90, 90), width=2)
    _draw_lines(img, d, ["authorised signature"], 30, y + 22, LINE_H, fb,
                boxes, "foot")
    img.save(path)
    return boxes


def render_panel(scene, path):
    """A generic labelled-value panel: a title line plus 'LABEL VALUE' rows.

    Same typography as the receipt (32px bold, marker-prefixed values), which
    is the configuration validated against a real OCR engine, so every v2
    family inherits that fidelity instead of re-earning it.
    """
    W = 820
    H = 150 + LINE_H * (len(scene["rows"]) + 1)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    boxes = []
    fb = font(FSIZE, mono=False, bold=True)
    _draw_lines(img, d, [scene["title"]], 30, 28, LINE_H, fb, boxes, "title")
    y = 28 + LINE_H + 12
    for r in scene["rows"]:
        y = _draw_lines(img, d, [r], 30, y, LINE_H, fb, boxes, "row")
    d.line([(30, y + 8), (W - 30, y + 8)], fill=(90, 90, 90), width=2)
    img.save(path)
    return boxes


def render_shapes(scene, path):
    """Coloured shapes of differing sizes and no text at all.

    This is the non-OCR perception surface: the answer depends on measuring a
    rendered object, so an OCR call returns nothing and the task cannot be
    solved by reading.
    """
    W, H = 760, 460
    img = Image.new("RGB", (W, H), (250, 250, 248))
    d = ImageDraw.Draw(img)
    slots = [(60 + 170 * (i % 4), 60 + 190 * (i // 4)) for i in range(8)]
    for sh, (x, y) in zip(scene["shapes"], slots):
        r = int(sh["size"] * 1.4)
        col = {"red": (190, 70, 70), "blue": (70, 110, 180),
               "green": (80, 150, 90), "yellow": (200, 175, 60),
               "purple": (140, 90, 170)}.get(sh["color"], (120, 120, 120))
        if sh["name"] == "circle":
            d.ellipse([x, y, x + r, y + r], fill=col)
        elif sh["name"] == "triangle":
            d.polygon([(x + r // 2, y), (x, y + r), (x + r, y + r)], fill=col)
        else:
            d.rectangle([x, y, x + r, y + r], fill=col)
    img.save(path)
    return []          # no glyphs: OCR has nothing to find, by design


def render_objects(scene, path):
    """Countable objects, no price tag and no text -- the counting channel
    without an OCR crutch."""
    W, H = 760, 420
    img = Image.new("RGB", (W, H), (245, 245, 242))
    d = ImageDraw.Draw(img)
    objs = [(scene["target"], 1)] * scene["n_target"]
    for dd in scene["distractors"]:
        objs += [(dd["name"], 0)] * dd["count"]
    slots = [(46 + 86 * (i % 8), 70 + 88 * (i // 8)) for i in range(32)]
    for (name, is_tgt), (x, y) in zip(objs, slots):
        col = (70, 110, 170) if is_tgt else (150, 150, 145)
        shape = _SHAPES.get(name, "rect")
        if shape == "ellipse":
            d.ellipse([x, y, x + 52, y + 52], fill=col)
        elif shape == "tall":
            d.rectangle([x + 14, y - 8, x + 38, y + 58], fill=col)
        else:
            d.rectangle([x, y + 8, x + 54, y + 50], fill=col)
    img.save(path)
    return []


def render(scene, path):
    if scene["kind"] == "shapes":
        return render_shapes(scene, path)
    if scene["kind"] == "objects":
        return render_objects(scene, path)
    if scene["kind"] == "textless":
        Image.new("RGB", (64, 64), (255, 255, 255)).save(path)
        return []
    if scene["kind"] == "panel":
        return render_panel(scene, path)
    if scene["kind"] == "receipt":
        return render_receipt(scene, path)
    if scene["kind"] == "shelf":
        return render_shelf(scene, path)
    if scene["kind"] == "form":
        return render_form(scene, path)
    return []
