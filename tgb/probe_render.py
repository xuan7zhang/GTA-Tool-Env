"""Pick a receipt/tag rendering that a REAL OCR engine can actually read.

The first TGB render was tuned for the simulator and EasyOCR lost load-bearing
fields on it (the quantity "3 @" vanished, "$15.04" came back as "S15" "04").
That would have made the "tools really execute" claim false in the one place it
matters. This probe renders the same content several ways and reports, per
variant, whether the real engine recovers every quantity and price.
"""
import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgb.scenes import font                                  # noqa: E402
from tgb.validate_real_ocr import gta_format                 # noqa: E402

ITEMS = [("Brush", 3, 30.47), ("Sander", 3, 27.46), ("Bolt", 3, 8.32),
         ("Nails", 2, 9.04)]
TAX = 8
WANT = ["3", "30.47", "3", "27.46", "3", "8.32", "2", "9.04", "8"]


def variant(name, size, bold, fmt, pad, dy):
    W = 720
    H = 2 * pad + dy * (len(ITEMS) + 2)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = font(size, mono=False, bold=bold)
    y = pad
    for nm, q, p in ITEMS:
        d.text((pad, y), fmt.format(nm=nm, q=q, p=p), fill=(0, 0, 0), font=f)
        y += dy
    d.text((pad, y + dy // 2), f"TAX RATE {TAX}%", fill=(0, 0, 0), font=f)
    path = f"/tmp/tgb_probe_{name}.png"
    img.save(path)
    return path


VARIANTS = [
    ("A_at_small", 22, False, "{nm}  {q} @ ${p:.2f}", 24, 34),
    ("B_x_bold", 30, True, "{nm}   x{q}   ${p:.2f}", 30, 48),
    ("C_qty_words", 32, True, "{nm}   QTY {q}   USD {p:.2f}", 32, 52),
    ("D_cols", 30, True, "{nm}      {q}      {p:.2f}", 30, 50),
]


def main():
    import easyocr
    r = easyocr.Reader(["en"], gpu=False, verbose=False)
    for name, size, bold, fmt, pad, dy in VARIANTS:
        p = variant(name, size, bold, fmt, pad, dy)
        out = gta_format(r.readtext(p))
        flat = out.replace(" ", "")
        got = sum(1 for w in WANT if w.replace(" ", "") in flat)
        print(f"\n=== {name}: recovered {got}/{len(WANT)} load-bearing fields")
        print(out)


if __name__ == "__main__":
    main()
