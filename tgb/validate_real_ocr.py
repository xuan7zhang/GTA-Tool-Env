"""Is the deterministic OCR simulator a faithful stand-in for a real OCR tool?

Runs the *real* engine (EasyOCR, the same one AgentLego serves to GTA) over the
rendered PNGs, formats its output exactly as the GTA OCR tool does, pushes it
through the *same* downstream parser the chain uses, and asks whether the
Calculator ends up with the same number.

    /datasets/omni_pretraining/gta2/envs/agentlego/bin/python \
        -m tgb.validate_real_ocr --n 200

This is the honest check on the "deterministic simulator" shortcut: a task
where real OCR does not reproduce the simulated intermediate is a task whose
`useful` condition would not survive a real perception tool, and the reported
rate is the size of that gap.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tgb.tools import _parse                                # noqa: E402


def gta_format(result):
    """EasyOCR detail output -> the GTA OCR tool's line format."""
    lines = []
    for box, text, _conf in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        lines.append(f"({int(min(xs))}, {int(min(ys))}, {int(max(xs))}, "
                     f"{int(max(ys))}) {text}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/datasets/omni_pretraining/gta2/results/taco/tgb")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()
    import easyocr

    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    step = max(1, len(tasks) // a.n)
    sub = tasks[::step][:a.n]
    reader = easyocr.Reader(["en"], gpu=a.gpu, verbose=False)

    stat = {}
    rows = []
    for t in sub:
        fam = t["family"]
        s = stat.setdefault(fam, dict(n=0, text_ok=0, value_ok=0))
        s["n"] += 1
        real = gta_format(reader.readtext(os.path.join(a.dir, t["image"])))
        sim = t["conditions"]["useful"]["outputs"].get("OCR", "")
        # (a) do the load-bearing FIELDS survive? Whole-line equality would be
        # the wrong test: a real engine emits one detection per word group, so
        # its line breaks never match the renderer's. What has to survive is
        # the quantities, prices and the tax rate.
        import re as _re
        want = _re.findall(r"QTY\s*\d+|USD\s*\d+\.\d{2}|\d+%",
                           _re.sub(r"^\([^)]*\)\s*", "", sim, flags=_re.M))
        flat = real.replace(" ", "")
        got = sum(1 for w in want if w.replace(" ", "") in flat)
        s["text_ok"] += (got == len(want)) if want else 1
        # (b) does the chain end up at the same number?
        plan = {p["tool"]: p for p in t["plan"]}
        ok = True
        for sym, how in plan["OCR"]["emits"].items():
            if _parse(real, how) != _parse(sim, how):
                ok = False
        s["value_ok"] += ok
        rows.append(dict(id=t["id"], family=fam, text_exact=int(got == len(want)),
                         value_match=int(ok)))

    json.dump(rows, open(os.path.join(a.dir, "real_ocr_validation.json"), "w"),
              indent=1)
    print(f"{'family':<22}{'n':>5}{'fields recovered':>18}"
          f"{'chain value match':>20}")
    for f, s in sorted(stat.items()):
        print(f"{f:<22}{s['n']:>5}{s['text_ok'] / s['n']:>17.1%}"
              f"{s['value_ok'] / s['n']:>19.1%}")
    tot = sum(s["n"] for s in stat.values())
    vm = sum(s["value_ok"] for s in stat.values())
    print(f"{'ALL':<22}{tot:>5}{'':>18}{vm / tot:>19.1%}")


if __name__ == "__main__":
    main()
