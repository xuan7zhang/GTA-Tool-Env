"""Invariant tests for the TGB generator.

These are not smoke tests -- each one guards a property the benchmark's claim
depends on. If any fails, the corresponding condition in the results table
stops meaning what it says.

    python -m tgb.tests.test_tgb          # or: pytest tgb/tests/test_tgb.py
"""
import os
import random
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from tgb import families, generate, scenes, tools          # noqa: E402
from tgb.coalitions import candidates                      # noqa: E402

_TMP = tempfile.mkdtemp(prefix="tgb_test_")
TASKS, REJECT = generate.build(8, _TMP, seed=4242)


def _ctxs(t):
    return {n: c["context"] for n, c in t["conditions"].items()}


def test_all_families_present():
    fams = {t["family"] for t in TASKS}
    assert fams == set(families.FAMILIES), fams
    assert len(TASKS) == 8 * 3, len(TASKS)


def test_non_echo_every_condition():
    """(1) No coalition's context may contain the gold. This is the condition
    the whole benchmark exists to keep, so it is checked on every context of
    every task, not on a sample."""
    for t in TASKS:
        for name, ctx in _ctxs(t).items():
            assert not (ctx and families.gold_in(ctx, t["gold"])), \
                (t["id"], name, t["gold"], ctx[:200])


def test_gold_not_in_question():
    for t in TASKS:
        assert not families.gold_in(t["question"], t["gold"]), t["id"]


def test_useful_coalition_is_sufficient():
    """(2) derivable-by-construction: replaying the one easy model-side step on
    the Calculator's real output must reproduce the gold exactly."""
    for t in TASKS:
        calc = t["conditions"]["useful"]["outputs"]["Calculator"]
        assert generate.final_step(t, calc) == t["gold"], (t["id"], calc)


def test_intermediate_is_what_the_chain_computed():
    for t in TASKS:
        assert t["conditions"]["useful"]["outputs"]["Calculator"] == \
            t["meta"]["intermediate"], t["id"]


def test_orphaned_downstream_errors():
    """(3) coalition-level: the downstream tool has no marginal without its
    upstream. Not asserted in prose -- the executed Calculator must fail."""
    for t in TASKS:
        out = t["conditions"]["no_upstream"]["outputs"]["Calculator"]
        assert "Error" in out or "NameError" in out, (t["id"], out)
        assert generate.final_step(t, out) is None


def test_partial_coalition_loses_the_intermediate():
    for t in TASKS:
        for name in ("no_downstream", "partial_1of2"):
            c = t["conditions"].get(name)
            if not c:
                continue
            assert t["meta"]["intermediate"] not in c["context"], (t["id"], name)


def test_corruption_changes_the_number():
    """(5) output quality is a separate axis: same tools, different result."""
    for t in TASKS:
        a = t["conditions"]["useful"]["outputs"]["Calculator"]
        b = t["conditions"]["corrupt"]["outputs"]["Calculator"]
        assert a != b, (t["id"], a, b)


def test_corruption_stays_parseable():
    """A corruption that broke parsing would collapse `corrupt` into
    `no_upstream` -- both would end in an error and the two failure modes
    could not be told apart."""
    for t in TASKS:
        b = t["conditions"]["corrupt"]["outputs"]["Calculator"]
        assert re.match(r"^-?\d+(\.\d+)?$", b), (t["id"], b)


def test_chain_actually_wired_not_pasted():
    """The Calculator's expression must be built out of the upstream output,
    so its literals have to appear in the upstream text."""
    for t in TASKS:
        tr = {s["tool"]: s for s in t["conditions"]["useful"]["trace"]}
        expr = tr["Calculator"]["args"]["expression"]
        assert not re.search(r"[A-Za-z_]", expr), (t["id"], expr)
        ups = " ".join(tr[u]["output"] for u in t["upstream"])
        # drop a trailing tax factor: it is derived from the "8%" on the
        # receipt, not copied from it
        core = re.sub(r"\)\s*\*\s*[\d.]+$", ")", expr)
        lits = re.findall(r"\d+\.\d{2}(?!\d)", core)
        assert lits, (t["id"], expr)
        for v in lits:
            assert v in ups, (t["id"], v, expr)


def test_wrong_coalition_is_contentless_not_noisy():
    """`wrong` must be on-topic prose with no usable figures -- otherwise it is
    a corruption control, not a tool-type control."""
    for t in TASKS:
        d = t["conditions"]["wrong"]["outputs"]["ImageDescription"]
        assert len(d) > 80, t["id"]
        assert not re.search(r"(\$|USD)\s*\d", d), (t["id"], d)
        assert not re.search(r"QTY\s*\d", d), (t["id"], d)
        assert t["meta"]["intermediate"] not in d, t["id"]


def test_full_menu_is_a_superset():
    for t in TASKS:
        assert set(t["gt_tools"]) <= set(t["conditions"]["full"]["tools"])
        assert len(t["conditions"]["full"]["tools"]) >= 12


def test_baseline_has_no_context():
    for t in TASKS:
        assert t["conditions"]["none"]["context"] == "", t["id"]


def test_images_exist_and_are_real_pngs():
    for t in TASKS:
        p = os.path.join(_TMP, t["image"])
        assert os.path.exists(p) and os.path.getsize(p) > 500, p
        with open(p, "rb") as f:
            assert f.read(4) == b"\x89PNG", p


def test_ocr_output_matches_the_rendered_pixels():
    """The simulated OCR must be a function of the render, not of the scene's
    semantics -- that is what makes the same PNG usable with a real OCR."""
    t = TASKS[0]
    boxes = scenes.render(t["scene"], os.path.join(_TMP, "_probe.png"))
    out = tools.ocr(t["scene"], boxes, random.Random(0))
    for b in boxes:
        assert b["text"] in out
    assert len(out.strip().split("\n")) == len(boxes)


def test_determinism():
    a, _ = generate.build(3, tempfile.mkdtemp(prefix="tgb_a_"), seed=999)
    b, _ = generate.build(3, tempfile.mkdtemp(prefix="tgb_b_"), seed=999)
    ka = [(t["id"], t["gold"], t["question"]) for t in a]
    kb = [(t["id"], t["gold"], t["question"]) for t in b]
    assert ka == kb
    assert [_ctxs(t) for t in a] == [_ctxs(t) for t in b]


def test_garble_always_changes_something():
    r = random.Random(1)
    for s in ["$12.40", "TAX RATE 8%", "Milk 2 @ $3.40", "PO-40757"]:
        assert tools.garble(s, r) != s, s


def test_candidate_menu_shape():
    for t in TASKS:
        names = [c["name"] for c in candidates(t)]
        assert names[0] == "none" and "useful" in names
        assert len(names) == (8 if len(t["upstream"]) > 1 else 7)


def _run():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in fns:
        try:
            f()
            print(f"  PASS  {n}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {n}: {str(e)[:200]}")
    print(f"\n{len(fns) - bad}/{len(fns)} passed   "
          f"(generator rejects: {REJECT or 'none'})")
    shutil.rmtree(_TMP, ignore_errors=True)
    return bad


if __name__ == "__main__":
    sys.exit(_run())
