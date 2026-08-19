"""Invariant tests for TGB-v2.

v1's tests carry over in spirit; the ones that matter here are the new
structural claims that make v2 a different experiment from v1:

  * every off-family compute tool is *active*, not inert -- it emits a number
    when fired blind (this is the property whose absence made v1's union mask
    near-optimal);
  * the union context therefore carries numbers that compete with the chain's
    intermediate;
  * the chain still errors when its upstream is masked out, so a downstream
    tool has no marginal alone.

    python -m tgb.tests.test_tgb_v2
"""
import os
import random
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from tgb import families_v2 as F2                            # noqa: E402
from tgb import generate_v2, scenes, tools_v2                # noqa: E402
from tgb.coalitions_v2 import FULL_MENU_V2, candidates_v2    # noqa: E402
from tgb.tools import TOOLS, run_chain                       # noqa: E402

tools_v2.register()
_TMP = tempfile.mkdtemp(prefix="tgb2_test_")
TASKS, REJECT = generate_v2.build(4, _TMP, seed=777)
NUM = re.compile(r"\d+\.\d{2}")


def _down(t):
    return [x for x in t["gt_tools"] if x not in t["upstream"]][-1]


def test_all_eight_families():
    assert {t["family"] for t in TASKS} == set(F2.FAMILIES_V2), \
        {t["family"] for t in TASKS}
    assert len(TASKS) == 4 * 8


def test_non_echo_every_condition():
    for t in TASKS:
        for n, c in t["conditions"].items():
            assert not (c["context"] and F2.gold_in(c["context"], t["gold"])), \
                (t["id"], n)


def test_gold_not_in_question():
    for t in TASKS:
        assert not F2.gold_in(t["question"], t["gold"]), t["id"]


def test_useful_is_sufficient():
    for t in TASKS:
        out = t["conditions"]["useful"]["outputs"][_down(t)]
        assert generate_v2.final_step(t, out) == t["gold"], (t["id"], out)


def test_orphaned_downstream_errors():
    for t in TASKS:
        out = t["conditions"]["no_upstream"]["outputs"].get(_down(t), "")
        assert "Error" in out or "NameError" in out, (t["id"], out)
        assert generate_v2.final_step(t, out) is None


def test_corruption_changes_the_number():
    for t in TASKS:
        a = t["conditions"]["useful"]["outputs"][_down(t)]
        b = t["conditions"]["corrupt"]["outputs"][_down(t)]
        assert a != b, (t["id"], a, b)


def test_offfamily_tools_are_active_not_inert():
    """The v2 claim. Fired blind on a task that does not need it, every
    compute tool must still emit a number -- that is what makes the union a
    compromise instead of a free lunch."""
    active = {}
    for t in TASKS:
        u = t["conditions"]["union"]["outputs"]
        for tool in F2.CHAIN_TOOLS:
            # Calculator is exempt on purpose: it is a pure expression
            # evaluator, so a blind call with no expression *should* error.
            # Giving it a fallback would also mean editing the shared v1
            # executor and silently changing already-generated v1 data.
            if tool in t["gt_tools"] or tool not in u or tool == "Calculator":
                continue
            # "active" means it reports a figure, not that the figure has two
            # decimals -- DurationCalc reports whole minutes.
            active.setdefault(tool, []).append(
                bool(re.search(r"\d", str(u[tool]))) and
                "no " not in str(u[tool]).lower() and
                "could not" not in str(u[tool]).lower())
    assert active, "no off-family compute tool was exercised"
    for tool, hits in active.items():
        assert any(hits), f"{tool} is inert on every off-family task"


def test_union_context_carries_competing_numbers():
    for t in TASKS:
        assert t["contamination"], t["id"]
        assert t["meta"]["intermediate"] not in t["contamination"], t["id"]


def test_wrong_is_another_familys_chain():
    for t in TASKS:
        w = set(t["conditions"]["wrong"]["tools"])
        assert w != set(t["gt_tools"]), t["id"]
        assert w & set(F2.CHAIN_TOOLS), t["id"]


def test_union_and_full_shapes():
    for t in TASKS:
        assert set(t["gt_tools"]) <= set(t["conditions"]["union"]["tools"])
        assert set(t["conditions"]["union"]["tools"]) <= set(FULL_MENU_V2)
        assert len(t["conditions"]["full"]["tools"]) >= 20


def test_each_hard_tool_owned_by_one_family():
    owner = {}
    for t in TASKS:
        for tool in t["gt_tools"]:
            if tool in F2.CHAIN_TOOLS and tool != "Calculator":
                owner.setdefault(tool, set()).add(t["family"])
    for tool, fams in owner.items():
        assert len(fams) == 1, (tool, fams)


def test_blind_call_on_another_family_is_misleading():
    """A tool fired blind on a task that does not own it must produce a number
    that is not that task's intermediate -- otherwise the contamination is a
    no-op. (Fired on its *own* family a blind call may legitimately coincide
    with the chain call, since the panel holds exactly the figure it wants.)"""
    seen = 0
    for t in TASKS:
        boxes = scenes.render(t["scene"], os.path.join(_TMP, "_p.png"))
        for tool in F2.CHAIN_TOOLS:
            if tool in t["gt_tools"] or tool == "Calculator":
                continue
            out = str(TOOLS[tool](t["scene"], boxes, random.Random(0)))
            if not NUM.search(out) and "ELAPSED" not in out:
                continue
            assert t["meta"]["intermediate"] not in out, (t["id"], tool, out)
            seen += 1
    assert seen >= 20, seen


def test_determinism():
    a, _ = generate_v2.build(3, tempfile.mkdtemp(prefix="tgb2_a_"), seed=31)
    b, _ = generate_v2.build(3, tempfile.mkdtemp(prefix="tgb2_b_"), seed=31)
    ka = [(t["id"], t["gold"], {n: c["context"]
                                for n, c in t["conditions"].items()}) for t in a]
    kb = [(t["id"], t["gold"], {n: c["context"]
                                for n, c in t["conditions"].items()}) for t in b]
    assert ka == kb


def test_images_are_real_pngs():
    for t in TASKS:
        p = os.path.join(_TMP, t["image"])
        assert os.path.exists(p) and os.path.getsize(p) > 500
        with open(p, "rb") as f:
            assert f.read(4) == b"\x89PNG"


def test_candidate_menu_shape():
    for t in TASKS:
        names = [c["name"] for c in candidates_v2(t)]
        for req in ("none", "useful", "union", "full", "corrupt", "wrong"):
            assert req in names, (t["id"], req)


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
