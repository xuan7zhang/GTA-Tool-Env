"""Invariant tests for TGB-v4, the text-only benchmark.

v3 and v4 were built without a test file -- the generator asserted, but nothing
guarded the properties from outside. Two defects reached a scored dataset as a
result, and both are now tests here:

  * a downstream tool that is not load-bearing (v3's `table_lookup` returned a
    single row, so the upstream alone beat the full chain);
  * a question whose phrasing lets the model report the chain's *intermediate*
    verbatim ("a coupon applies, what is payable?" -- 86% echo, 7B accuracy
    0.163, 14B 0.015).

    python -m tgb.tests.test_tgb_v4
"""
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from tgb import families_v4 as F4                              # noqa: E402
from tgb import generate_v4, tools_v2, tools_v4                # noqa: E402
from tgb.coalitions_v4 import FULL_MENU_V4, candidates_v4      # noqa: E402

tools_v2.register()
tools_v4.register()
_TMP = tempfile.mkdtemp(prefix="tgb4_test_")
# 40 per family, not 5: the gold-diversity bound is 0.6*n, and at n=5 it
# accepted a family with 96 distinct answers out of 400 -- the very
# collapse it exists to catch.
TASKS, REJECT = generate_v4.build(40, _TMP, seed=555)
NO_CHAIN = {"no_tool"}
NO_UP = {"no_tool", "compute_only"}


def _down(t):
    return [x for x in t["gt_tools"] if x not in t["upstream"]][-1]


def test_all_ten_families():
    assert {t["family"] for t in TASKS} == set(F4.FAMILIES_V4), \
        {t["family"] for t in TASKS}


def test_text_only_no_image_anywhere():
    """The reason v4 exists: v1/v2 put OCR in every chain and v3 still had
    three visual families, so no earlier result separates tool use from
    reading text off a picture."""
    for t in TASKS:
        assert "OCR" not in t["gt_tools"], t["id"]
        assert t["scene"]["kind"] == "textless", (t["id"], t["scene"]["kind"])
        assert t.get("has_ocr", 0) == 0, t["id"]


def test_non_echo_every_condition():
    for t in TASKS:
        for n, c in t["conditions"].items():
            assert not (c["context"] and F4.gold_in(c["context"], t["gold"])), \
                (t["id"], n)


def test_gold_not_in_question():
    for t in TASKS:
        assert not F4.gold_in(t["question"], t["gold"]), t["id"]


def test_useful_is_sufficient():
    for t in TASKS:
        if t["family"] in NO_CHAIN:
            continue
        out = t["conditions"]["useful"]["outputs"][_down(t)]
        assert generate_v4.final_step(t, out) == t["gold"], (t["id"], out)


def test_orphaned_downstream_errors():
    for t in TASKS:
        if t["family"] in NO_UP:
            continue
        out = t["conditions"]["no_upstream"]["outputs"].get(_down(t), "")
        assert "Error" in out or "NameError" in out or "no " in out.lower(), \
            (t["id"], out)


def test_downstream_is_load_bearing():
    """The v3 lesson. An upstream that hands over a single ready-made pair of
    numbers makes the downstream decorative: v3's `table_lookup` scored 0.267
    with the upstream alone against 0.198 for the full chain. Every v4 upstream
    must therefore emit a *set* -- rows, samples, passages -- so the
    aggregation is real work."""
    multi = {"table_total": r"qty \d+", "table_filter": r"qty \d+",
             "sensor_mean": r",", "sensor_convert": r",",
             "schedule_gap": r"\d\d \d\d", "doc_two_facts": r"price of"}
    for t in TASKS:
        pat = multi.get(t["family"])
        if not pat:
            continue
        up_out = t["conditions"]["useful"]["outputs"][t["upstream"][0]]
        assert len(re.findall(pat, up_out)) >= 2, (t["id"], t["family"], up_out)


def test_question_uses_the_comparative_form():
    """Phrasing is load-bearing too, which is not obvious and cost a dataset.
    "A 600.00 coupon applies. What is the amount payable?" let the model report
    the Calculator's subtotal verbatim on 86% of tasks; the comparative form
    the working families already used fixed it. `no_tool` is exempt -- it has
    no intermediate to confuse with."""
    for t in TASKS:
        if t["family"] == "no_tool":
            continue
        q = t["question"].lower()
        assert "by how much" in q or "by how many" in q, (t["id"], t["question"])


def test_corruption_changes_the_number():
    for t in TASKS:
        # NO_UP, not NO_CHAIN: `compute_only` has no upstream to corrupt, so
        # its chain is a constant template and corruption is a no-op by
        # construction rather than by defect
        if t["family"] in NO_UP:
            continue
        d = _down(t)
        assert t["conditions"]["useful"]["outputs"][d] != \
            t["conditions"]["corrupt"]["outputs"][d], t["id"]


def test_controls_are_what_they_claim():
    """`no_tool` must be answerable with nothing and `compute_only` must need
    no perception -- these are the only two families that make Acc(E_none)
    non-zero, without which dA collapses onto Acc."""
    for t in TASKS:
        if t["family"] == "no_tool":
            assert t["gt_tools"] == [] and t["plan"] == [], t["id"]
            assert t["conditions"]["useful"]["context"] == "", t["id"]
        if t["family"] == "compute_only":
            assert t["upstream"] == [], t["id"]
            assert t["gt_tools"] == ["Calculator"], t["id"]


def test_union_and_full_shapes():
    for t in TASKS:
        assert set(t["gt_tools"]) <= set(t["conditions"]["union"]["tools"])
        assert set(t["conditions"]["union"]["tools"]) <= set(FULL_MENU_V4)


def test_candidate_menu_shape():
    for t in TASKS:
        names = [c["name"] for c in candidates_v4(t)]
        for req in ("none", "useful", "union", "full", "corrupt", "wrong"):
            assert req in names, (t["id"], req)


def test_questions_are_unique():
    """Prompt text must identify the task. The first cut repeated a question on
    326 of `table_total`'s 400 tasks -- six batch names and a round-number
    amount were the only things that varied -- so different tables arrived
    under an identical prompt. The contexts differed, which hid it, but the
    input the model conditions on was genuinely ambiguous."""
    qs = [t["question"] for t in TASKS]
    assert len(set(qs)) == len(qs), \
        f"{len(qs) - len(set(qs))} duplicate questions"


def test_gold_is_not_concentrated():
    """A family whose k is a round multiple of its own answer's scale collapses
    the answer space: `permit = int(dur/30)*30` made gold = dur mod 30, so 400
    tasks shared 29 distinct golds and guessing scored 3.4%."""
    import collections
    by = collections.defaultdict(set)
    for t in TASKS:
        by[t["family"]].add(t["gold"])
    for f, g in by.items():
        n = sum(1 for t in TASKS if t["family"] == f)
        assert len(g) >= max(2, int(0.6 * n)), (f, len(g), n)


class _Stub:
    """A tokenizer with the one property that matters here: a leading space
    binds to the word after it, so " 215" and "215" are different tokens."""

    def __init__(self):
        self.vocab = {}

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(t, len(self.vocab))
                              for t in re.findall(r" ?[^\s]+|\s+", s)]}


def test_gold_span_is_tokenized_in_place():
    """The scorer must tokenize the continuation *in context*. Tokenizing the
    gold on its own and concatenating scores the space-less token '215' at a
    position where the model emits ' 215' -- and because a model made confident
    by a useful tool puts more mass on ' 215', dL turns negative exactly where
    the tool helps. schedule_gap read dL = -1.79 at 0.830 accuracy before this.
    """
    from tgb.score_dl import span
    tok = _Stub()
    p = "Question: how long?\n\nFinal answer:"
    full, n, ng = span(tok, p, "215", 4096)
    assert full[n:] == tok(" 215")["input_ids"], "gold span lost its boundary"
    assert full[n:] != tok("215")["input_ids"], "scoring the space-less variant"
    assert ng == len(full) - n and ng > 0


def test_truncation_keeps_the_gold_and_the_prompt_tail():
    """The old code cut the prompt's tail to make room, which eats the
    "Final answer:" the gold has to follow."""
    from tgb.score_dl import span
    tok = _Stub()
    p = "filler " * 200 + "Final answer:"
    full, n, ng = span(tok, p, "13.53", 40)
    assert len(full) <= 40 and ng > 0
    assert full[n:] == tok(" 13.53")["input_ids"]
    assert tok(" Final")["input_ids"][0] in full[:n], "prompt tail was trimmed"


def test_determinism():
    a, _ = generate_v4.build(3, tempfile.mkdtemp(prefix="tgb4_a_"), seed=41)
    b, _ = generate_v4.build(3, tempfile.mkdtemp(prefix="tgb4_b_"), seed=41)
    ka = [(t["id"], t["gold"], {n: c["context"]
                                for n, c in t["conditions"].items()}) for t in a]
    kb = [(t["id"], t["gold"], {n: c["context"]
                                for n, c in t["conditions"].items()}) for t in b]
    assert ka == kb


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
