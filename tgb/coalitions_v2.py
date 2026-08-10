"""Candidate coalitions for TGB-v2.

Same shape as v1 plus the condition that decides the whole question:

    union   the union of every family's ground-truth chain

`union` is the best a *global* policy can do without abstaining -- it is what a
greedy environment search converges to when every chain must stay runnable. In
v1 it was near-optimal because the surplus tools were inert. In v2 the surplus
tools fire blind and emit competing numbers, so `union` should be a genuinely
bad policy and per-task masking should have something to win. Whether the
likelihood signal actually wins it is the experiment, not an assumption.

`wrong` is another family's chain rather than a contentless describer, which is
a sharper control now that off-family tools produce confident output.
"""
from .families_v2 import NUISANCE, UNION_TOOLS, WRONG_PARTNER
from .tools import IMAGE_TOOLS

FULL_MENU_V2 = UNION_TOOLS + NUISANCE + IMAGE_TOOLS

LABELS = {"none": "none", "useful": "useful", "no_downstream": "partial",
          "no_upstream": "partial", "partial_1of2": "partial",
          "wrong": "wrong", "corrupt": "corrupt", "union": "union",
          "full": "full"}


def candidates_v2(task):
    gt = list(task["gt_tools"])
    up = list(task["upstream"])
    down = [t for t in gt if t not in up]
    out = [
        dict(name="none", tools=[], corrupt=[]),
        dict(name="useful", tools=gt, corrupt=[]),
        dict(name="no_downstream", tools=up, corrupt=[]),
        dict(name="no_upstream", tools=down, corrupt=[]),
        dict(name="wrong", tools=WRONG_PARTNER[task["family"]], corrupt=[]),
        dict(name="corrupt", tools=gt, corrupt=[task["corrupt_tool"]]),
        dict(name="union", tools=UNION_TOOLS, corrupt=[]),
        dict(name="full", tools=FULL_MENU_V2, corrupt=[]),
    ]
    if len(up) > 1:
        out.insert(3, dict(name="partial_1of2", tools=up[1:] + down,
                           corrupt=[]))
    for c in out:
        c["label"] = LABELS[c["name"]]
    return out
