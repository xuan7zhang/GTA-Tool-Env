"""Candidate coalitions for TGB-v3.

Same shape as v2. Two families need special handling and the handling is the
point of the family:

  * `no_tool` has an empty ground-truth chain, so `useful` is the empty mask and
    coincides with `none`. The benchmark's first task where calling nothing is
    correct.
  * `compute_only` has a single-tool chain with no upstream, so `no_downstream`
    is empty and `no_upstream` is the chain itself.
"""
from .families_v3 import NUISANCE_V3, UNION_TOOLS_V3, WRONG_PARTNER_V3
from .tools import IMAGE_TOOLS

FULL_MENU_V3 = UNION_TOOLS_V3 + NUISANCE_V3 + IMAGE_TOOLS

LABELS = {"none": "none", "useful": "useful", "no_downstream": "partial",
          "no_upstream": "partial", "wrong": "wrong", "corrupt": "corrupt",
          "union": "union", "full": "full"}


def candidates_v3(task):
    gt = list(task["gt_tools"])
    up = list(task["upstream"])
    down = [t for t in gt if t not in up]
    out = [
        dict(name="none", tools=[], corrupt=[]),
        dict(name="useful", tools=gt, corrupt=[]),
        dict(name="no_downstream", tools=up, corrupt=[]),
        dict(name="no_upstream", tools=down, corrupt=[]),
        dict(name="wrong", tools=WRONG_PARTNER_V3[task["family"]], corrupt=[]),
        dict(name="corrupt", tools=gt,
             corrupt=[task["corrupt_tool"]] if task["corrupt_tool"] else []),
        dict(name="union", tools=UNION_TOOLS_V3, corrupt=[]),
        dict(name="full", tools=FULL_MENU_V3, corrupt=[]),
    ]
    for c in out:
        c["label"] = LABELS[c["name"]]
    return out
