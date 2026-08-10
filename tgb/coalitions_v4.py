"""Candidate coalitions for TGB-v4 (text-only)."""
from .families_v4 import NUISANCE_V4, UNION_TOOLS_V4, WRONG_PARTNER_V4

FULL_MENU_V4 = UNION_TOOLS_V4 + NUISANCE_V4
LABELS = {"none": "none", "useful": "useful", "no_downstream": "partial",
          "no_upstream": "partial", "partial_1of2": "partial",
          "wrong": "wrong", "corrupt": "corrupt", "union": "union",
          "full": "full"}


def candidates_v4(task):
    gt = list(task["gt_tools"])
    up = list(task["upstream"])
    down = [t for t in gt if t not in up]
    out = [
        dict(name="none", tools=[], corrupt=[]),
        dict(name="useful", tools=gt, corrupt=[]),
        dict(name="no_downstream", tools=up, corrupt=[]),
        dict(name="no_upstream", tools=down, corrupt=[]),
        dict(name="wrong", tools=WRONG_PARTNER_V4[task["family"]], corrupt=[]),
        dict(name="corrupt", tools=gt,
             corrupt=[task["corrupt_tool"]] if task["corrupt_tool"] else []),
        dict(name="union", tools=UNION_TOOLS_V4, corrupt=[]),
        dict(name="full", tools=FULL_MENU_V4, corrupt=[]),
    ]
    if len(up) > 1:
        out.insert(3, dict(name="partial_1of2", tools=up[1:] + down, corrupt=[]))
    for c in out:
        c["label"] = LABELS[c["name"]]
    return out
