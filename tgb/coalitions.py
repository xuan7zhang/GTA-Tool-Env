"""Candidate tool coalitions scored for every TGB task.

The point of the benchmark is that a task does not come with one evidence
string, it comes with a *menu* and several ways to spend it. Each condition
below is a different subset of that menu, executed for real on the same scene,
so any difference in gold-likelihood is attributable to the tool set alone.

    none          no tools                      (baseline for dL)
    useful        the ground-truth chain
    partial_1of2  one upstream tool dropped     (coalition incomplete)
    no_downstream upstream only, no Calculator  (raw evidence, hard final step)
    no_upstream   Calculator only               (orphaned downstream -> error)
    wrong         ImageDescription + Calculator (right modality, no content)
    corrupt       the GT chain, upstream noisy  (right tools, bad output)
    full          the whole 12-tool menu        (the all-tools environment)

`useful` is the only condition that is both complete and clean, so the
pre-registered prediction is dL(useful) > every other condition, with
no_upstream and wrong at the bottom.
"""
from .tools import IMAGE_TOOLS

FULL_MENU = ["OCR", "ImageDescription", "CountGivenObject", "TextToBbox",
             "GoogleSearch", "Calculator", "Solver"] + IMAGE_TOOLS

LABELS = {"none": "none", "useful": "useful", "partial_1of2": "partial",
          "no_downstream": "partial", "no_upstream": "partial",
          "wrong": "wrong", "corrupt": "corrupt", "full": "full"}


def candidates(task):
    """Return [{name, tools, corrupt, label}] for one task."""
    gt = list(task["gt_tools"])
    up = list(task["upstream"])
    down = [t for t in gt if t not in up]
    out = [
        dict(name="none", tools=[], corrupt=[]),
        dict(name="useful", tools=gt, corrupt=[]),
        dict(name="no_downstream", tools=up, corrupt=[]),
        dict(name="no_upstream", tools=down, corrupt=[]),
        dict(name="wrong", tools=["ImageDescription"] + down, corrupt=[]),
        dict(name="corrupt", tools=gt, corrupt=[task["corrupt_tool"]]),
        dict(name="full", tools=FULL_MENU, corrupt=[]),
    ]
    if len(up) > 1:
        # drop the *first* upstream tool, keep the rest of the chain: the
        # sharpest test of coalition completeness, since the surviving tools
        # are all correct and clean and the context still looks informative.
        out.insert(3, dict(name="partial_1of2", tools=up[1:] + down, corrupt=[]))
    for c in out:
        c["label"] = LABELS[c["name"]]
    return out
