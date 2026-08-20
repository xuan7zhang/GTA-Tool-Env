"""Cache what each poison tool would return, per GTA task.

The eight poison tools are deterministic pure functions of their text argument
(a sha256 pick from a fixed wrong-answer pool), so their observations can be
materialised without the tool server. They are the reason a padded pool is a
real test: unlike an inert distractor, keeping one of these puts a confident
wrong answer into the context, so a wrong keep-decision finally costs
something.

Writes tool_outputs22.json = the nine text-renderable real tools' cached
outputs plus the eight poison ones, the context build_loo_mask needs.
"""
import json, sys, os

BIG = "/datasets/omni_pretraining/gta2"
POISON_DIR = "/project/6101776/xzhan576/gta2-envlab/envgen/variants/poison"
sys.path.insert(0, POISON_DIR)
import poison_tools as P  # noqa: E402

NAMES = json.load(open(f"{POISON_DIR}/names.json"))
DS = json.load(open(f"{BIG}/data/gta_dataset/dataset.json"))
TO = json.load(open(f"{BIG}/results/inject_opt/tool_outputs.json"))


def question(k):
    for m in DS[str(k)]["dialogs"]:
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(
                x.get("text", "") for x in c if isinstance(x, dict))
    return ""


# each tool's text argument; image arguments are ignored by these stubs
ARG = {
    "DirectImageQA":  lambda q: dict(image=None, question=q),
    "InstantAnswer":  lambda q: dict(question=q),
    "PreciseOCR":     lambda q: dict(image=None),
    "SmartCount":     lambda q: dict(image=None, object=q[:40]),
    "ExpertMath":     lambda q: dict(expression=q),
    "VerifiedFact":   lambda q: dict(query=q),
    "ImageDetailPro": lambda q: dict(image=None),
    "QuickBBox":      lambda q: dict(image=None, object=q[:40]),
}

out, n = {}, 0
for k in TO:
    q = question(k)
    rec = dict(TO[k])
    for name in NAMES:
        cls = getattr(P, name)
        try:
            rec[name] = cls.apply(cls.__new__(cls), **ARG[name](q))
            n += 1
        except Exception as e:
            rec[name] = f"[{name} unavailable: {e}]"
    out[k] = rec
p = f"{BIG}/results/inject_opt/tool_outputs22.json"
json.dump(out, open(p, "w"))
print(f"wrote {p}: {len(out)} tasks, {n} poison outputs")
print("sample:", json.dumps({k: out['0'][k] for k in NAMES[:3]}, indent=1)[:400])
