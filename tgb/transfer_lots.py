"""Serve one model's LOTS masks to another model, on TGB-v4.

The masks are per task, so a mask chosen by model X is directly servable by
model Y: same task, same tool outputs, only the menu differs. This is the
answer-likelihood counterpart of a scale-transfer matrix.

    GD_TAG=14b GD_MODEL=/path/Qwen2.5-14B-Instruct python -m tgb.transfer_lots \
        --src 3b,7b,14b,32b --limit 1200

Arms per evaluating model: its own mask (diagonal), each other model's mask
(off-diagonal), plus `full` (the whole 15-tool menu), `oracle` (the task's
gt_tools) and `random` (size-matched to the evaluator's own mask, so that
"fewer tools" and "the right tools" stay separable).
"""
import argparse, collections, json, os, random, statistics as st, tempfile, zlib

from vllm import LLM, SamplingParams

from . import scenes, tools_v2, tools_v4
from .generate import TPL, context
from .score_dl import correct
from .tools import run_chain

DIR = "/datasets/omni_pretraining/gta2/results/taco/tgb4"


def load_masks(tag, fixed_tau=None):
    """masks at a common threshold across models.

    Each model's own tau maximises its own train accuracy, which makes the
    rows incomparable: 32B's self-optimal tau keeps 14.8 of 15 tools, so its
    "mask" is just the full menu. Holding tau fixed asks what the matrix is
    for, whose ranking of tools is better, rather than who picked the kindest
    threshold for themselves.
    """
    p = os.path.join(DIR, f"tgb_loo_{tag}_self_S.json")
    d = json.load(open(p))
    sw = d["sweep"]
    want = d["selected"]["tau"] if fixed_tau is None else fixed_tau
    cand = [k for k in sw if k != "S"]
    tau = min(cand, key=lambda k: abs(float(k) - want))
    return sw[tau]["masks"], d["S"], float(tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--src", default="3b,7b,14b,32b",
                    help="comma list of source tags whose masks to serve")
    ap.add_argument("--mask-tau", dest="mask_tau", type=float, default=None,
                    help="use every source model's mask at this threshold")
    ap.add_argument("--limit", type=int, default=1200,
                    help="evaluate on the first N test tasks (deterministic)")
    ap.add_argument("--train", type=int, default=400,
                    help="same train split the mask runs held out")
    a = ap.parse_args()
    tools_v2.register(); tools_v4.register()
    tag = os.environ["GD_TAG"]

    tasks = json.load(open(os.path.join(a.dir, "tgb_tasks.json")))
    step = max(1, len(tasks) // a.train)
    tr_ids = {t["id"] for t in tasks[::step][:a.train]}
    te = [t for t in tasks if t["id"] not in tr_ids]
    _st = max(1, len(te) // a.limit) if a.limit else 1
    test = te[::_st][: a.limit] if a.limit else te
    tmp = tempfile.mkdtemp(prefix="tgb_xfer_")
    for t in test:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    srcs = [s for s in a.src.split(",") if s]
    masks, S = {}, None
    for s in srcs:
        try:
            m, S_, tau = load_masks(s, a.mask_tau)
            masks[s] = m
            S = S or S_
            print(f"[{tag}] loaded masks from {s} (tau={tau}, {len(m)} tasks)", flush=True)
        except FileNotFoundError:
            print(f"[{tag}] no mask file for {s}, skipping", flush=True)
    if S is None:
        from .coalitions_v4 import FULL_MENU_V4
        S = list(FULL_MENU_V4)

    def ctx_of(t, mask):
        rng = random.Random(zlib.crc32(f"{t['id']}/{'+'.join(sorted(mask))}".encode()))
        outs, _ = run_chain(t, t["scene"], t["boxes"], set(mask), rng, corrupt_tools=())
        return context(outs, [s["tool"] for s in t["plan"]])

    arms = {}
    for s in srcs:
        if s in masks:
            arms[f"from_{s}"] = {t["id"]: masks[s].get(t["id"], list(S)) for t in test}
    arms["full"] = {t["id"]: list(S) for t in test}
    arms["oracle"] = {t["id"]: (list(t["gt_tools"]) or list(S)) for t in test}
    own = masks.get(tag)
    if own:
        rng = random.Random(0)
        arms["random"] = {t["id"]: rng.sample(list(S), max(1, len(own.get(t["id"], S))))
                          for t in test}

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    sp = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")), temperature=0)

    prompts, meta = [], []
    for arm, mp in arms.items():
        for t in test:
            prompts.append(TPL.format(q=t["question"], o=ctx_of(t, mp[t["id"]])))
            meta.append((arm, t["id"]))
    print(f"[{tag}] {len(arms)} arms x {len(test)} tasks = {len(prompts)} generations", flush=True)
    outs = llm.generate([{"prompt": p} for p in prompts], sp)

    gold = {t["id"]: t["gold"] for t in test}
    gt = {t["id"]: set(t["gt_tools"]) for t in test}
    res = collections.defaultdict(lambda: dict(acc=[], tools=[], rec=[]))
    for (arm, tid), o in zip(meta, outs):
        r = res[arm]
        r["acc"].append(correct(o.outputs[0].text, gold[tid]))
        m = arms[arm][tid]
        r["tools"].append(len(m))
        if gt[tid]:
            r["rec"].append(len(set(m) & gt[tid]) / len(gt[tid]))

    out = {"eval_model": tag, "n_test": len(test), "arms": {}}
    print(f"\n[{tag}] {'arm':14s}{'acc':>8}{'tools':>8}{'recall':>8}")
    for arm, r in res.items():
        row = dict(acc=st.mean(r["acc"]), tools=st.mean(r["tools"]),
                   recall=st.mean(r["rec"]) if r["rec"] else None)
        out["arms"][arm] = row
        print(f"[{tag}] {arm:14s}{row['acc']:8.3f}{row['tools']:8.2f}"
              f"{(row['recall'] if row['recall'] is not None else float('nan')):8.3f}")
    p = os.path.join(a.dir, f"tgb_transfer_{tag}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
