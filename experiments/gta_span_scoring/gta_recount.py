"""Recount GTA answer accuracy on the answerable subset.

The shipped `answer_acc` divides by all 229 tasks but can only ever credit the
172 that carry a textual reference; the other 57 are image-generation tasks it
scores 0 by construction. That denominator flatters any method that drops the
five image-output tools, which is exactly what a text-likelihood estimator
does, so we report the 172-task subset where every arm is judged on tasks it
could in principle answer. Both numbers are printed; nothing is hidden.

    python gta_recount.py <run_dir> [<run_dir> ...]
"""
import json, os, re, sys, glob, collections, statistics as st

DATA = "/datasets/omni_pretraining/gta2/data/gta_dataset/dataset.json"


def iscorrect(pred, ref):
    count = 0
    for aliases in ref["whitelist"]:
        pat = r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b"
        if re.search(pat, pred, re.IGNORECASE):
            count += 1
    if not ref["blacklist"]:
        return count == len(ref["whitelist"])
    bk = r"\b(?:" + "|".join(re.escape(a) for al in ref["blacklist"] for a in al) + r")\b"
    return count == len(ref["whitelist"]) and not re.search(bk, pred, re.IGNORECASE)


def response(preds):
    """the evaluator reads preds[0][-1] and credits it only when it is an
    assistant answer rather than a tool call."""
    if not isinstance(preds, list) or not preds:
        return None
    traj = preds[0]
    if not isinstance(traj, list) or not traj:
        return None
    last = traj[-1]
    if not isinstance(last, dict) or "tool_calls" in last:
        return None
    if last.get("role") != "assistant":
        return None
    return last.get("content")


def main():
    items = json.load(open(DATA))
    items = items if isinstance(items, list) else list(items.values())
    refs = {str(i): it.get("gt_answer") for i, it in enumerate(items)}
    dict_ids = {k for k, v in refs.items() if isinstance(v, dict)}
    list_ids = {k for k, v in refs.items() if isinstance(v, list) and v}
    img_ids = {k for k, v in refs.items() if not v}

    for run in sys.argv[1:]:
        hits = glob.glob(f"{run}/**/predictions/**/gta_bench_end.json", recursive=True)
        if not hits:
            print(f"{run}: no predictions"); continue
        pd = json.load(open(sorted(hits)[-1]))
        ok = und = 0
        for k, v in pd.items():
            if k not in dict_ids:
                continue
            ans = response(v.get("prediction") or v.get("assistant_outputs"))
            if ans is None:
                und += 1; continue
            if iscorrect(str(ans), refs[k]):
                ok += 1
        n_all = len(pd)
        print(f"{os.path.basename(run):22s} scored-subset(dict refs)={len(dict_ids)}  "
              f"correct={ok}  acc_172dict={100*ok/len(dict_ids):5.2f}  "
              f"acc_if_229={100*ok/n_all:5.2f}  no-answer-emitted={und}")


if __name__ == "__main__":
    main()
