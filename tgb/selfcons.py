"""Zero-label deployable proxy on TGB: self-consistency of K sampled answers
per coalition.

    GD_TAG=7b GD_K=8 python -m tgb.selfcons --tasks ... --out ...

Agreement = frequency of the modal normalized answer among K samples. It needs
no gold, so unlike dL it is deployable; the question the paper asks of it here
is whether it can rank *coalitions* (not just evidence) well enough to drive
selection. Run on a subsample -- K x |conditions| generations per task is the
most expensive thing in the suite.
"""
import argparse
import collections
import json
import os
import re

from vllm import LLM, SamplingParams

TPL = ("Answer the question with a short final answer only.\n\n"
       "Question: {q}\n\nTool output:\n{o}\n\nFinal answer:")


def norm(t):
    m = re.search(r"-?\d+(?:\.\d+)?", t.replace(",", ""))
    return f"{float(m.group()):.2f}" if m else t.strip().lower()[:24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.environ.get(
        "TGB_TASKS", "/datasets/omni_pretraining/gta2/results/taco/tgb/tgb_tasks.json"))
    ap.add_argument("--out", default=os.environ.get(
        "TGB_OUT", "/datasets/omni_pretraining/gta2/results/taco/tgb"))
    ap.add_argument("--n", type=int, default=400, help="tasks to subsample")
    a = ap.parse_args()
    tag, K = os.environ.get("GD_TAG", "7b"), int(os.environ.get("GD_K", "8"))
    tasks = json.load(open(a.tasks))
    step = max(1, len(tasks) // a.n)
    tasks = tasks[::step][:a.n]

    llm = LLM(model=os.environ.get(
        "GD_MODEL",
        "/datasets/omni_pretraining/gta2/models/Qwen2.5-7B-Instruct"),
        tensor_parallel_size=int(os.environ.get("GD_TP", "1")), dtype="bfloat16",
        gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
        max_model_len=4096)
    sp = SamplingParams(n=K, max_tokens=24, temperature=0.7, top_p=0.95, seed=7)

    prompts, meta = [], []
    for t in tasks:
        for name, c in t["conditions"].items():
            prompts.append(TPL.format(q=t["question"], o=c["context"]))
            meta.append((t["id"], name))
    outs = llm.generate([{"prompt": p} for p in prompts], sp)

    rows = {t["id"]: dict(id=t["id"], family=t["family"], conds={})
            for t in tasks}
    for (tid, name), o in zip(meta, outs):
        answers = [norm(c.text) for c in o.outputs]
        top, cnt = collections.Counter(answers).most_common(1)[0]
        rows[tid]["conds"][name] = dict(agree=cnt / len(answers), modal=top)
    path = os.path.join(a.out, f"tgb_selfcons_{tag}_K{K}.json")
    json.dump(list(rows.values()), open(path, "w"), indent=1)
    print(f"[{tag}] wrote {path} ({len(rows)} tasks, K={K})")


if __name__ == "__main__":
    main()
