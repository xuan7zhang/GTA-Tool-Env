"""Score every TGB coalition with one model: gold likelihood + greedy answer.

    GD_MODEL=$GTA_BIG/models/Qwen2.5-7B-Instruct GD_TAG=7b \
      python -m tgb.score_dl --tasks $TGB/tgb_tasks.json --out $TGB

For each task x each candidate coalition it records

    L    length-normalized log p(gold | question, coalition outputs)
    dL   L - L(no tools)                       <- the tool-utility signal
    acc  does greedy decoding under that coalition actually produce the gold

Both in one pass, because `acc` is what turns the ranking claim into a
closed-loop claim: closed_loop.py needs the accuracy the agent *would have
got* had a selector picked that coalition, and that is only well defined if
it was measured under exactly the context the selector was scoring.

`derivable` is accuracy under the `useful` coalition -- the capability gate.

Chunked and resumable
---------------------
Results are appended to `tgb_scored_<tag>.jsonl` after every chunk of tasks and
the final `.json` is written at the end. Three separate overnight runs on this
cluster were killed mid-flight -- the log stops inside a progress bar with no
error -- and because scoring used to write once at the very end, a run cut off
at 79% produced nothing at all. Now a kill costs at most one chunk, and
re-running the same command skips whatever is already on disk.
"""
import argparse
import json
import os
import re

from vllm import LLM, SamplingParams

TPL = ("Answer the question with a short final answer only.\n\n"
       "Question: {q}\n\nTool output:\n{o}\n\nFinal answer:")
MAXLEN = int(os.environ.get("GD_MAXLEN", "4096"))


def span(tok, prompt, gold, maxlen, cap=None):
    """Token ids for `prompt + " " + gold`, plus where the gold starts and how
    long it is, so the caller can average the logprobs of exactly the gold.

    The obvious version -- tokenize the gold on its own and concatenate -- is
    wrong, and wrong in a way that flips signs rather than adding noise. TPL
    ends in "Final answer:" with no trailing space, so the model's next token
    is " "; tokenizing "215" alone gives the space-less token '215', which the
    model would essentially never emit at that position. A model made
    *confident* by a useful tool puts more mass on " 215" and therefore less on
    the variant being scored, so dL goes negative exactly where the tool helps:
    schedule_gap read dL = -1.79 while answering 0.830 correctly.

    Severity tracks how much of the gold that one token is. Llama merges "215"
    into a single token, so it is 100% of L (mean -13.3); Qwen splits it into
    '2','1','5', so it is 33% (mean -4). That, not length normalisation, is why
    the two models' L medians sat 3.5 nat apart.

    Truncation trims the head. The old code cut the prompt's tail, which eats
    "Final answer:" itself -- the one part of the prompt that has to survive.
    """
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full = tok(prompt + " " + gold, add_special_tokens=False)["input_ids"]
    n = 0
    while n < len(ids) and n < len(full) and ids[n] == full[n]:
        n += 1
    ng = len(full) - n
    if ng <= 0:                      # a merge ate the boundary; fall back
        gi = tok(" " + gold, add_special_tokens=False)["input_ids"]
        full, n, ng = ids + gi, len(ids), len(gi)
    if cap and ng > cap:
        full, ng = full[:n + cap], cap
    if len(full) > maxlen:
        cut = len(full) - maxlen
        full, n = full[cut:], n - cut
    return full, n, ng


def correct(text, gold):
    """Numeric match at a digit boundary, with 3 and 3.00 treated as equal.

    Without the equivalence a family whose answer is a count -- minutes,
    units, a capacity -- is scored wrong for every task, because the generator
    formats gold to two decimals and the model quite reasonably answers "3".
    Two v4 families read 0.000 across every model and every condition purely
    from this; the chains were fine and the models had it right.
    """
    t = text.replace(",", "")
    g = gold.strip().lstrip("$")
    forms = {g}
    try:
        v = float(g)
        forms |= {f"{v:.2f}", f"{v:g}"}
        if v == int(v):
            forms.add(str(int(v)))
    except ValueError:
        pass
    return int(any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d])", t)
                   for f in forms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.environ.get(
        "TGB_TASKS", "/datasets/omni_pretraining/gta2/results/taco/tgb/tgb_tasks.json"))
    ap.add_argument("--out", default=os.environ.get(
        "TGB_OUT", "/datasets/omni_pretraining/gta2/results/taco/tgb"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=200,
                    help="tasks per flush; a kill costs at most this many")
    ap.add_argument("--restart", action="store_true",
                    help="ignore any existing shard and start over")
    a = ap.parse_args()

    tag = os.environ.get("GD_TAG", "7b")
    model = os.environ.get(
        "GD_MODEL", "/datasets/omni_pretraining/gta2/models/Qwen2.5-7B-Instruct")
    tasks = json.load(open(a.tasks))
    if a.limit:
        tasks = tasks[:a.limit]

    shard = os.path.join(a.out, f"tgb_scored_{tag}.jsonl")
    done = {}
    if os.path.exists(shard) and not a.restart:
        with open(shard) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue            # a torn final line from a hard kill
                done[r["id"]] = r
        print(f"[{tag}] resuming: {len(done)} tasks already scored")
    todo = [t for t in tasks if t["id"] not in done]
    print(f"[{tag}] {len(todo)} tasks to score, chunk={a.chunk}")

    if todo:
        llm = LLM(model=model,
                  tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
                  dtype="bfloat16",
                  gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
                  max_model_len=MAXLEN)
        tok = llm.get_tokenizer()
        sp_lp = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)
        # 64, not 24: at 24 a verbose model (Qwen3 opens with "Okay, let me try
        # to figure out...") is scored wrong for being verbose rather than for
        # being wrong, which would silently shrink the derivable subset for that
        # model only. The gold check is a substring search, so a longer window
        # can only rescue a correct answer that arrives late.
        sp_gen = SamplingParams(max_tokens=int(os.environ.get("GD_GENTOK", "64")),
                                temperature=0)

        fh = open(shard, "a", buffering=1)
        for c0 in range(0, len(todo), a.chunk):
            batch = todo[c0:c0 + a.chunk]
            prompts, meta = [], []
            for t in batch:
                for name, cd in t["conditions"].items():
                    prompts.append(TPL.format(q=t["question"], o=cd["context"]))
                    meta.append((t["id"], name))
            gold_by = {t["id"]: t["gold"] for t in batch}

            jobs = [span(tok, p, gold_by[tid], MAXLEN)
                    for p, (tid, _) in zip(prompts, meta)]
            outs = llm.generate([{"prompt_token_ids": j[0]} for j in jobs], sp_lp)
            L = []
            for (full, plen, ng), o in zip(jobs, outs):
                lps = []
                for k in range(plen, plen + ng):
                    d = o.prompt_logprobs[k]
                    if d is None:
                        continue
                    lp = d.get(full[k])
                    if lp is not None:
                        lps.append(lp.logprob if hasattr(lp, "logprob")
                                   else float(lp))
                L.append(sum(lps) / len(lps) if lps else None)
            gouts = llm.generate([{"prompt": p} for p in prompts], sp_gen)

            rows = {t["id"]: dict(id=t["id"], family=t["family"],
                                  gold=t["gold"], conds={}) for t in batch}
            for (tid, name), ll, go in zip(meta, L, gouts):
                txt = go.outputs[0].text
                rows[tid]["conds"][name] = dict(L=ll,
                                                acc=correct(txt, gold_by[tid]),
                                                gen=txt.strip()[:80])
            for t in batch:
                r = rows[t["id"]]
                base = r["conds"].get("none", {}).get("L")
                for name, cd in r["conds"].items():
                    cd["dL"] = (cd["L"] - base) if (cd["L"] is not None
                                                    and base is not None) else None
                r["derivable"] = r["conds"].get("useful", {}).get("acc", 0)
                r["n_tools"] = {n: len(t["conditions"][n]["tools"])
                                for n in r["conds"]}
                fh.write(json.dumps(r) + "\n")
                done[t["id"]] = r
            fh.flush()
            os.fsync(fh.fileno())
            print(f"[{tag}] flushed {min(c0 + a.chunk, len(todo))}/{len(todo)}",
                  flush=True)
        fh.close()

    res = [done[t["id"]] for t in tasks if t["id"] in done]
    path = os.path.join(a.out, f"tgb_scored_{tag}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"[{tag}] wrote {path}  ({len(res)} tasks)")
    d = sum(r["derivable"] for r in res)
    print(f"[{tag}] derivable (correct under the useful coalition): "
          f"{d}/{len(res)} = {d / max(1, len(res)):.1%}")


if __name__ == "__main__":
    main()
