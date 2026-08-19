"""RaTA-Set-PORTS Stage 1.3 -- semantic retrieval baselines B1 and B2.

Same frozen model as the agent, same encoder for both arms; the only thing that
differs is the query representation:

    B1  cos(E(raw question), E(tool spec))
    B2  cos(E(structured summary), E(tool spec))

The project ships no sentence encoder, so the encoder is the agent's own model
with last-token pooling -- the standard decoder-only choice, and the honest one
here because it keeps "frozen model already in the project" true.

Cardinality is not fixed by fiat. Three mask rules are evaluated and their one
free parameter is chosen on dev, never on test:

    top-k            k in {1..5}
    global threshold on the cosine
    variable size    score >= max_score - delta

Retrieval metrics are reported, and then the frozen agent is actually run,
because on this benchmark they disagree: Stage 0 measured a per-task oracle
whose set-F1 against the ground-truth chain is 0.382 yet which scores 8.5 pp
*above* the ground-truth chain itself. Set-F1 is not the objective.

    GD_TAG=7b GD_MODEL=... python -m tgb.rata.stage1_retrieval
"""
import argparse
import collections
import json
import os
import random
import statistics as st
import tempfile
import zlib

import torch

from .. import scenes, tools_v2
from ..add_chains_v2 import CHAINS
from ..families_v2 import NUISANCE, UNION_TOOLS
from ..generate import TPL, context
from ..score_dl import correct
from ..tools import IMAGE_TOOLS, run_chain
from .tool_specs import SPECS, doc_text

tools_v2.register()

MENU = UNION_TOOLS + NUISANCE + IMAGE_TOOLS
OUT = "/datasets/omni_pretraining/gta2/results/taco/tgb2/rata_set_ports"
TGB2 = "/datasets/omni_pretraining/gta2/results/taco/tgb2"
FAM = ["f1_extract_compute", "f2_visual_reason", "f3_retrieve_reason",
       "gauge_over", "temp_over", "equation_short", "timetable_over",
       "invoice_currency"]


def embed(texts, model_path, bs=16):
    """Last-token pooled hidden states from the frozen agent model."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16,
                                    device_map="cuda").eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                    truncation=True, max_length=512).to("cuda")
            h = mdl(**b).last_hidden_state
            last = b["attention_mask"].sum(1) - 1
            v = h[torch.arange(h.size(0)), last].float()
            vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
    del mdl
    torch.cuda.empty_cache()
    return torch.cat(vecs)


def metrics(sel, ids, T, ACC, ntok):
    r, p, f1, tr, sz, ac, tk = [], [], [], [], [], [], []
    for i in ids:
        S, G = set(sel[i]), set(CHAINS[T[i]["family"]])
        rr = len(S & G) / len(G)
        pp = len(S & G) / max(1, len(S))
        r.append(rr)
        p.append(pp)
        f1.append(0 if rr + pp == 0 else 2 * rr * pp / (rr + pp))
        tr.append(int(G <= S))
        sz.append(len(S))
        ac.append(ACC[(i, tuple(sorted(S)))])
        tk.append(ntok[(i, tuple(sorted(S)))])
    return dict(acc=st.mean(ac), gt_recall=st.mean(r), gt_precision=st.mean(p),
                set_f1=st.mean(f1), tracc=st.mean(tr), mask_size=st.mean(sz),
                ctx_tok=st.mean(tk))


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")
    model = os.environ["GD_MODEL"]

    rows = []
    for s in ("train", "dev", "test"):
        with open(f"{OUT}/summaries/{s}_{tag}.jsonl") as fh:
            rows += [json.loads(x) for x in fh]
    S = {r["task_id"]: r for r in rows}
    tasks_all = {t["id"]: t for t in json.load(open(f"{TGB2}/tgb_tasks.json"))}
    tasks = [tasks_all[r["task_id"]] for r in rows]
    T = {t["id"]: t for t in tasks}
    dev = [r["task_id"] for r in rows if r["split"] == "dev"]
    test = [r["task_id"] for r in rows if r["split"] == "test"]
    print(f"[{tag}] dev {len(dev)}  test {len(test)}")

    # ------------------------------------------------------------- encode
    docs = [doc_text(sp) for sp in SPECS]
    names = [sp["tool_name"] for sp in SPECS]
    qs_raw = [S[i]["raw_question"] for i in S]
    qs_sum = [S[i]["summary_text"] for i in S]
    order = list(S)
    E = embed(docs + qs_raw + qs_sum, model)
    D = E[:len(docs)]
    Qr = E[len(docs):len(docs) + len(order)]
    Qs = E[len(docs) + len(order):]
    SC = {"B1_raw": (Qr @ D.T), "B2_summary": (Qs @ D.T)}
    pos = {i: k for k, i in enumerate(order)}

    # ------------------------------------------- candidate mask rules
    def masks_for(arm, rule, param):
        out = {}
        for i in order:
            v = SC[arm][pos[i]]
            if rule == "topk":
                idx = torch.topk(v, param).indices.tolist()
            elif rule == "thresh":
                idx = (v >= param).nonzero().flatten().tolist() or \
                    [int(v.argmax())]
            else:                                   # relative window
                idx = (v >= v.max() - param).nonzero().flatten().tolist()
            out[i] = sorted(names[j] for j in idx)
        return out

    GRID = ([("topk", k) for k in range(1, 6)] +
            [("thresh", t) for t in [.55, .6, .65, .7, .75, .8]] +
            [("relwin", d) for d in [.01, .02, .03, .05, .08, .12]])
    CAND = {f"{arm}|{r}|{p}": masks_for(arm, r, p)
            for arm in SC for r, p in GRID}

    # ------------------------------------------------- execute + score
    tmp = tempfile.mkdtemp(prefix="rata1_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))
    need = sorted({(i, tuple(sorted(m[i]))) for m in CAND.values()
                   for i in order} |
                  {(i, tuple(sorted(CHAINS[T[i]["family"]]))) for i in order})
    print(f"  distinct (task, mask) to execute: {len(need)}")

    from vllm import LLM, SamplingParams
    llm = LLM(model=model, tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.80")),
              max_model_len=4096)
    tk = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=64, temperature=0)
    prompts, ntok = [], {}
    for i, M in need:
        t = T[i]
        rng = random.Random(zlib.crc32(f"{i}/{'+'.join(M)}".encode()))
        outs, _ = run_chain(t, t["scene"], t["boxes"], set(M), rng,
                            corrupt_tools=())
        ctx = context(outs, [x["tool"] for x in t["plan"]])
        ntok[(i, M)] = len(tk(ctx, add_special_tokens=False)["input_ids"])
        prompts.append(TPL.format(q=t["question"], o=ctx))
    gens = llm.generate([{"prompt": p} for p in prompts], sp)
    ACC = {k: correct(g.outputs[0].text, T[k[0]]["gold"])
           for k, g in zip(need, gens)}

    # ---------------------------------------- select rule on dev, report test
    res, chosen = {}, {}
    for arm in SC:
        best, bestv = None, -1
        for key, m in CAND.items():
            if not key.startswith(arm):
                continue
            v = metrics(m, dev, T, ACC, ntok)["acc"]
            if v > bestv:
                best, bestv = key, v
        chosen[arm] = best
        res[arm] = {"rule": best, "dev_acc": bestv,
                    "test": metrics(CAND[best], test, T, ACC, ntok)}
        res[arm]["by_family"] = {
            f: metrics(CAND[best], [i for i in test if T[i]["family"] == f],
                       T, ACC, ntok)["acc"] for f in FAM}
        res[arm]["all_rules_test"] = {
            k.split("|", 1)[1]: metrics(m, test, T, ACC, ntok)
            for k, m in CAND.items() if k.startswith(arm)}

    json.dump({f"{i}|{'+'.join(M)}": v for (i, M), v in ACC.items()},
              open(f"{OUT}/eval/stage1_mask_acc_{tag}.json", "w"))
    gt = {i: sorted(CHAINS[T[i]["family"]]) for i in order}
    res["M12_gt_toolset"] = {"test": metrics(gt, test, T, ACC, ntok)}
    json.dump({"tag": tag, "chosen": chosen, "res": res},
              open(f"{OUT}/eval/stage1_retrieval_{tag}.json", "w"), indent=1)
    json.dump({k: {i: v[i] for i in order} for k, v in CAND.items()
               if k in chosen.values()},
              open(f"{OUT}/retriever/stage1_masks_{tag}.json", "w"), indent=1)

    print(f"\n[{tag}] Stage-1 retrieval, dev-selected rule, test n={len(test)}")
    print(f"  {'arm':<16} {'rule':<14} {'acc':>6} {'setF1':>6} {'TRACC':>6} "
          f"{'rec':>5} {'prec':>5} {'|S|':>5} {'tok':>6}")
    for k in ["B1_raw", "B2_summary", "M12_gt_toolset"]:
        m = res[k]["test"]
        rule = res[k].get("rule", "-").split("|", 1)[-1]
        print(f"  {k:<16} {rule:<14} {m['acc']:6.3f} {m['set_f1']:6.3f} "
              f"{m['tracc']:6.3f} {m['gt_recall']:5.2f} {m['gt_precision']:5.2f}"
              f" {m['mask_size']:5.1f} {m['ctx_tok']:6.1f}")

    b1, b2 = res["B1_raw"]["test"], res["B2_summary"]["test"]
    gate = {"d_set_f1_pp": 100 * (b2["set_f1"] - b1["set_f1"]),
            "d_tracc_pp": 100 * (b2["tracc"] - b1["tracc"]),
            "d_acc_pp": 100 * (b2["acc"] - b1["acc"])}
    gate["pass"] = bool(gate["d_set_f1_pp"] >= 3 or gate["d_tracc_pp"] >= 3
                        or gate["d_acc_pp"] >= 2)
    json.dump(gate, open(f"{OUT}/eval/stage1_gate_{tag}.json", "w"), indent=1)
    print(f"\n  STAGE-1 GATE  dsetF1 {gate['d_set_f1_pp']:+.1f}pp  "
          f"dTRACC {gate['d_tracc_pp']:+.1f}pp  dAcc {gate['d_acc_pp']:+.1f}pp"
          f"  -> {'PASS' if gate['pass'] else 'FAIL'}")
    print("\n  per-family acc (B2 summary):")
    for f in FAM:
        print(f"    {f:<22} {res['B2_summary']['by_family'][f]:.3f}  "
              f"(B1 {res['B1_raw']['by_family'][f]:.3f})")


if __name__ == "__main__":
    main()
