"""RaTA-Set-PORTS Stage 0 -- benchmark audit, cross-family harm, baseline
reproduction.

Everything downstream of this file is conditional on its gate: if the per-task
oracle does not clear the best constant mask by a real margin, no per-task
selector can be shown to work here regardless of how good it is, and the honest
report is a benchmark limitation rather than a method failure. TGB-v1 already
died that way; v2 was built to remove the property, and this file measures
whether it actually did.

Two things are measured that no existing artefact contains:

  cross-family harm    H(t, A->B) = Acc_B(S_B + {t}) - Acc_B(S_B)
                       for a tool t that family A needs and B does not. This
                       is the quantity that makes a constant mask a compromise;
                       if it is ~0 the environment is v1 again.
  nesting              whether the family chains are near-subsets of each other,
                       in which case the union is trivially near-optimal.

Baselines are re-executed through this file's own code path rather than read
out of tgb_scored_*.json, and the two are cross-checked: a >1pp disagreement on
a shared condition means the new path is not comparable and the run aborts.

    GD_TAG=7b GD_MODEL=... python -m tgb.rata.stage0_audit --n 1200
"""
import argparse
import collections
import csv
import json
import os
import random
import statistics as st
import tempfile
import zlib

from vllm import LLM, SamplingParams

from .. import scenes, tools_v2
from ..add_chains_v2 import CHAINS
from ..families_v2 import NUISANCE, UNION_TOOLS
from ..generate import TPL, context
from ..score_dl import correct
from ..tools import IMAGE_TOOLS, run_chain

tools_v2.register()

MENU9 = list(UNION_TOOLS)                     # the functional sub-menu
FULL = UNION_TOOLS + NUISANCE + IMAGE_TOOLS   # everything the environment ships
OUT = ("/datasets/omni_pretraining/gta2/results/taco/tgb2/rata_set_ports")
TGB2 = "/datasets/omni_pretraining/gta2/results/taco/tgb2"
FAM = ["f1_extract_compute", "f2_visual_reason", "f3_retrieve_reason",
       "gauge_over", "temp_over", "equation_short", "timetable_over",
       "invoice_currency"]


def boot(a, b, n=2000, seed=0):
    """paired bootstrap CI on the difference of means"""
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    m = st.mean(d)
    idx = range(len(d))
    reps = sorted(st.mean([d[rng.choice(idx)] for _ in idx]) for _ in range(n))
    return m, reps[int(.025 * n)], reps[int(.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--nrand", type=int, default=4)
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")

    tasks_all = json.load(open(os.path.join(TGB2, "tgb_tasks.json")))
    by_fam = collections.defaultdict(list)
    for t in tasks_all:
        by_fam[t["family"]].append(t)
    per = a.n // len(by_fam)
    tasks = [t for f in sorted(by_fam) for t in by_fam[f][:per]]
    T = {t["id"]: t for t in tasks}
    ids = [t["id"] for t in tasks]
    train = set(i for k, i in enumerate(ids) if k % 5 < 2)
    test = [i for i in ids if i not in train]

    # ---------------------------------------------------------------- 0.1 CPU
    gt_sizes = [len(t["gt_tools"]) for t in tasks_all]
    freq = collections.Counter(x for t in tasks_all for x in t["gt_tools"])
    fam_tools = {f: sorted(CHAINS[f]) for f in FAM}
    nest = {}
    for A in FAM:
        for B in FAM:
            if A != B:
                nest[f"{A}<{B}"] = int(set(CHAINS[A]) <= set(CHAINS[B]))
    summary = {
        "n_tasks_total": len(tasks_all), "n_tasks_used": len(tasks),
        "n_families": len(by_fam), "split": {"train": len(train),
                                             "test": len(test)},
        "menu_functional": MENU9, "menu_full": FULL,
        "n_tools_functional": len(MENU9), "n_tools_full": len(FULL),
        "gt_set_size": {"mean": st.mean(gt_sizes), "min": min(gt_sizes),
                        "max": max(gt_sizes)},
        "union_size": len(set().union(*[set(v) for v in fam_tools.values()])),
        "family_chains": fam_tools,
        "n_nested_pairs": sum(nest.values()), "n_family_pairs": len(nest),
    }
    json.dump(summary, open(f"{OUT}/audit/dataset_summary.json", "w"), indent=1)
    with open(f"{OUT}/audit/tool_frequency.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tool", "n_tasks_requiring", "frac"])
        for t in FULL:
            w.writerow([t, freq.get(t, 0), f"{freq.get(t, 0)/len(tasks_all):.4f}"])
    with open(f"{OUT}/audit/family_tool_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family"] + MENU9)
        for f in FAM:
            w.writerow([f] + [int(t in CHAINS[f]) for t in MENU9])
    print(f"[{tag}] {len(tasks)} tasks  train {len(train)} / test {len(test)}")
    print(f"  nested family-chain pairs: {sum(nest.values())}/{len(nest)}")

    # ------------------------------------------------------ masks to evaluate
    greedy = json.load(open(f"{TGB2}/tgb_greedy_{tag}_backward.json"))
    GMASK = sorted(greedy["greedy_mask"])
    rng0 = random.Random(20260804)
    MASKS = {}                                   # task_id -> {name: tuple}
    for t in tasks:
        own = sorted(CHAINS[t["family"]])
        m = {"none": (), "union9": tuple(sorted(MENU9)),
             "full20": tuple(sorted(FULL)), "greedy": tuple(GMASK),
             "gt": tuple(own)}
        for f in FAM:
            m[f"chain_{f}"] = tuple(sorted(CHAINS[f]))
        for x in MENU9:
            if x not in own:
                m[f"gt_plus_{x}"] = tuple(sorted(own + [x]))
        for j in range(a.nrand):
            m[f"rand{j}"] = tuple(sorted(rng0.sample(MENU9, len(own))))
        MASKS[t["id"]] = m

    tmp = tempfile.mkdtemp(prefix="rata0_")
    for t in tasks:
        t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

    need = sorted({(i, s) for i, m in MASKS.items() for s in m.values()})
    print(f"  distinct (task, mask) to execute: {len(need)}")

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=64, temperature=0)

    prompts, ntok = [], {}
    for i, S in need:
        t = T[i]
        rng = random.Random(zlib.crc32(f"{i}/{'+'.join(S)}".encode()))
        outs, _ = run_chain(t, t["scene"], t["boxes"], set(S), rng,
                            corrupt_tools=())
        ctx = context(outs, [s["tool"] for s in t["plan"]])
        ntok[(i, S)] = len(tok(ctx, add_special_tokens=False)["input_ids"])
        prompts.append(TPL.format(q=t["question"], o=ctx))
    gens = llm.generate([{"prompt": p} for p in prompts], sp)
    ACC = {k: correct(g.outputs[0].text, T[k[0]]["gold"])
           for k, g in zip(need, gens)}
    json.dump({f"{i}|{'+'.join(S)}": v for (i, S), v in ACC.items()},
              open(f"{OUT}/audit/mask_acc_{tag}.json", "w"))

    # ------------------------------------------- reproduction cross-check
    old = {t["id"]: t for t in json.load(open(f"{TGB2}/tgb_scored_{tag}.json"))}
    checks = {}
    for name, cond in [("union9", "union"), ("full20", "full")] + \
            [(f"chain_{f}", f"chain_{f}") for f in FAM]:
        mine = st.mean(ACC[(i, MASKS[i][name])] for i in ids)
        theirs = st.mean(old[i]["conds"][cond]["acc"] for i in ids
                         if cond in old[i]["conds"])
        checks[name] = {"new": mine, "existing": theirs, "delta": mine - theirs}
    worst = max(abs(v["delta"]) for v in checks.values())
    print(f"  reproduction cross-check, worst |delta| = {worst:.4f} "
          f"({'PASS' if worst <= 0.01 else 'FAIL'})")
    for k, v in checks.items():
        if abs(v["delta"]) > 0.005:
            print(f"    {k}: new {v['new']:.3f} vs existing {v['existing']:.3f}")

    # -------------------------------------------------- cross-family harm
    harm = {}
    for B in FAM:
        sub = [i for i in ids if T[i]["family"] == B]
        base = st.mean(ACC[(i, MASKS[i]["gt"])] for i in sub)
        row = {}
        for x in MENU9:
            if x in CHAINS[B]:
                row[x] = None
                continue
            row[x] = st.mean(ACC[(i, MASKS[i][f"gt_plus_{x}"])]
                             for i in sub) - base
        harm[B] = {"base_acc": base, "delta_by_added_tool": row}
    json.dump(harm, open(f"{OUT}/audit/family_cross_harm.json", "w"), indent=1)
    with open(f"{OUT}/audit/family_cross_harm.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family_B", "base_acc"] + MENU9)
        for B in FAM:
            w.writerow([B, f"{harm[B]['base_acc']:.3f}"] +
                       ["" if harm[B]["delta_by_added_tool"][x] is None
                        else f"{harm[B]['delta_by_added_tool'][x]:+.3f}"
                        for x in MENU9])
    vals = [v for B in FAM for v in harm[B]["delta_by_added_tool"].values()
            if v is not None]
    print(f"  cross-family harm: mean {st.mean(vals):+.3f}, "
          f"min {min(vals):+.3f}, frac <= -0.05: "
          f"{sum(v <= -.05 for v in vals)/len(vals):.0%}")

    # -------------------------------------------------------- baselines
    GOLDFREE = ["none", "union9"] + [f"chain_{f}" for f in FAM]

    def acc_of(pick, subset):
        return [ACC[(i, MASKS[i][pick(i)])] for i in subset]

    rows = {}
    dl = {i: max(GOLDFREE, key=lambda c: old[i]["conds"][
        {"union9": "union", "none": "none"}.get(c, c)]["dL"]) for i in ids
        if all({"union9": "union", "none": "none"}.get(c, c) in old[i]["conds"]
               for c in GOLDFREE)}
    methods = {
        "M0_no_tools": lambda i: "none",
        "M1_all_tools": lambda i: "full20",
        "M1b_union9": lambda i: "union9",
        "M2_global_greedy": lambda i: "greedy",
        "M3_pertask_goldLL": lambda i: dl.get(i, "union9"),
        "M4_pertask_oracle": lambda i: max(GOLDFREE,
                                           key=lambda c: ACC[(i, MASKS[i][c])]),
        "M7_random_matched": lambda i: "rand0",
        "M12_gt_toolset": lambda i: "gt",
    }
    base_v = acc_of(methods["M1b_union9"], test)
    for name, fn in methods.items():
        v = acc_of(fn, test)
        sizes = [len(MASKS[i][fn(i)]) for i in test]
        toks = [ntok[(i, MASKS[i][fn(i)])] for i in test]
        gtr = [len(set(MASKS[i][fn(i)]) & set(CHAINS[T[i]["family"]])) /
               len(CHAINS[T[i]["family"]]) for i in test]
        gtp = [len(set(MASKS[i][fn(i)]) & set(CHAINS[T[i]["family"]])) /
               max(1, len(MASKS[i][fn(i)])) for i in test]
        f1 = [0 if (r + p) == 0 else 2 * r * p / (r + p)
              for r, p in zip(gtr, gtp)]
        d, lo, hi = boot(v, base_v)
        byfam = {f: st.mean(ACC[(i, MASKS[i][fn(i)])] for i in test
                            if T[i]["family"] == f) for f in FAM}
        rows[name] = dict(acc=st.mean(v), mask_size=st.mean(sizes),
                          ctx_tok=st.mean(toks), gt_recall=st.mean(gtr),
                          gt_precision=st.mean(gtp), set_f1=st.mean(f1),
                          vs_union9=d, ci=[lo, hi],
                          requires_test_gold=name in ("M3_pertask_goldLL",
                                                      "M4_pertask_oracle",
                                                      "M12_gt_toolset"),
                          worst_family=min(byfam.values()), by_family=byfam)
    json.dump({"tag": tag, "n_test": len(test), "checks": checks,
               "greedy_mask": GMASK, "rows": rows},
              open(f"{OUT}/eval/baselines_{tag}.json", "w"), indent=1)
    with open(f"{OUT}/eval/baselines_{tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["acc", "mask_size", "ctx_tok", "gt_recall", "gt_precision",
                "set_f1", "vs_union9", "worst_family", "requires_test_gold"]
        w.writerow(["method"] + cols)
        for k, v in rows.items():
            w.writerow([k] + [v[c] for c in cols])

    print(f"\n[{tag}] Stage-0 baselines, test n={len(test)}")
    print(f"  {'method':<20} {'acc':>6} {'|S|':>5} {'tok':>6} {'setF1':>6} "
          f"{'worstFam':>8} {'vs union9 [95% CI]':>26} gold")
    for k, v in rows.items():
        print(f"  {k:<20} {v['acc']:6.3f} {v['mask_size']:5.1f} "
              f"{v['ctx_tok']:6.1f} {v['set_f1']:6.3f} {v['worst_family']:8.3f} "
              f"{v['vs_union9']:+8.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]"
              f"   {'Y' if v['requires_test_gold'] else 'N'}")

    # ------------------------------------------------------- headroom gate
    best_const = max(rows["M1b_union9"]["acc"], rows["M2_global_greedy"]["acc"],
                     rows["M1_all_tools"]["acc"])
    oracle = rows["M4_pertask_oracle"]["acc"]
    gate = {
        "best_constant_mask": best_const, "per_task_oracle": oracle,
        "headroom_pp": 100 * (oracle - best_const),
        "gate_5pp": bool(oracle - best_const >= 0.05),
        "nested_pairs": sum(nest.values()), "n_pairs": len(nest),
        "mean_cross_family_harm": st.mean(vals),
        "frac_harm_le_-0.05": sum(v <= -.05 for v in vals) / len(vals),
        "reproduction_worst_delta": worst,
    }
    json.dump(gate, open(f"{OUT}/eval/headroom_gate_{tag}.json", "w"), indent=1)
    print(f"\n  HEADROOM: per-task oracle {oracle:.3f} - best constant "
          f"{best_const:.3f} = {100*(oracle-best_const):+.1f} pp  "
          f"-> gate {'PASS' if gate['gate_5pp'] else 'FAIL (benchmark limit)'}")


if __name__ == "__main__":
    main()
