#!/usr/bin/env python3
"""Per-sample causal decomposition between runs A (passthrough) and B
(tool_free), plus survival under C (corrupt_output).

Recomputes per-sample end-mode correctness from OpenCompass prediction files
using the same scoring functions as GTABenchEvaluator (regex whitelist for
objective refs; sentence-transformer simscore for subjective refs, binarized
at --sim-threshold, default 0.5).

Outputs analysis/causal_report.md with:
  - the 2x2 table: both-correct / tool-rescued / tool-hurt / both-wrong
  - % tasks solvable without tools  ( B-correct / all )
  - % of A-correct, tool-using answers that survive C  (tool-dependence)

Usage:
  python causal_2x2.py --run-a .../results/baseline --run-b .../results/tool_free \
      --run-c .../results/corrupt --dataset .../gta_dataset/dataset.json \
      --out causal_report.md
Run inside the opencompass env (needs sentence_transformers).
"""
import argparse
import glob
import gzip
import json
import os.path as osp


def latest_ts_dir(run_dir):
    cands = [d for d in glob.glob(osp.join(run_dir, "*")) if osp.isdir(d)
             and osp.basename(d)[:8].isdigit()]
    return max(cands) if cands else None


def load_predictions(run_dir, dataset_abbr="gta_bench_end"):
    """Return {idx(int): sample_record} merged across shards."""
    ts = latest_ts_dir(run_dir)
    assert ts, f"no opencompass output under {run_dir}"
    preds = {}
    for f in sorted(glob.glob(osp.join(ts, "predictions", "*", f"{dataset_abbr}*.json"))):
        data = json.load(open(f))
        for k, v in data.items():
            preds[int(k)] = v
    assert preds, f"no predictions for {dataset_abbr} under {ts}"
    return preds


def final_answer(sample_rec):
    """Mirror GTABenchEvaluator 'every': answer = last step of first round."""
    pred = sample_rec.get("prediction")
    if not pred:
        return None
    rounds = pred if isinstance(pred[0], list) else [pred]
    last = rounds[0][-1] if rounds[0] else None
    if not isinstance(last, dict) or "tool_calls" in last:
        return None
    if last.get("role") != "assistant":
        return None
    return last.get("content")


def used_tools(sample_rec):
    pred = sample_rec.get("prediction") or []
    rounds = pred if (pred and isinstance(pred[0], list)) else [pred]
    tools = []
    for step in (rounds[0] or []):
        if isinstance(step, dict) and "tool_calls" in step:
            tools.append(step["tool_calls"][0]["function"]["name"])
    return tools


class Scorer:
    def __init__(self, sim_threshold):
        self.sim_threshold = sim_threshold
        self._st = None

    def _sim(self, pred, refs):
        if self._st is None:
            from sentence_transformers import SentenceTransformer, util
            import numpy as np
            self._st = SentenceTransformer("all-mpnet-base-v2")
            self._util, self._np = util, np
        pe = self._st.encode(pred, convert_to_tensor=True)
        best = 0.0
        for s in refs:
            ge = self._st.encode(s, convert_to_tensor=True)
            best = max(best, float(self._np.maximum(
                self._util.cos_sim(pe, ge).cpu().numpy(), 0)[0][0]))
        return best

    def correct(self, pred, ref):
        """ref: dict(whitelist/blacklist) -> exact bool; list -> sim >= thr;
        None/empty (imggen-only tasks) -> None (excluded)."""
        import re
        if pred is None or not ref:
            return None if not ref else False
        if isinstance(ref, dict):
            count = 0
            for aliases in ref["whitelist"]:
                pat = r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b"
                if re.search(pat, pred, re.IGNORECASE):
                    count += 1
            if ref.get("blacklist"):
                bk = r"\b(?:" + "|".join(
                    re.escape(a) for al in ref["blacklist"] for a in al) + r")\b"
                return count == len(ref["whitelist"]) and not re.search(
                    bk, pred, re.IGNORECASE)
            return count == len(ref["whitelist"])
        return self._sim(pred, ref) >= self.sim_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="passthrough run dir")
    ap.add_argument("--run-b", required=True, help="tool_free run dir")
    ap.add_argument("--run-c", help="corrupt_output run dir (optional)")
    ap.add_argument("--dataset", required=True, help="gta dataset.json (for refs)")
    ap.add_argument("--sim-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="causal_report.md")
    args = ap.parse_args()

    data = json.load(open(args.dataset))
    refs = {int(k): v["gt_answer"] for k, v in data.items()}

    runs = {"A": load_predictions(args.run_a), "B": load_predictions(args.run_b)}
    if args.run_c:
        runs["C"] = load_predictions(args.run_c)

    scorer = Scorer(args.sim_threshold)
    per_sample = {}
    for idx, ref in sorted(refs.items()):
        rec = {"ref_kind": "objective" if isinstance(ref, dict)
               else ("subjective" if ref else "imggen")}
        for rname, preds in runs.items():
            if idx not in preds:
                rec[rname] = None
                continue
            ans = final_answer(preds[idx])
            rec[rname] = scorer.correct(ans, ref)
            rec[f"{rname}_tools"] = used_tools(preds[idx])
        per_sample[idx] = rec

    scored = {i: r for i, r in per_sample.items()
              if r["A"] is not None and r["B"] is not None}
    n = len(scored)
    both = sum(1 for r in scored.values() if r["A"] and r["B"])
    rescued = sum(1 for r in scored.values() if r["A"] and not r["B"])
    hurt = sum(1 for r in scored.values() if not r["A"] and r["B"])
    neither = sum(1 for r in scored.values() if not r["A"] and not r["B"])

    a_correct = {i: r for i, r in scored.items() if r["A"]}
    a_correct_tool_using = {i: r for i, r in a_correct.items() if r.get("A_tools")}
    if "C" in runs:
        survive_c = {i: r for i, r in a_correct_tool_using.items() if r.get("C")}

    lines = []
    lines.append("# GTA-Atomic causal health-check (A=passthrough, B=tool_free"
                 + (", C=corrupt_output)" if args.run_c else ")"))
    lines.append("")
    lines.append(f"Scored samples (answer-type tasks with predictions in both A and B): "
                 f"**{n}** / {len(per_sample)} total; "
                 f"sim-threshold for subjective refs: {args.sim_threshold}")
    lines.append("")
    lines.append("## 2x2 decomposition (A vs B)")
    lines.append("")
    lines.append("| | B correct (no tools) | B wrong (no tools) |")
    lines.append("|---|---|---|")
    lines.append(f"| **A correct (tools)** | both-correct: {both} ({both/n:.1%}) "
                 f"| tool-rescued: {rescued} ({rescued/n:.1%}) |")
    lines.append(f"| **A wrong (tools)** | tool-hurt: {hurt} ({hurt/n:.1%}) "
                 f"| both-wrong: {neither} ({neither/n:.1%}) |")
    lines.append("")
    lines.append("## Headline motivation stats")
    lines.append("")
    b_correct = both + hurt
    lines.append(f"- **% of tasks solvable without tools** (B correct): "
                 f"**{b_correct}/{n} = {b_correct/n:.1%}**")
    if "C" in runs and a_correct_tool_using:
        surv = len(survive_c)
        lines.append(f"- **% of tool-using A-correct answers that survive output "
                     f"corruption (C)** — answers that do NOT actually depend on "
                     f"tool outputs: **{surv}/{len(a_correct_tool_using)} = "
                     f"{surv/len(a_correct_tool_using):.1%}**")
    lines.append(f"- A-correct samples: {len(a_correct)}; of those, "
                 f"{len(a_correct_tool_using)} made >=1 tool call.")
    lines.append("")
    lines.append("## Per-sample table")
    lines.append("")
    lines.append("| idx | ref_kind | A | B |" + (" C |" if args.run_c else ""))
    lines.append("|---|---|---|---|" + ("---|" if args.run_c else ""))
    for i, r in sorted(per_sample.items()):
        row = f"| {i} | {r['ref_kind']} | {r.get('A')} | {r.get('B')} |"
        if args.run_c:
            row += f" {r.get('C')} |"
        lines.append(row)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    json_out = args.out.replace(".md", "_per_sample.json")
    with open(json_out, "w") as f:
        json.dump(per_sample, f, indent=1, default=str)
    print(f"wrote {args.out} and {json_out}")
    print(f"2x2: both={both} rescued={rescued} hurt={hurt} neither={neither} (n={n})")


if __name__ == "__main__":
    main()
