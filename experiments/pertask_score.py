"""Exactly reproduce GTA's end-mode per-task answer_acc so we can break the
RegionRead result down by task type. Objective (dict ref) -> iscorrect (word-
boundary whitelist regex on the LAST trajectory message); subjective (list ref)
-> simscore (all-mpnet-base-v2 cosine). Mirrors GTABenchEvaluator.score(mode='every').

Usage: python pertask_score.py <run_glob_prefix> [<run_glob_prefix> ...]
  e.g. python pertask_score.py rr_ q3b_ q14b_ l3b_ l8b_
Prints per-task mean score (over seeds) per condition, and the uniq/select/other
strata. Verifies the reproduced mean matches the stored aggregate answer_acc.
"""
import ast
import glob
import json
import re
import sys
import statistics as st

import numpy as np
from sentence_transformers import SentenceTransformer, util

BASE = "/datasets/omni_pretraining/gta2/results"
IDS = [9, 10, 11, 40, 84, 164, 165, 166]
TAG = {9: 'select', 40: 'select', 84: 'select', 11: 'uniq', 164: 'uniq',
       165: 'uniq', 10: 'other', 166: 'other'}
QSHORT = {9: 'middle dog breed', 10: 'front man holding', 11: 'purple fruit taste',
          40: 'right fruit region', 84: 'males-females lions', 164: 'white cat eyes',
          165: 'white eggs mood', 166: 'largest balloon color'}

_MODEL = None


def sim(pred, refs):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer('all-mpnet-base-v2')
    pe = _MODEL.encode(pred, convert_to_tensor=True)
    best = 0.0
    for s in refs:
        ge = _MODEL.encode(s, convert_to_tensor=True)
        best = max(best, float(np.maximum(util.cos_sim(pe, ge).cpu().numpy(), 0)[0][0]))
    return best


def iscorrect(pred, ref):
    count = 0
    for aliases in ref['whitelist']:
        pat = r'\b(?:' + '|'.join(re.escape(a) for a in aliases) + r')\b'
        if re.search(pat, pred, re.IGNORECASE):
            count += 1
    if not ref['blacklist']:
        return 1.0 if count == len(ref['whitelist']) else 0.0
    pat_bk = r'\b(?:' + '|'.join(re.escape(a) for al in ref['blacklist'] for a in al) + r')\b'
    return 1.0 if (count == len(ref['whitelist']) and not re.search(pat_bk, pred, re.IGNORECASE)) else 0.0


def last_answer(preds):
    """get_response_type(preds[0][-1]) — score only if it's a plain answer."""
    turn = preds[0] if preds and isinstance(preds[0], list) else preds
    if not turn:
        return None
    m = turn[-1]
    if not isinstance(m, dict):
        return None
    if 'tool_calls' in m:
        return None            # ends on a tool call -> not an answer -> 0
    if m.get('role') == 'assistant':
        return m.get('content') or ''
    return None


def score_run(rid):
    rj = sorted(glob.glob(f"{BASE}/{rid}/*/results/*/gta_bench_end.json"))
    if not rj:
        return None, None
    r = json.load(open(rj[-1]))
    per = {}
    for pos, i in enumerate(IDS):
        d = r['details'][str(pos)]
        preds = d['predictions']
        if isinstance(preds, str):
            preds = ast.literal_eval(preds)
        ref = d['references']
        ref = json.loads(ref) if isinstance(ref, str) else ref
        if not ref:
            per[i] = None            # empty ref (e.g. image-gen task) -> excluded from answer_acc
            continue
        ans = last_answer(preds)
        if ans is None:
            per[i] = 0.0
        elif isinstance(ref, dict):
            per[i] = iscorrect(ans, ref)
        else:
            per[i] = sim(ans, ref)
    return per, r.get('answer_acc')


def main(prefixes):
    for pfx in prefixes:
        conds = {}
        for cond in ['baseline', 'composed']:
            seeds = sorted(glob.glob(f"{BASE}/{pfx}{cond}_s*"))
            per_acc = {i: [] for i in IDS}
            repro, stored = [], []
            for sd in seeds:
                rid = sd.split('/')[-1]
                per, agg = score_run(rid)
                if per is None:
                    continue
                scored = [per[i] for i in IDS if per[i] is not None]  # non-empty-ref tasks
                for i in IDS:
                    if per[i] is not None:
                        per_acc[i].append(per[i])
                repro.append(st.mean(scored) * 100)
                stored.append(agg)
            conds[cond] = per_acc
            if repro:
                print(f"[{pfx}{cond}] n_seed={len(repro)} repro_mean={st.mean(repro):.1f} "
                      f"stored_mean={st.mean([s for s in stored if s is not None]):.1f}")
        if not conds.get('baseline') or not conds['baseline'][IDS[0]]:
            print(f"  (no runs for {pfx})"); continue
        b, c = conds['baseline'], conds['composed']
        print(f"  {'id':>4} {'type':<7} {'task':<18} base comp   Δ")
        for i in IDS:
            bv = st.mean(b[i]) if b[i] else float('nan')
            cv = st.mean(c[i]) if c[i] else float('nan')
            print(f"  {i:>4} {TAG[i]:<7} {QSHORT[i]:<18} {bv:.2f} {cv:.2f} {cv-bv:+.2f}")
        for grp in ['uniq', 'select', 'other']:
            gi = [i for i in IDS if TAG[i] == grp]
            bm = st.mean([st.mean(b[i]) for i in gi if b[i]])
            cm = st.mean([st.mean(c[i]) for i in gi if c[i]])
            print(f"  [{grp}] base {bm:.2f} -> comp {cm:.2f}  Δ{cm-bm:+.2f}")
        print()


if __name__ == '__main__':
    main(sys.argv[1:] or ['rr_'])
