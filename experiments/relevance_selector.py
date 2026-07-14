"""No-GT per-task tool-relevance selector (Phase 1, offline).

Signal: query <-> tool semantic match via all-mpnet-base-v2. For honest-but-
irrelevant tools, relevance is QUERY-dependent (unlike poison, which is tool-
intrinsic input-degeneracy), so query-tool matching is the usable no-GT signal.

Selection rule per task: rank the 14 tools by cos(query, tool_text), keep top-k
(optionally union with tools above an absolute threshold). Evaluate the predicted
set against GTA's GT per-task tools (P/R/F1, all-relevant-kept rate, pool size).
GT is used ONLY to score the selector, never to build the mask.

Writes the predicted per-task keep-lists to predicted_masks.json for the online run.
"""
import json
import os
import statistics as st

import numpy as np
from sentence_transformers import SentenceTransformer, util

DS = os.path.join(os.environ['GTA_BIG'], 'data/gta_dataset')
TOOLMETA = os.path.join(DS, 'toolmeta.json')
OUT = os.path.join(os.environ['GTA_BIG'], 'results/inject_opt/predicted_masks.json')

meta = json.load(open(TOOLMETA))
tools = list(meta.keys())
def tool_text(name):
    d = meta[name]
    desc = d.get('description') if isinstance(d, dict) else d
    return f'{name}. {desc}'

items = list(json.load(open(os.path.join(DS, 'dataset.json'))).values())
queries = [it['dialogs'][0]['content'] for it in items]
gt = [ {t['name'] for t in it['tools']} for it in items ]

model = SentenceTransformer('all-mpnet-base-v2')
tool_emb = model.encode([tool_text(t) for t in tools], convert_to_tensor=True)
q_emb = model.encode(queries, convert_to_tensor=True)
sims = util.cos_sim(q_emb, tool_emb).cpu().numpy()   # [n_task, 14]

def evaluate(select_fn, label):
    P=R=F=allkept=0.0; pool=[]; preds={}
    for i in range(len(items)):
        keep = select_fn(sims[i])
        preds[i] = [tools[j] for j in keep]
        pset = set(preds[i]); g = gt[i]
        inter = len(pset & g)
        prec = inter/len(pset) if pset else 0
        rec  = inter/len(g) if g else 1
        P+=prec; R+=rec; F+=(2*prec*rec/(prec+rec) if prec+rec else 0)
        allkept += 1.0 if g <= pset else 0.0
        pool.append(len(pset))
    n=len(items)
    print(f"{label:<22} P={P/n:.2f} R={R/n:.2f} F1={F/n:.2f} "
          f"all-relevant-kept={allkept/n*100:4.0f}%  avg_pool={st.mean(pool):.1f}")
    return preds

print(f"GT: avg pool {st.mean(len(g) for g in gt):.1f} tools/task; full pool = {len(tools)}\n")
# top-k sweep
for k in [2,3,4,5,6]:
    evaluate(lambda s,k=k: list(np.argsort(-s)[:k]), f"top-{k}")
print()
# threshold sweep (union of top-1 with above-threshold, so never empty)
for th in [0.15,0.20,0.25,0.30]:
    def sel(s, th=th):
        order=np.argsort(-s); keep=[order[0]]
        keep += [j for j in order[1:] if s[j]>=th]
        return keep
    evaluate(sel, f"top1+thr>={th}")

# pick the operating point to save (favor recall while pruning ~half+): top-4
CHOSEN_K = int(os.getenv('SEL_K', '4'))
preds = evaluate(lambda s: list(np.argsort(-s)[:CHOSEN_K]), f"[SAVED] top-{CHOSEN_K}")
json.dump({str(i): p for i,p in preds.items()}, open(OUT,'w'))
print(f"\nsaved predicted per-task masks (top-{CHOSEN_K}) -> {OUT}")
