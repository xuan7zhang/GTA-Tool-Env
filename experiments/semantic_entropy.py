import ast,glob,json,re,math,statistics as st
from sentence_transformers import SentenceTransformer,util
from collections import Counter
B="/datasets/omni_pretraining/gta2/results"
def fa(preds):
    turn=preds[0] if preds and isinstance(preds[0],list) else preds
    for m in reversed(turn or []):
        if isinstance(m,dict) and not m.get("tool_calls") and m.get("role")=="assistant" and m.get("content"): return m["content"]
def iscorrect(ans,ref):
    if not isinstance(ref,dict) or not ref.get("whitelist"): return None
    a=ans.lower(); return 1.0 if all(any(re.search(r"\b"+re.escape(x.lower())+r"\b",a) for x in g) for g in ref["whitelist"]) else 0.0
def det(rid):
    j=sorted(glob.glob(f"{B}/{rid}/*/results/*/gta_bench_end.json")); return json.load(open(j[-1])) if j else None
def answers(rid):
    r=det(rid); out={}
    if r:
        for pos,d in r["details"].items():
            p=d["predictions"]; p=ast.literal_eval(p) if isinstance(p,str) else p; a=fa(p)
            if a: out[pos]=a[:200]
    return out
def auc(pairs):
    pos=[s for s,l in pairs if l==1]; neg=[s for s,l in pairs if l==0]
    if not pos or not neg: return None
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
M=SentenceTransformer("all-mpnet-base-v2")
cfgs=["oracle","full","corrupt"]; lab={}
for c in cfgs:
    r=det(f"ll7b_{c}")
    if r:
        for pos,d in r["details"].items():
            p=d["predictions"]; p=ast.literal_eval(p) if isinstance(p,str) else p; a=fa(p)
            rf=d["references"]; rf=json.loads(rf) if isinstance(rf,str) else rf
            if a and iscorrect(a,rf) is not None: lab[(c,pos)]=iscorrect(a,rf)
def cluster_ent(ans,thr=0.75):
    emb=M.encode(ans); K=len(ans); used=[-1]*K; nc=0
    import numpy as np
    for i in range(K):
        if used[i]>=0: continue
        used[i]=nc
        for j in range(i+1,K):
            if used[j]<0 and float(util.cos_sim(emb[i],emb[j]))>=thr: used[j]=nc
        nc+=1
    cnt=Counter(used); return -sum((v/K)*math.log(v/K) for v in cnt.values()), cnt.most_common(1)[0][1]/K
rows_e=[]; rows_a=[]; means={c:[] for c in cfgs}
for c in cfgs:
    ks=[answers(f"sc_{c}_k{k}") for k in range(1,6)]; ks=[a for a in ks if a]
    for pos in (ks[0] if ks else {}):
        al=[a[pos] for a in ks if a.get(pos)]
        if len(al)<3: continue
        ent,agree=cluster_ent(al); means[c].append(ent)
        if (c,pos) in lab: rows_e.append((ent,lab[(c,pos)])); rows_a.append((agree,lab[(c,pos)]))
print("== 真·语义熵(mpnet聚类 thr=0.75) ==")
print("配置级 语义熵均值:", {c:round(st.mean(means[c]),3) for c in cfgs if means[c]}, "(好配置oracle应低熵)")
ae=auc(rows_e); aa=auc(rows_a)
print(f"逐题AUC 语义熵->对错: {round(ae,3) if ae else None} |AUC-.5|={round(abs(ae-.5),3) if ae else None} (n={len(rows_e)})")
print(f"逐题AUC 一致性->对错: {round(aa,3) if aa else None} |AUC-.5|={round(abs(aa-.5),3) if aa else None}")
racc=['oracle','corrupt','full']; rsig=sorted([c for c in cfgs if means[c]],key=lambda c:st.mean(means[c]))
print(f"配置级排序: 真acc={racc}  语义熵升序={rsig}  {'✓好配置低熵' if rsig==racc else '✗'}")
