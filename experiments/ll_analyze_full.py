"""Full signal battery analysis. For each entropy/likelihood variant: per-task AUC
(predict correctness) and config-level ranking (does mean-signal rank configs like
true accuracy?). Configs span oracle(high)/full(low)/corrupt(lowest)."""
import ast,glob,json,re,math,statistics as st
from collections import Counter
B="/datasets/omni_pretraining/gta2/results"
STOP=set("the a an of to in on at for and or is are was were be been it its this that as with by from we you i need answer question final so can will would should there their".split())
def norm(s):
    s=(s or '').lower(); s=re.sub(r'(final answer:|thought:|```|answer:)','',s); return re.sub(r'[^a-z0-9 ]','',s).strip()
def fa(preds):
    turn=preds[0] if preds and isinstance(preds[0],list) else preds
    for m in reversed(turn or []):
        if isinstance(m,dict) and not m.get('tool_calls') and m.get('role')=='assistant' and m.get('content'): return m['content']
def iscorrect(ans,ref):
    if not isinstance(ref,dict) or not ref.get('whitelist'): return None
    a=ans.lower(); return 1.0 if all(any(re.search(r'\b'+re.escape(x.lower())+r'\b',a) for x in g) for g in ref['whitelist']) else 0.0
def auc(pairs):
    pos=[s for s,l in pairs if l==1]; neg=[s for s,l in pairs if l==0]
    if not pos or not neg: return None
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
def details(rid):
    j=sorted(glob.glob(f"{B}/{rid}/*/results/*/gta_bench_end.json")); 
    return (json.load(open(j[-1])) if j else None)

def tok_sigs(d):
    lp=d.get('tok_logprobs') or []; en=d.get('tok_entropy') or []; mg=d.get('tok_margin') or []; tk=d.get('tokens') or []
    if not lp: return None
    ent=[l for t,l in zip(tk,lp) if t and t.strip().lower() not in STOP and re.search(r'[a-z0-9]',(t or '').lower()) and len(t.strip())>1]
    return {
     'mean_lp':sum(lp)/len(lp),'min_lp':min(lp),'sum_lp':sum(lp),'ppl':math.exp(-sum(lp)/len(lp)),
     'entity_min_lp':(min(ent) if ent else min(lp)),
     'frac_lowconf':sum(1 for x in lp if x<-0.5)/len(lp),
     'mean_ent':(sum(en)/len(en) if en else 0),'max_ent':(max(en) if en else 0),'min_ent':(min(en) if en else 0),
     'mean_margin':(sum(mg)/len(mg) if mg else 0),'min_margin':(min(mg) if mg else 0),
    }
TOKSIG=['mean_lp','min_lp','sum_lp','ppl','entity_min_lp','frac_lowconf','mean_ent','max_ent','min_ent','mean_margin','min_margin']

# ---- correctness label + token signals from greedy runs ----
print("========== TOKEN-LEVEL (greedy) ==========")
cfgs=['oracle','full','corrupt']; tok_rows={c:[] for c in cfgs}; acc={}
for c in cfgs:
    r=details(f"ll7b_{c}"); acc[c]=r['answer_acc'] if r else None
    if not r: continue
    ll=[json.loads(l) for l in open(f"{B}/ll_signal/ll7b_{c}.jsonl") if l.strip()]
    nll=[(norm(x['content']),x) for x in ll if x.get('content','').strip()]
    for pos,d in r['details'].items():
        preds=d['predictions']; preds=ast.literal_eval(preds) if isinstance(preds,str) else preds
        ans=fa(preds); ref=d['references']; ref=json.loads(ref) if isinstance(ref,str) else ref
        cor=iscorrect(ans,ref)
        if not ans or cor is None: continue
        na=norm(ans)[:40]; hit=next((e for cc,e in nll if na and (na in cc or cc[:40]==na)),None)
        if not hit: continue
        s=tok_sigs(hit)
        if s: tok_rows[c].append((s,cor,pos))
print("true acc:", {c:round(acc[c],1) for c in cfgs if acc[c] is not None})
allrows=[(s,l) for c in cfgs for s,l,_ in tok_rows[c]]
print(f"\n{'signal':<14}{'AUC':>7}{'|AUC-.5|':>9}   config-means(oracle/full/corrupt)  rank-match")
for sg in TOKSIG:
    a=auc([(s[sg],l) for s,l in allrows if s[sg] is not None])
    means={c:(st.mean([s[sg] for s,_,_ in tok_rows[c]]) if tok_rows[c] else None) for c in cfgs}
    oa=[c for c in cfgs if acc[c] is not None]; racc=sorted(oa,key=lambda c:-acc[c])
    rsig=sorted([c for c in cfgs if means[c] is not None],key=lambda c:-means[c])
    match='?'
    if len(rsig)==len(racc): match='✓' if rsig==racc else ('✓(inv)' if rsig==racc[::-1] else '✗')
    mstr='/'.join(f"{means[c]:.3f}" if means[c] is not None else '-' for c in cfgs)
    print(f"{sg:<14}{(round(a,3) if a else 0):>7}{(round(abs(a-.5),3) if a else 0):>9}   {mstr:<30} {match}")

# ---- semantic entropy from sampled runs ----
print("\n========== ANSWER-LEVEL SEMANTIC (temp 0.8, K=5) ==========")
def answers(rid):
    r=details(rid); out={}
    if not r: return out
    for pos,d in r['details'].items():
        preds=d['predictions']; preds=ast.literal_eval(preds) if isinstance(preds,str) else preds
        a=fa(preds)
        if a: out[pos]=norm(a)[:60]
    return out
sem_rows={c:[] for c in cfgs}
for c in cfgs:
    ks=[answers(f"sc_{c}_k{k}") for k in range(1,6)]; ks=[a for a in ks if a]
    if len(ks)<2: print(f"{c}: <2 samples"); continue
    # correctness label from greedy run
    lab={pos:cor for s,cor,pos in tok_rows[c]}
    for pos in ks[0]:
        ans=[a.get(pos) for a in ks if a.get(pos)]
        if len(ans)<2: continue
        cnt=Counter(ans); K=len(ans)
        agree=cnt.most_common(1)[0][1]/K
        sement=-sum((v/K)*math.log(v/K) for v in cnt.values())
        sem_rows[c].append(({'agreement':agree,'n_distinct':len(cnt),'sem_entropy':sement}, lab.get(pos), pos))
SEMSIG=['agreement','n_distinct','sem_entropy']
alls=[(s,l) for c in cfgs for s,l,_ in sem_rows[c] if l is not None]
print(f"{'signal':<14}{'AUC':>7}{'|AUC-.5|':>9}   config-means(oracle/full/corrupt)  rank-match")
for sg in SEMSIG:
    a=auc([(s[sg],l) for s,l in alls if s[sg] is not None and l is not None])
    means={c:(st.mean([s[sg] for s,_,_ in sem_rows[c]]) if sem_rows[c] else None) for c in cfgs}
    oa=[c for c in cfgs if acc[c] is not None]; racc=sorted(oa,key=lambda c:-acc[c])
    rsig=sorted([c for c in cfgs if means[c] is not None],key=lambda c:-means[c])
    match='?'
    if len(rsig)==len(racc): match='✓' if rsig==racc else ('✓(inv)' if rsig==racc[::-1] else '✗')
    mstr='/'.join(f"{means[c]:.3f}" if means[c] is not None else '-' for c in cfgs)
    print(f"{sg:<14}{(round(a,3) if a else 0):>7}{(round(abs(a-.5),3) if a else 0):>9}   {mstr:<30} {match}")
print("\n注: |AUC-.5| = 判别力(越大越有信号); rank-match ✓ = 信号均值排序=真acc排序(能选配置)")
