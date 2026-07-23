"""Test sharper answer-confidence signals as label-free guides for tool-space opt:
mean/min token logprob, mean/max per-token entropy, and answer-entity-token logprob.
Config-level (does signal track acc across oracle/full/poison?) + per-task AUC (does
signal predict correctness?)."""
import ast, glob, json, re, statistics as st
B="/datasets/omni_pretraining/gta2/results"; DS="/datasets/omni_pretraining/gta2/data/gta_dataset"
STOP=set("the a an of to in on at for and or is are was were be been it its this that as with by from we you i need answer question final so can will would should".split())

def norm(s):
    s=(s or '').lower(); s=re.sub(r'(final answer:|thought:|```|answer:)','',s); return re.sub(r'[^a-z0-9 ]','',s).strip()
def fa(preds):
    turn=preds[0] if preds and isinstance(preds[0],list) else preds
    for m in reversed(turn or []):
        if isinstance(m,dict) and not m.get('tool_calls') and m.get('role')=='assistant' and m.get('content'): return m['content']
def iscorrect(ans,ref):
    if not isinstance(ref,dict) or not ref.get('whitelist'): return None
    a=ans.lower()
    ok=all(any(re.search(r'\b'+re.escape(x.lower())+r'\b',a) for x in g) for g in ref['whitelist'])
    if ref.get('blacklist') and re.search(r'\b(?:'+'|'.join(re.escape(x) for g in ref['blacklist'] for x in g)+r')\b',a,re.I): ok=False
    return 1.0 if ok else 0.0
def entity_ll(d):
    toks=d.get('tokens') or []; lps=d.get('tok_logprobs') or []
    pairs=[(t,l) for t,l in zip(toks,lps) if t and t.strip().lower() not in STOP and re.search(r'[a-z0-9]',t.strip().lower()) and len(t.strip())>1]
    return min(l for _,l in pairs) if pairs else (min(lps) if lps else None)
def auc(pairs):
    pos=[s for s,l in pairs if l==1]; neg=[s for s,l in pairs if l==0]
    if not pos or not neg: return None
    return sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))

SIGS=['mean_logprob','min_logprob','mean_entropy','max_entropy','entity_ll']
print(f"{'config':<10}{'acc':>6}  "+' '.join(f'{s:>12}' for s in SIGS)+f"{'n':>5}")
cfg_means={s:{} for s in SIGS}; cfg_rows={}
for rid in ['ll_oracle','ll_full','ll_poison']:
    j=sorted(glob.glob(f"{B}/{rid}/*/results/*/gta_bench_end.json"))
    if not j: print(f"{rid}: no result"); continue
    r=json.load(open(j[-1])); acc=r['answer_acc']
    ll=[json.loads(l) for l in open(f"{B}/ll_signal/{rid}.jsonl") if l.strip()]
    nll=[(norm(d['content']),d) for d in ll if d.get('content','').strip()]
    rows=[]
    for pos,d in r['details'].items():
        preds=d['predictions']; preds=ast.literal_eval(preds) if isinstance(preds,str) else preds
        ans=fa(preds); ref=d['references']; ref=json.loads(ref) if isinstance(ref,str) else ref
        cor=iscorrect(ans,ref)
        if not ans or cor is None: continue
        na=norm(ans)[:40]
        hit=next((e for c,e in nll if na and (na in c or c[:40]==na)),None)
        if not hit: continue
        sig={'mean_logprob':hit['mean_logprob'],'min_logprob':hit.get('min_logprob'),
             'mean_entropy':hit.get('mean_entropy'),'max_entropy':hit.get('max_entropy'),
             'entity_ll':entity_ll(hit)}
        rows.append((sig,cor))
    cfg_rows[rid]=rows
    means={s:st.mean([r[0][s] for r in rows if r[0][s] is not None]) if rows else None for s in SIGS}
    for s in SIGS: cfg_means[s][rid]=means[s]
    print(f"{rid:<10}{acc:6.1f}  "+' '.join(f'{(round(means[s],3) if means[s] is not None else 0):>12}' for s in SIGS)+f"{len(rows):>5}")

print("\n=== 逐题 AUC(信号预测正确性, >0.5=有信号) ===")
for rid,rows in cfg_rows.items():
    print(f"{rid}: "+' '.join(f"{s}={auc([(r[0][s],r[1]) for r in rows if r[0][s] is not None])}" for s in SIGS))
# 合并所有配置的逐题
allrows=[r for rows in cfg_rows.values() for r in rows]
print("\n合并所有配置逐题 AUC:")
for s in SIGS:
    a=auc([(r[0][s],r[1]) for r in allrows if r[0][s] is not None]); print(f"  {s}: AUC={round(a,3) if a else '-'} (n={sum(1 for r in allrows if r[0][s] is not None)})")
print("\n=== 配置级: 信号均值随acc(oracle/full/poison)怎么变 ===")
for s in SIGS: print(f"  {s}: "+' '.join(f"{k.split('_')[1]}={round(v,3) if v is not None else '-'}" for k,v in cfg_means[s].items()))
