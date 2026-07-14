import json,os,sys,statistics as st
BIG=os.environ['GTA_BIG']; DS=os.path.join(BIG,'data/gta_dataset')
router=json.load(open(sys.argv[1])); out=sys.argv[2]
emb=json.load(open(os.path.join(BIG,'results/inject_opt/predicted_masks.json')))
gt=[{t['name'] for t in it['tools']} for it in json.load(open(os.path.join(DS,'dataset.json'))).values()]
COMMON={'OCR','ImageDescription'}
res={}; R=ak=0.0; pool=[]
for i in range(len(gt)):
    s=set(router.get(str(i),[]))|set(emb[str(i)])|COMMON; res[str(i)]=sorted(s)
    g=gt[i]; R+=len(s&g)/len(g) if g else 1; ak+=1.0 if g<=s else 0; pool.append(len(s))
n=len(gt); json.dump(res,open(out,'w'))
print(f"hybrid -> {out}  R={R/n:.2f} all-kept={ak/n*100:.0f}% pool={st.mean(pool):.1f}")
