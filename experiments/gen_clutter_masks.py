"""Build per-task masks (clean pool, no injection) for the clutter-structure battery:
  oracle       : GT tools only
  dose_k_sN    : GT + k random non-GT distractors (seed N)
  cat_<C>      : GT + all non-GT tools of category C
  full         : all 14
Written to results/inject_opt/clutter_masks/<config>.json (task_idx -> tool list)."""
import json,os,random
BIG=os.environ['GTA_BIG']; DS=f"{BIG}/data/gta_dataset"
TOOLS=list(json.load(open(f"{DS}/toolmeta.json")).keys())
items=list(json.load(open(f"{DS}/dataset.json")).values())
GT=[sorted({t['name'] for t in it['tools']}) for it in items]
CAT={'perception':['OCR','ImageDescription','RegionAttributeDescription','TextToBbox'],
     'logic':['Calculator','Solver','Plot','MathOCR','CountGivenObject'],
     'operation':['DrawBox','AddText','GoogleSearch'],
     'creativity':['TextToImage','ImageStylization']}
OUT=f"{BIG}/results/inject_opt/clutter_masks"; os.makedirs(OUT,exist_ok=True)
def save(name,fn):
    json.dump({str(i):sorted(set(fn(i))) for i in range(len(items))},open(f"{OUT}/{name}.json","w"))
save("oracle",lambda i:GT[i])
save("full",lambda i:TOOLS)
for k in [2,4,6,8]:
    for s in [1,2]:
        def fn(i,k=k,s=s):
            rng=random.Random(1000*s+i); non=[t for t in TOOLS if t not in GT[i]]
            return GT[i]+rng.sample(non,min(k,len(non)))
        save(f"dose{k}_s{s}",fn)
for c,tools in CAT.items():
    save(f"cat_{c}",lambda i,tools=tools:GT[i]+[t for t in tools if t not in GT[i]])
import statistics as st
print("生成的配置:", sorted(os.listdir(OUT)))
for name in ['oracle','dose2_s1','dose4_s1','dose8_s1','cat_perception','cat_logic','full']:
    d=json.load(open(f"{OUT}/{name}.json")); print(f"  {name}: avg pool {st.mean(len(v) for v in d.values()):.1f}")
