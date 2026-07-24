import ast,glob,json,statistics as st
B="/datasets/omni_pretraining/gta2/results"; DS="/datasets/omni_pretraining/gta2/data/gta_dataset"
items=list(json.load(open(f"{DS}/dataset.json")).values()); GT=[{t['name'] for t in it['tools']} for it in items]
MDIR=f"{B}/inject_opt/clutter_masks"
def acc(rid):
    j=sorted(glob.glob(f"{B}/{rid}/*/results/*/gta_bench_end.json")); return json.load(open(j[-1]))['answer_acc'] if j else None
def pool(cfg):
    d=json.load(open(f"{MDIR}/{cfg}.json")); return st.mean(len(v) for v in d.values())
def selacc(rid):  # 调用命中GT相关工具的比例
    j=sorted(glob.glob(f"{B}/{rid}/*/results/*/gta_bench_end.json"))
    if not j: return None
    det=json.load(open(j[-1]))['details']; hit=tot=0
    for pos,d in det.items():
        g=GT[int(pos)]; p=d['predictions']; p=ast.literal_eval(p) if isinstance(p,str) else p
        for turn in p:
            if not isinstance(turn,list): turn=[turn]
            for m in turn:
                if isinstance(m,dict) and m.get('tool_calls'): tot+=1; hit+=1 if m['tool_calls'][0]['function']['name'] in g else 0
    return hit/tot*100 if tot else None
print("=== 剂量-响应(pool size -> acc, 选择精度) ===")
print(f"{'config':<10}{'pool':>6}{'acc':>7}{'sel_acc%':>9}")
def m(cfgs):
    a=[acc(f"cl_{c}") for c in cfgs]; a=[x for x in a if x is not None]; return st.mean(a) if a else None
for label,cfgs in [('oracle',['oracle']),('dose2',['dose2_s1','dose2_s2']),('dose4',['dose4_s1','dose4_s2']),
                   ('dose6',['dose6_s1','dose6_s2']),('dose8',['dose8_s1','dose8_s2']),('full',['full'])]:
    a=m(cfgs); p=pool(cfgs[0].replace('_s1','_s1') if 'dose' in cfgs[0] else cfgs[0]) if 'dose' not in cfgs[0] else pool(cfgs[0])
    sa=selacc(f"cl_{cfgs[0]}")
    print(f"{label:<10}{p:6.1f}{(round(a,1) if a else 0):>7}{(round(sa,0) if sa else 0):>9}")
print("\n=== 类别归因(oracle加某类无关工具的伤害/每工具) ===")
o=acc("cl_oracle")
print(f"oracle acc={round(o,1) if o else '-'} (pool 2.3)")
print(f"{'category':<14}{'pool':>6}{'acc':>7}{'Δvs_oracle':>11}{'伤害/工具':>10}")
for c in ['cat_perception','cat_logic','cat_operation','cat_creativity']:
    a=acc(f"cl_{c}"); p=pool(c)
    if a is not None and o is not None:
        added=p-2.3; dmg=(a-o); per=dmg/added if added>0 else 0
        print(f"{c[4:]:<14}{p:6.1f}{a:7.1f}{dmg:+11.1f}{per:+10.2f}")
