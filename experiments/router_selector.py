"""No-GT per-task tool selector via an LLM router (Phase 1b, offline eval).

Embedding match failed (recall 0.50, all-kept 21%) because honest relevance needs
task REASONING, not surface similarity. Here a single focused LLM call per task
predicts the needed tools. This is still no-GT (no labels); it just spends a cheap
upfront reasoning step. Any served model can be the router (can differ from the
agent). Scored against GTA GT (used only to score).

Usage: python router_selector.py <router_name> <router_port> [out_suffix]
"""
import json, os, re, sys, urllib.request
import statistics as st

DS = os.path.join(os.environ['GTA_BIG'], 'data/gta_dataset')
meta = json.load(open(os.path.join(DS, 'toolmeta.json')))
tools = list(meta.keys())
def desc(n):
    d = meta[n]; return d.get('description') if isinstance(d, dict) else d
items = list(json.load(open(os.path.join(DS, 'dataset.json'))).values())
gt = [{t['name'] for t in it['tools']} for it in items]

RNAME, RPORT = sys.argv[1], sys.argv[2]
SUF = sys.argv[3] if len(sys.argv) > 3 else RNAME
URL = f'http://127.0.0.1:{RPORT}/v1/chat/completions'
OUT = os.path.join(os.environ['GTA_BIG'], f'results/inject_opt/router_masks_{SUF}.json')

TOOLLIST = '\n'.join(f'- {t}: {desc(t)}' for t in tools)
SYS = ("You are a tool router. Given a user request and a catalog of tools, select "
       "every tool that could PLAUSIBLY help fulfil the request. It is far worse to "
       "OMIT a tool that turns out to be needed than to include an extra one, so err "
       "toward inclusion: return 3 to 6 tool names. Only drop tools that are clearly "
       "unrelated. Answer with ONLY a JSON list of tool names, e.g. "
       "[\"OCR\",\"ImageDescription\",\"Calculator\"].")

def ask(query):
    body = json.dumps({
        "model": RNAME, "temperature": 0.0, "max_tokens": 100,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": f"Request: {query}\n\nTools:\n{TOOLLIST}\n\nNeeded tools (JSON list):"}]
    }).encode()
    r = urllib.request.urlopen(urllib.request.Request(URL, data=body,
        headers={"Content-Type": "application/json"}), timeout=60)
    txt = json.loads(r.read())['choices'][0]['message']['content']
    m = re.search(r'\[.*?\]', txt, re.S)
    names = []
    if m:
        try: names = [x for x in json.loads(m.group(0)) if x in tools]
        except Exception: names = [t for t in tools if re.search(rf'"{re.escape(t)}"', m.group(0))]
    return names or [tools[0]]

preds = {}
P=R=F=allkept=0.0; pool=[]
for i, it in enumerate(items):
    keep = set(ask(it['dialogs'][0]['content']))
    preds[i] = sorted(keep)
    g = gt[i]; inter = len(keep & g)
    prec = inter/len(keep) if keep else 0
    rec = inter/len(g) if g else 1
    P+=prec; R+=rec; F+=(2*prec*rec/(prec+rec) if prec+rec else 0)
    allkept += 1.0 if g <= keep else 0.0
    pool.append(len(keep))
    if (i+1) % 50 == 0: print(f"  ...{i+1}/{len(items)}")
n=len(items)
print(f"\nROUTER={SUF}  P={P/n:.2f} R={R/n:.2f} F1={F/n:.2f} "
      f"all-relevant-kept={allkept/n*100:.0f}%  avg_pool={st.mean(pool):.1f}  (GT avg {st.mean(len(g) for g in gt):.1f}, full 14)")
json.dump({str(i): p for i, p in preds.items()}, open(OUT, 'w'))
print(f"saved -> {OUT}")
