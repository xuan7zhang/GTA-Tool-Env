"""Which number is right: Stage 0's 0.439/0.576 or qll_cov's 0.553/0.679?

Same masks, same task ids, same code path -- rebuilt in a fresh process and
generated twice, once as a single batch and once split in two, so a numeric
batching effect would show up as a difference between the two.
"""
import collections, json, os, random, statistics as st, tempfile, zlib
from vllm import LLM, SamplingParams
from .. import scenes, tools_v2
from ..add_chains_v2 import CHAINS
from ..families_v2 import UNION_TOOLS
from ..generate import TPL, context
from ..score_dl import correct
from ..tools import run_chain
tools_v2.register()
TGB2 = "/datasets/omni_pretraining/gta2/results/taco/tgb2"

T = json.load(open(f"{TGB2}/tgb_tasks.json"))
by = collections.defaultdict(list)
for t in T:
    by[t["family"]].append(t)
tasks = [t for f in sorted(by) for t in by[f][:150]]
test = [t for k, t in enumerate(tasks) if k % 5 >= 2]
tmp = tempfile.mkdtemp()
for t in tasks:
    t["boxes"] = scenes.render(t["scene"], os.path.join(tmp, "s.png"))

def build(t, S):
    rng = random.Random(zlib.crc32(f"{t['id']}/{'+'.join(sorted(S))}".encode()))
    outs, _ = run_chain(t, t["scene"], t["boxes"], set(S), rng, corrupt_tools=())
    return TPL.format(q=t["question"], o=context(outs, [x["tool"] for x in t["plan"]]))

llm = LLM(model=os.environ["GD_MODEL"], dtype="bfloat16",
          gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.80")),
          max_model_len=4096)
sp = SamplingParams(max_tokens=64, temperature=0)
U = sorted(UNION_TOOLS)
for nm, mk in [("union9", lambda t: U), ("gt", lambda t: sorted(CHAINS[t["family"]]))]:
    ps = [build(t, mk(t)) for t in test]
    print(f"\n--- {nm}: sample prompt tail ---\n{ps[0][-260:]}")
    o1 = llm.generate([{"prompt": p} for p in ps], sp)
    a1 = st.mean(correct(o.outputs[0].text, t["gold"]) for o, t in zip(o1, test))
    h1 = llm.generate([{"prompt": p} for p in ps[:360]], sp)
    h2 = llm.generate([{"prompt": p} for p in ps[360:]], sp)
    a2 = st.mean(correct(o.outputs[0].text, t["gold"])
                 for o, t in zip(list(h1) + list(h2), test))
    print(f"{nm}: one-batch {a1:.4f}   split-batch {a2:.4f}")
    print(f"  first gen: {o1[0].outputs[0].text[:70]!r}  gold={test[0]['gold']}")
