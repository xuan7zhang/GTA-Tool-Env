"""RaTA-Set-PORTS Stage 1.1 -- structured task abstraction.

One deterministic generation per task turns the question into a requirement
schema. The schema is the query representation every later stage retrieves
with, so its failure modes propagate: the OpRoute run on this same benchmark
showed a decomposer will silently drop a step it believes it can do mentally
(a Celsius-to-Fahrenheit conversion, a clock subtraction), and the families
whose step was dropped scored 0.000 and 0.033 while the rest matched oracle
routing.

The schema therefore defines an operation by *what changes about the quantity*
-- its kind, scale, unit system or representation -- rather than by whether it
is difficult. That is the only edit to the requested schema, and A5 tests it.

Hard constraints, all asserted after generation rather than assumed:
  no tool name appears in the summary
  the gold answer does not appear in the summary
  no family or task id appears in the summary

    GD_TAG=7b GD_MODEL=... python -m tgb.rata.stage1_summaries --n 1200
"""
import argparse
import collections
import json
import os
import re
import statistics as st

from vllm import LLM, SamplingParams

from ..add_chains_v2 import CHAINS
from ..families_v2 import NUISANCE, UNION_TOOLS
from ..tools import IMAGE_TOOLS

MENU = UNION_TOOLS + NUISANCE + IMAGE_TOOLS
OUT = "/datasets/omni_pretraining/gta2/results/taco/tgb2/rata_set_ports"
TGB2 = "/datasets/omni_pretraining/gta2/results/taco/tgb2"

SCHEMA = """Summarise what a task REQUIRES, as structured JSON. You are not \
solving it and you do not know what tools exist.

Fields:
- input_modalities: which of "image", "text" the task supplies.
- required_observations: what has to be read or perceived from the input, \
described generically.
- required_operations: what has to be done to those observations. Define each \
one by WHAT CHANGES about the quantity -- its kind, its scale, its unit \
system, its currency, its representation -- not by how hard it is. If a \
reading has to end up in a different unit system, a different temperature \
scale, a different currency, a different time representation, or has to be \
recovered from a relation it satisfies, that is a required operation and you \
must list it even if you could do the arithmetic in your head.
- output_type: "number" or "text".
- precision_requirement: "exact" or "approximate".
- steps: ordered, each {{"id": n, "function": ..., "depends_on": [ids]}} where \
function is the generic operation class.

Never name a tool, a piece of software, an API or a service. Never write a \
number that appears in the task. Never state an answer.

Example.
Task: The image shows a fuel gauge in litres. Our tank holds 60.00 gallons. \
How many gallons short of full is it?
{{"input_modalities": ["image", "text"],
 "required_observations": ["a numeric quantity printed on a rendered document"],
 "required_operations": ["restate a quantity in a different system of volume measurement",
                         "take the difference between two numeric quantities"],
 "output_type": "number",
 "precision_requirement": "exact",
 "steps": [{{"id": 1, "function": "read a numeric quantity off a rendered document", "depends_on": []}},
           {{"id": 2, "function": "restate a quantity in a different system of volume measurement", "depends_on": [1]}},
           {{"id": 3, "function": "take the difference between two numeric quantities", "depends_on": [2]}}]}}

Now do the same. Reply with JSON only.
Task: {q}"""

FIELDS = ["input_modalities", "required_observations", "required_operations",
          "output_type", "precision_requirement", "steps"]


def summary_text(s, drop=()):
    """The string later stages embed. `drop` implements the A5 ablation."""
    parts = []
    if "input_modality" not in drop:
        parts += [f"input: {x}" for x in s.get("input_modalities", [])]
    if "required_observation" not in drop:
        parts += [f"observe: {x}" for x in s.get("required_observations", [])]
    if "operation" not in drop:
        parts += [f"operation: {x}" for x in s.get("required_operations", [])]
    if "output_type" not in drop:
        parts.append(f"output: {s.get('output_type', 'number')}")
    if "dependency" not in drop:
        for st_ in s.get("steps", []):
            dep = ",".join(str(d) for d in st_.get("depends_on", []))
            parts.append(f"step {st_.get('id')}: {st_.get('function')}"
                         + (f" (after {dep})" if dep else ""))
    return " ; ".join(str(p) for p in parts)


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    for cand in ([m.group(0), m.group(0).rsplit("}", 1)[0] + "}"] if m else []):
        try:
            d = json.loads(cand)
            if "required_operations" in d or "steps" in d:
                return d, True
        except Exception:
            continue
    return None, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    a = ap.parse_args()
    tag = os.environ.get("GD_TAG", "7b")

    tasks_all = json.load(open(os.path.join(TGB2, "tgb_tasks.json")))
    by_fam = collections.defaultdict(list)
    for t in tasks_all:
        by_fam[t["family"]].append(t)
    per = a.n // len(by_fam)
    tasks = [t for f in sorted(by_fam) for t in by_fam[f][:per]]
    # test is unchanged from Stage 0 (k%5>=2) so every baseline stays
    # comparable; dev is carved out of the Stage-0 train half.
    split = {}
    for k, t in enumerate(tasks):
        split[t["id"]] = "test" if k % 5 >= 2 else ("dev" if k % 5 == 1
                                                   else "train")
    print(f"[{tag}] {len(tasks)} tasks  "
          f"{collections.Counter(split.values()).most_common()}")

    llm = LLM(model=os.environ["GD_MODEL"],
              tensor_parallel_size=int(os.environ.get("GD_TP", "1")),
              dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("GD_UTIL", "0.85")),
              max_model_len=4096)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=700, temperature=0)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": SCHEMA.format(q=t["question"])}],
        tokenize=False, add_generation_prompt=True) for t in tasks]
    outs = llm.generate([{"prompt": p} for p in prompts], sp)

    rows, nvalid, leaks = [], 0, collections.Counter()
    lower_tools = {m.lower() for m in MENU}
    for t, o in zip(tasks, outs):
        d, ok = parse(o.outputs[0].text)
        nvalid += ok
        d = d or {"input_modalities": ["image", "text"],
                  "required_observations": ["information shown in the input"],
                  "required_operations": ["produce the requested value"],
                  "output_type": "number", "precision_requirement": "exact",
                  "steps": [{"id": 1, "function": "answer the task",
                             "depends_on": []}]}
        txt = summary_text(d)
        low = txt.lower()
        if any(re.search(rf"\b{re.escape(m)}\b", low) for m in lower_tools):
            leaks["tool_name"] += 1
        if re.search(r"(?<![\d.])" + re.escape(t["gold"].lstrip("$")) +
                     r"(?![\d])", txt.replace(",", "")):
            leaks["gold_answer"] += 1
        if t["family"] in low or t["id"] in low:
            leaks["family_or_id"] += 1
        rows.append({"task_id": t["id"], "family": t["family"],
                     "split": split[t["id"]], "raw_question": t["question"],
                     "structured_summary": d, "summary_text": txt,
                     "valid_json": bool(ok),
                     "n_steps": len(d.get("steps", [])),
                     "raw": o.outputs[0].text[:800]})

    os.makedirs(f"{OUT}/summaries", exist_ok=True)
    for s in ("train", "dev", "test"):
        with open(f"{OUT}/summaries/{s}_{tag}.jsonl", "w") as fh:
            for r in rows:
                if r["split"] == s:
                    fh.write(json.dumps(r) + "\n")
    stats = {
        "tag": tag, "n": len(rows), "valid_json_frac": nvalid / len(rows),
        "mean_steps": st.mean(r["n_steps"] for r in rows),
        "mean_summary_chars": st.mean(len(r["summary_text"]) for r in rows),
        "leakage": dict(leaks),
        "n_unique_operation_strings": len(
            {x for r in rows
             for x in r["structured_summary"].get("required_operations", [])}),
        "ops_by_family": {f: collections.Counter(
            x for r in rows if r["family"] == f
            for x in r["structured_summary"].get("required_operations", [])
        ).most_common(4) for f in sorted(by_fam)},
    }
    json.dump(stats, open(f"{OUT}/summaries/stats_{tag}.json", "w"), indent=1)
    print(f"  valid JSON {stats['valid_json_frac']:.1%}  "
          f"mean steps {stats['mean_steps']:.2f}  "
          f"mean chars {stats['mean_summary_chars']:.0f}")
    print(f"  leakage: {dict(leaks) or 'none'}")
    print(f"  unique required_operation strings: "
          f"{stats['n_unique_operation_strings']}")
    for f, c in stats["ops_by_family"].items():
        print(f"    {f:<22} " + " | ".join(f"{k[:42]}x{v}" for k, v in c[:2]))


if __name__ == "__main__":
    main()
