"""Phase 0 — harness and leakage validation (spec §5).

Stop-the-experiment gate. Writes results/taco/phase0_audit/*.json and
results/taco/PHASE0_AUDIT.md. Exits non-zero if any invariant fails.

Checks
  A1 all expected tasks load (229) and the dataset hash matches the split
  A2 all expected tools are registered on the live tool server (14)
  A3 transformations change only the intended output properties
     (19k-case offline suite: preservation, no invented facts, F0 identity,
      monotone length ladder, image tools untouched, never raises)
  A4 no answer labels are injected into prompts (gold strings absent from the
     agent-visible prompt, and absent from every transformed tool output that
     did not already contain them natively)
  A5 a forced global mask is identical across queries (menu invariance)
  A6 query-level conditions are labelled QUERY-LEVEL TOOL SELECTION
  A7 held-out tasks are inaccessible: the runner's leakage guard fires
  A8 transformed tool outputs preserve factual content on live tool returns

Usage: python taco/phase0_audit.py --lane 1 --model qwen2.5-7b-instruct
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taco import transform as T          # noqa: E402
from taco import runner as R             # noqa: E402

BIG = "/datasets/omni_pretraining/gta2"
TACO = f"{BIG}/results/taco"
OUT = f"{TACO}/phase0_audit"


def check(results, name, ok, detail=""):
    results.append(dict(check=name, ok=bool(ok), detail=str(detail)[:600]))
    print(("  PASS " if ok else "  FAIL ") + name + ("  " + str(detail)[:160] if detail else ""))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", type=int, default=1)
    ap.add_argument("--model", default="qwen2.5-7b-instruct")
    ap.add_argument("--skip-live", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = []
    ports = R.LANE_PORTS[a.lane]

    print("A1 dataset + split")
    raw = open(R.DS_PATH, "rb").read()
    ds = json.loads(raw)
    splits = json.load(open(f"{TACO}/splits.json"))
    check(res, "A1.tasks_loaded_229", len(ds) == 229, f"n={len(ds)}")
    check(res, "A1.dataset_hash_matches_split",
          hashlib.sha256(raw).hexdigest() == splits["dataset_sha256"])
    check(res, "A1.split_partitions_dataset",
          sorted(splits["calibration_ids"] + splits["heldout_ids"]) == list(range(229))
          and not set(splits["calibration_ids"]) & set(splits["heldout_ids"]),
          f"{splits['calibration_size']}/{splits['heldout_size']}")

    print("A2 tool registry (live)")
    expected = sorted(json.load(open(R.TOOLMETA)))
    live = None
    if not a.skip_live:
        try:
            spec = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:{ports['proxy']}/openapi.json", timeout=60).read())
            live = sorted(p.strip("/") for p in spec.get("paths", {}))
        except Exception as e:
            live = f"ERROR {e}"
    check(res, "A2.tools_registered", isinstance(live, list) and set(expected) <= set(live),
          f"expected {len(expected)}, live {len(live) if isinstance(live, list) else live}")

    print("A3 transform suite (offline)")
    rc = subprocess.run([sys.executable, f"{os.path.dirname(os.path.abspath(__file__))}"
                         "/test_transform.py"], capture_output=True, text=True)
    tv = json.load(open(f"{OUT}/transform_validation.json"))["summary"]
    check(res, "A3.transform_suite", tv["n_failures"] == 0,
          f"{tv['n_cases']} cases, {tv['n_failures']} failures")

    print("A4 no gold-answer injection")
    # (i) transformed outputs must not introduce a gold string that the native
    #     output did not already contain.
    fx = json.load(open(f"{TACO}/interventions/tool_output_fixtures.json"))
    golds = []
    for i in range(229):
        g = ds[str(i)].get("gt_answer")
        if isinstance(g, dict):
            golds += [al for grp in g["whitelist"] for al in grp]
    golds = sorted({g for g in golds if len(g) > 3})[:400]
    injected = []
    for tool, samples in fx.items():
        if tool not in T.TOOL_VOCAB:
            continue
        for s in samples[:10]:
            for fmt in T.FORMATS:
                out, _ = T.transform(tool, s, {"format": fmt, "length": {"level": "L5",
                                                                        "mechanism": "irrelevant"}})
                for g in golds:
                    if re.search(r"\b" + re.escape(g) + r"\b", out, re.I) and \
                       not re.search(r"\b" + re.escape(g) + r"\b", s, re.I):
                        injected.append((tool, fmt, g))
    check(res, "A4.no_gold_injected_by_transform", not injected, injected[:5])
    # (ii) the filler text itself contains no gold string
    filler = " ".join(T.FILLER_SENTENCES).format(tool="OCR")
    bad_filler = [g for g in golds if re.search(r"\b" + re.escape(g) + r"\b", filler, re.I)]
    check(res, "A4.filler_carries_no_gold", not bad_filler, bad_filler[:5])

    print("A5/A8 live smoke: menu invariance + live factual preservation")
    live_ok, menu_detail, pres_detail = None, "", ""
    if not a.skip_live:
        ids = [i for i in splits["calibration_ids"]
               if any(t["name"] == "OCR" for t in ds[str(i)]["tools"])][:3]
        wd = f"{OUT}/smoke"
        os.makedirs(wd, exist_ok=True)
        dump = f"{wd}/prompts.txt"
        for f in (dump, f"{wd}/transform.jsonl"):
            open(f, "w").close()
        tools = list(json.load(open(R.TOOLMETA)))
        R.set_proxy(ports["proxy"], dict(
            mode="passthrough", mask=tools, per_tool_modes={}, phi_toolmeta=None,
            unavailable_tools=R.UNAVAILABLE, log_path=f"{wd}/proxy.jsonl",
            taco_spec={"default": {"format": "F5", "length": {"level": "L4"}},
                       "per_tool": {}},
            taco_log_path=f"{wd}/transform.jsonl", run_meta=dict(run_id="phase0_smoke")))
        env = dict(os.environ)
        env.update(GTA_MODEL_NAME=a.model,
                   GTA_LLM_URL=f"http://127.0.0.1:{ports['llm']}/v1/chat/completions",
                   GTA_TOOLSERVER=f"http://127.0.0.1:{ports['proxy']}",
                   GTA_TOOLMETA=R.TOOLMETA, GTA_EVAL_MODES="end", GTA_TEMP="0",
                   GTA_TASK_IDS=",".join(map(str, ids)),
                   GTA_EXTRA_TOOLS=",".join(tools), GTA_DUMP_PROMPT=dump,
                   GTA_MAX_TURN="6")
        env.pop("GTA_PREDICTED_MASK", None)
        with open(f"{wd}/smoke.log", "w") as lf:
            subprocess.run(["python", f"{R.LAB}/GTA/opencompass/run.py",
                            f"{R.LAB}/configs/gta_atomic_env.py",
                            "--max-num-workers", "1", "--debug", "-w", wd],
                           env=env, cwd=f"{BIG}/ocrun", stdout=lf,
                           stderr=subprocess.STDOUT)
        menus = [ln.split(":", 1)[1].strip() for ln in open(dump)
                 if ln.startswith("MENU(")]
        uniq = sorted({",".join(sorted(m.split(","))) for m in menus})
        live_ok = len(uniq) == 1 and len(menus) >= 1
        menu_detail = f"{len(menus)} menus, {len(uniq)} distinct; " + (uniq[0][:200] if uniq else "")
        # Direct proxy exercise: the agent is tool-lazy under a full menu (a
        # documented property of this testbed), so the live transform path is
        # driven directly rather than hoping the agent calls a tool.
        import requests
        img = f"{BIG}/data/gta_dataset/{ds[str(ids[0])]['files'][0]['path']}"
        direct = []
        for fmt in ["F0", "F1", "F4", "F5"]:
            R.set_proxy(ports["proxy"], dict(
                taco_spec={"default": {"format": fmt, "length": {"level": "L4"}},
                           "per_tool": {}},
                taco_log_path=f"{wd}/transform.jsonl"))
            rr = requests.post(f"http://127.0.0.1:{ports['proxy']}/OCR",
                               files={"image": ("i.jpg", open(img, "rb"), "image/jpeg")},
                               timeout=300)
            direct.append((fmt, rr.status_code, (rr.text or "")[:120]))
        tl = [json.loads(x) for x in open(f"{wd}/transform.jsonl") if x.strip()]
        applied = [x for x in tl if x.get("applied")]
        fails = [x for x in applied if not x["preservation"]["units_present"]]
        # the F0 control must come back byte-identical to the native output
        f0 = [x for x in applied if x["format"] == "F0"]
        f0_ok = all(x["identical_to_native"] for x in f0) and bool(f0)
        json.dump(dict(direct=direct, records=tl),
                  open(f"{OUT}/live_transform_check.json", "w"), indent=1)
        pres_detail = (f"{len(applied)} live transforms ({len(direct)} direct proxy "
                       f"calls), {len(fails)} preservation failures, F0 identity "
                       f"{'ok' if f0_ok else 'BROKEN'}")
        applied = applied if f0_ok else []
        # gold strings must not appear in a prompt unless the tool output put
        # them there (i.e. the harness itself never injects the label)
        prompt_txt = open(dump).read()
        leaked = []
        for i in ids:
            g = ds[str(i)].get("gt_answer")
            if isinstance(g, dict):
                for grp in g["whitelist"]:
                    for al in grp:
                        if len(al) > 6 and al.lower() in prompt_txt.lower() \
                           and not any(al.lower() in (x.get("original") or "").lower()
                                       for x in tl):
                            leaked.append((i, al))
        check(res, "A4.no_gold_in_prompt", not leaked, leaked[:5])
        check(res, "A5.forced_menu_invariant_across_queries", live_ok, menu_detail)
        check(res, "A8.live_factual_preservation", applied and not fails, pres_detail)
    else:
        check(res, "A5.forced_menu_invariant_across_queries", False, "skipped (--skip-live)")
        check(res, "A8.live_factual_preservation", False, "skipped (--skip-live)")

    print("A6 condition labelling")
    manifests = glob.glob(f"{TACO}/configs/*.jsonl")
    unl = []
    for m in manifests:
        for ln in open(m):
            c = json.loads(ln)
            if c.get("predicted_mask") and c.get("label") != "QUERY-LEVEL TOOL SELECTION":
                unl.append((os.path.basename(m), c["run_id"]))
            if not c.get("label"):
                unl.append((os.path.basename(m), c["run_id"]))
    check(res, "A6.query_level_conditions_labelled", not unl,
          f"{len(manifests)} manifests; {len(unl)} unlabelled")

    print("A7 held-out leakage guard")
    fired = False
    try:
        R.run_one(dict(run_id="phase0_leakguard", stage="phase0", kind="calibration",
                       task_set=[splits["heldout_ids"][0]], menu={"mode": "per_task"}),
                  a.lane, a.model, splits, ds, list(json.load(open(R.TOOLMETA))))
    except SystemExit as e:
        fired = "LEAKAGE GUARD" in str(e)
    check(res, "A7.leakage_guard_fires", fired)

    ok = all(r["ok"] for r in res)
    json.dump(dict(all_pass=ok, checks=res), open(f"{OUT}/phase0_checks.json", "w"), indent=1)

    with open(f"{TACO}/PHASE0_AUDIT.md", "w") as f:
        f.write("# TACO Phase 0 — harness and leakage audit\n\n")
        f.write("Verdict: **%s**\n\n" % ("ALL INVARIANTS HOLD" if ok else "FAILED — STOP"))
        f.write("| check | result | detail |\n|---|---|---|\n")
        for r in res:
            f.write("| `%s` | %s | %s |\n" % (r["check"], "PASS" if r["ok"] else "**FAIL**",
                                              r["detail"].replace("|", "/")[:200]))
        f.write("\n## Notes\n\n")
        f.write("* A3 detail: `phase0_audit/transform_validation.json` "
                "(%d cases).\n" % tv["n_cases"])
        f.write("* A5 is checked on a live forced-mask run: every query's visible menu "
                "must be the same set, which is what makes the environment *global* "
                "rather than query-level.\n")
        f.write("* A7 attempts a calibration-kind run whose task set contains a held-out "
                "id; the runner must refuse it.\n")
        f.write("* Split reuse rationale and achieved stratification: `splits.json`.\n")
    print("\nPhase 0:", "ALL PASS" if ok else "FAILURES PRESENT", "->", f"{TACO}/PHASE0_AUDIT.md")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
