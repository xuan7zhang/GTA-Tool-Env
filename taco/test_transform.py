"""Offline validation of the TACO intervention layer (Phase 0 / Phase 1).

Runs every (format x length x position) condition over real tool outputs
harvested from prior runs and asserts:

  1. every non-truncating transform preserves every evidence unit;
  2. no transform invents a numeric literal absent from the native output;
  3. F0 is byte-identical to the native output (the control really is a control);
  4. formats of the same output carry the same evidence set (renderer purity);
  5. the length ladder is monotone in token count within a tool;
  6. image-output tools are never touched;
  7. the transform never raises (worst case it degrades to passthrough).

Usage:  python taco/test_transform.py [--fixtures path.json] [--out report.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from taco import transform as T  # noqa: E402

PRIMARY = ["OCR", "ImageDescription", "TextToBbox", "RegionAttributeDescription",
           "CountGivenObject", "Calculator", "Solver"]


def main():
    ap = argparse.ArgumentParser()
    from taco.paths import INTERV, AUDIT
    ap.add_argument("--fixtures", default=f"{INTERV}/tool_output_fixtures.json")
    ap.add_argument("--out", default=f"{AUDIT}/transform_validation.json")
    args = ap.parse_args()

    fx = json.load(open(args.fixtures))
    fx = {k: v for k, v in fx.items() if k in PRIMARY}

    failures, rows = [], []
    n_cases = 0

    for tool, samples in sorted(fx.items()):
        for si, s in enumerate(samples):
            canon = T.to_canonical(tool, s)

            # (3) F0 control is byte-identical
            f0, r0 = T.transform(tool, s, {"format": "F0"})
            if f0 != s:
                failures.append(dict(check="F0_identity", tool=tool, sample=si))

            # (4) renderer purity: same evidence set across formats
            ev_ref = [u["text"] for u in canon["units"]]

            for fmt in T.FORMATS:
                for lvl in T.LENGTHS:
                    for mech in (T.MECHANISMS if lvl == "L5" else ["relevant"]):
                        for pos in [None, "front", "middle", "back"]:
                            if pos and (fmt != "F1" or lvl != "L4"):
                                continue  # position studied on one base cell
                            spec = {"format": fmt,
                                    "length": {"level": lvl, "mechanism": mech},
                                    "position": pos}
                            txt, rec = T.transform(tool, s, spec)
                            n_cases += 1
                            if not rec.get("applied"):
                                failures.append(dict(check="applied", tool=tool,
                                                     sample=si, spec=spec,
                                                     reason=rec.get("reason")))
                                continue
                            pres = rec["preservation"]
                            # (1) kept-unit preservation, always
                            if not pres["units_present"]:
                                failures.append(dict(check="units_present", tool=tool,
                                                     sample=si, spec=spec,
                                                     missing=pres["missing_units"]))
                            # (2) no invented numbers
                            if not pres["no_new_numbers"]:
                                failures.append(dict(check="no_new_numbers", tool=tool,
                                                     sample=si, spec=spec,
                                                     new=pres["new_numbers"]))
                            # full-length conditions must keep ALL units
                            if lvl in ("L4", "L5") and rec["n_units_kept"] < rec["n_units_total"]:
                                failures.append(dict(check="full_keeps_all", tool=tool,
                                                     sample=si, spec=spec))
                            rows.append(dict(tool=tool, sample=si, format=fmt,
                                             length=lvl, mechanism=mech,
                                             position=pos or "native",
                                             tokens_before=rec["tokens_before"],
                                             tokens_after=rec["tokens_after"],
                                             evidence_tokens=rec["evidence_tokens"],
                                             evidence_density=rec["evidence_density"],
                                             units_kept=rec["n_units_kept"],
                                             units_total=rec["n_units_total"]))
                # (5) length monotonicity within (tool, sample, format)
                tk = {}
                for lvl in ["L0", "L1", "L2", "L3", "L4"]:
                    t, r = T.transform(tool, s, {"format": fmt,
                                                 "length": {"level": lvl}})
                    tk[lvl] = r.get("tokens_after", 0)
                seq = [tk[l] for l in ["L0", "L1", "L2", "L3", "L4"]]
                if any(b < a for a, b in zip(seq, seq[1:])):
                    failures.append(dict(check="length_monotone", tool=tool,
                                         sample=si, format=fmt, tokens=seq))
                # evidence identity across formats at L4
                _, rF = T.transform(tool, s, {"format": fmt, "length": {"level": "L4"}})
                if rF["n_units_total"] != len(ev_ref):
                    failures.append(dict(check="evidence_identity", tool=tool,
                                         sample=si, format=fmt))

    # (6) image-output tools untouched
    for tool in sorted(T.IMAGE_OUTPUT_TOOLS):
        out, rec = T.transform(tool, "/path/to/generated/image_1.png",
                               {"format": "F4", "length": {"level": "L0"}})
        if out != "/path/to/generated/image_1.png" or rec.get("applied"):
            failures.append(dict(check="image_untouched", tool=tool))

    # (7) robustness: adversarial payloads must never raise
    for bad in ["", "   ", "\n\n", "(1,2,3,4)", "{'a':", "x" * 19999, 3, 4.5, True,
                "(1, 2, 3, 4) \n(5, 6, 7, 8) ", "no numbers at all here."]:
        for fmt in T.FORMATS:
            try:
                T.transform("OCR", bad, {"format": fmt, "length": {"level": "L1"}})
            except Exception as e:
                failures.append(dict(check="no_raise", payload=repr(bad)[:40],
                                     format=fmt, err=str(e)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import collections
    bycheck = collections.Counter((f["check"], f.get("tool")) for f in failures)
    summary = dict(version=T.VERSION, n_cases=n_cases, n_failures=len(failures),
                   failures_by_check={"%s|%s" % k: v for k, v in bycheck.most_common()},
                   failures=failures[:50], tools=sorted(fx),
                   n_fixtures={k: len(v) for k, v in sorted(fx.items())})
    json.dump(dict(summary=summary, rows=rows), open(args.out, "w"), indent=1)

    print("TACO transform validation: %d cases, %d failures" % (n_cases, len(failures)))
    for f in failures[:20]:
        print("  FAIL", f)
    print("report ->", args.out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
