#!/usr/bin/env python3
"""Acceptance unit tests (no GPU). Run: python test_gate.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchestrate import evaluate_gate, expand_jobs, plan, validate_wiring, is_gpu
import yaml

CALLFREQ_GATE = {"name": "run_iter2_declutter", "metric": "mean_acc", "op": ">", "value": 13.5}


def test_gate_threshold():
    # gate must trigger at the paper's cf_attractive mean 14.49, not at 13.02
    assert evaluate_gate({"mean_acc": 14.49}, CALLFREQ_GATE) is True, "should fire at 14.49"
    assert evaluate_gate({"mean_acc": 13.02}, CALLFREQ_GATE) is False, "should NOT fire at 13.02"
    assert evaluate_gate({"mean_acc": 13.5}, CALLFREQ_GATE) is False, "strict > : 13.5 does not fire"
    assert evaluate_gate({}, CALLFREQ_GATE) is False, "missing metric -> no fire"
    print("PASS test_gate_threshold")


def test_plan_and_wiring():
    cfg = yaml.safe_load((Path(__file__).parent / "run_config.yaml").read_text())
    jobs = expand_jobs(cfg["jobs"])
    ordered = plan(jobs)
    # (1) all non-GPU jobs come before any GPU job
    first_gpu = next(i for i, j in enumerate(ordered) if is_gpu(j))
    assert all(not is_gpu(j) for j in ordered[:first_gpu]), "non-GPU must precede GPU"
    assert any(is_gpu(j) for j in ordered), "must have GPU jobs"
    # (2) wiring self-consistent
    assert validate_wiring(jobs) == [], "wiring must be self-consistent"
    # (3) iter2 requires_flag exists and is produced by the callfreq gate
    iter2 = [j for j in jobs if j.get("requires_flag") == "run_iter2_declutter"]
    producers = [j for j in jobs if (j.get("sets_flag") or {}).get("name") == "run_iter2_declutter"]
    assert iter2 and producers, "iter2 must exist and have a producing gate"
    print("PASS test_plan_and_wiring")


if __name__ == "__main__":
    test_gate_threshold()
    test_plan_and_wiring()
    print("ALL TESTS PASS")
