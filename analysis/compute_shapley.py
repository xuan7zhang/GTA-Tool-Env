#!/usr/bin/env python3
"""Compute exact per-tool Shapley values from a full GTA coalition sweep.

The characteristic value v(S) is a GTA evaluator metric (``answer_acc`` by
default) for the run whose available candidate-tool set is S.  Exact scores are
only emitted for seeds with all 2**n coalitions; incomplete sweeps are reported
but never silently approximated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def exact_shapley(
    values: Mapping[int, float], candidate_tools: list[str]
) -> dict[str, float]:
    """Return exact Shapley values for a bit-mask-indexed characteristic game."""
    n = len(candidate_tools)
    expected = set(range(1 << n))
    missing = expected - set(values)
    extra = set(values) - expected
    if missing or extra:
        raise ValueError(
            f"exact Shapley needs all {1 << n} coalitions "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    scores = {tool: 0.0 for tool in candidate_tools}
    for index, tool in enumerate(candidate_tools):
        bit = 1 << index
        for coalition in range(1 << n):
            if coalition & bit:
                continue
            size = bin(coalition).count("1")
            # |S|! (n-|S|-1)! / n! = 1 / (n * C(n-1, |S|))
            weight = 1.0 / (n * math.comb(n - 1, size))
            scores[tool] += weight * (
                float(values[coalition | bit]) - float(values[coalition])
            )
    return scores


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if row.get("experiment") != "exact_tool_shapley":
                raise ValueError(
                    f"{path}:{line_number}: not an exact_tool_shapley row"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no Shapley rows")
    return rows


def validate_manifest(
    rows: list[dict[str, Any]],
) -> tuple[list[str], list[str], str]:
    candidates = list(rows[0].get("candidate_tools") or [])
    fixed = list(rows[0].get("fixed_tools") or [])
    menu_semantics = str(rows[0].get("menu_semantics") or "unspecified")
    if not candidates:
        raise ValueError("manifest has no candidate_tools")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("candidate_tools") != candidates:
            raise ValueError("candidate_tools/order changes within the manifest")
        if row.get("fixed_tools") != fixed:
            raise ValueError("fixed_tools changes within the manifest")
        if str(row.get("menu_semantics") or "unspecified") != menu_semantics:
            raise ValueError("menu_semantics changes within the manifest")
        index = int(row["coalition_index"])
        seed = str(row.get("seed", 0))
        key = (seed, index)
        if key in seen:
            raise ValueError(f"duplicate coalition {index} for seed {seed}")
        seen.add(key)
        expected_coalition = [
            tool for bit, tool in enumerate(candidates) if index & (1 << bit)
        ]
        if row.get("coalition") != expected_coalition:
            raise ValueError(
                f"coalition_index {index} does not match coalition tool list"
            )
    return candidates, fixed, menu_semantics


def find_result_file(run_dir: Path, dataset: str) -> Path | None:
    direct = sorted(run_dir.glob(f"*/results/*/{dataset}.json"))
    if direct:
        return direct[-1]
    recursive = sorted(run_dir.glob(f"**/results/*/{dataset}.json"))
    return recursive[-1] if recursive else None


def load_metric(path: Path, metric: str) -> float:
    with path.open() as handle:
        payload = json.load(handle)
    value = payload.get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"metric {metric!r} is absent or non-numeric in {path}")
    return float(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    *,
    manifest: Path,
    results_root: Path,
    out_dir: Path,
    dataset: str = "gta_bench_end",
    metric: str = "answer_acc",
) -> dict[str, Any]:
    rows = read_manifest(manifest)
    candidate_tools, fixed_tools, menu_semantics = validate_manifest(rows)
    expected_indices = set(range(1 << len(candidate_tools)))
    out_dir.mkdir(parents=True, exist_ok=True)

    values_by_seed: dict[str, dict[int, float]] = defaultdict(dict)
    manifest_indices_by_seed: dict[str, set[int]] = defaultdict(set)
    coalition_rows: list[dict[str, Any]] = []
    for row in rows:
        seed = str(row.get("seed", 0))
        index = int(row["coalition_index"])
        manifest_indices_by_seed[seed].add(index)
        result_path = find_result_file(results_root / row["run_id"], dataset)
        status = "missing"
        value: float | None = None
        error = ""
        if result_path is not None:
            try:
                value = load_metric(result_path, metric)
                values_by_seed[seed][index] = value
                status = "ok"
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                status = "invalid"
                error = str(exc)
        coalition_rows.append(
            {
                "seed": seed,
                "coalition_id": row.get("coalition_id"),
                "coalition_index": index,
                "num_tools": len(row.get("coalition") or []),
                "tools": ",".join(row.get("coalition") or []),
                "run_id": row["run_id"],
                "status": status,
                "metric": metric,
                "value": value,
                "result_file": str(result_path) if result_path else "",
                "error": error,
            }
        )

    seeds = sorted(manifest_indices_by_seed)
    manifest_missing = {
        seed: sorted(expected_indices - manifest_indices_by_seed[seed]) for seed in seeds
    }
    result_missing = {
        seed: sorted(expected_indices - set(values_by_seed[seed])) for seed in seeds
    }
    complete_seeds = [
        seed
        for seed in seeds
        if not manifest_missing[seed] and not result_missing[seed]
    ]

    per_seed_rows: list[dict[str, Any]] = []
    scores_by_tool: dict[str, list[float]] = defaultdict(list)
    efficiency_rows: list[dict[str, Any]] = []
    full_index = (1 << len(candidate_tools)) - 1
    for seed in complete_seeds:
        values = values_by_seed[seed]
        scores = exact_shapley(values, candidate_tools)
        total_delta = values[full_index] - values[0]
        residual = sum(scores.values()) - total_delta
        efficiency_rows.append(
            {
                "seed": seed,
                "empty_value": values[0],
                "full_value": values[full_index],
                "full_minus_empty": total_delta,
                "sum_shapley": sum(scores.values()),
                "efficiency_residual": residual,
            }
        )
        for tool in candidate_tools:
            score = scores[tool]
            scores_by_tool[tool].append(score)
            per_seed_rows.append(
                {
                    "seed": seed,
                    "tool": tool,
                    "shapley_value": score,
                    "full_minus_empty": total_delta,
                    "share_of_full_minus_empty": (
                        score / total_delta if total_delta else None
                    ),
                }
            )

    mean_total_delta = (
        statistics.fmean(row["full_minus_empty"] for row in efficiency_rows)
        if efficiency_rows
        else None
    )
    shapley_rows: list[dict[str, Any]] = []
    for tool in candidate_tools:
        tool_scores = scores_by_tool.get(tool, [])
        if not tool_scores:
            continue
        mean_score = statistics.fmean(tool_scores)
        stdev = statistics.stdev(tool_scores) if len(tool_scores) > 1 else 0.0
        shapley_rows.append(
            {
                "tool": tool,
                "metric": metric,
                "shapley_value": mean_score,
                "shapley_stdev": stdev,
                "shapley_sem": stdev / math.sqrt(len(tool_scores)),
                "num_complete_seeds": len(tool_scores),
                "share_of_mean_full_minus_empty": (
                    mean_score / mean_total_delta if mean_total_delta else None
                ),
            }
        )
    shapley_rows.sort(key=lambda item: item["shapley_value"], reverse=True)

    write_csv(
        out_dir / "coalition_results.csv",
        coalition_rows,
        [
            "seed",
            "coalition_id",
            "coalition_index",
            "num_tools",
            "tools",
            "run_id",
            "status",
            "metric",
            "value",
            "result_file",
            "error",
        ],
    )
    write_csv(
        out_dir / "shapley_by_seed.csv",
        per_seed_rows,
        [
            "seed",
            "tool",
            "shapley_value",
            "full_minus_empty",
            "share_of_full_minus_empty",
        ],
    )
    write_csv(
        out_dir / "shapley_scores.csv",
        shapley_rows,
        [
            "tool",
            "metric",
            "shapley_value",
            "shapley_stdev",
            "shapley_sem",
            "num_complete_seeds",
            "share_of_mean_full_minus_empty",
        ],
    )

    complete = len(complete_seeds) == len(seeds) and bool(seeds)
    summary = {
        "manifest": str(manifest),
        "results_root": str(results_root),
        "dataset": dataset,
        "metric": metric,
        "candidate_tools": candidate_tools,
        "fixed_tools": fixed_tools,
        "menu_semantics": menu_semantics,
        "num_candidate_tools": len(candidate_tools),
        "expected_coalitions_per_seed": 1 << len(candidate_tools),
        "seeds": seeds,
        "complete_seeds": complete_seeds,
        "complete": complete,
        "missing_manifest_coalitions": manifest_missing,
        "missing_result_coalitions": result_missing,
        "efficiency_checks": efficiency_rows,
        "shapley_scores": shapley_rows,
        "outputs": {
            "coalitions": str(out_dir / "coalition_results.csv"),
            "by_seed": str(out_dir / "shapley_by_seed.csv"),
            "scores": str(out_dir / "shapley_scores.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default="gta_bench_end")
    parser.add_argument("--metric", default="answer_acc")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="exit nonzero when any seed lacks a coalition result",
    )
    args = parser.parse_args()

    summary = analyze(
        manifest=args.manifest,
        results_root=args.results,
        out_dir=args.out,
        dataset=args.dataset,
        metric=args.metric,
    )
    print(json.dumps(summary, indent=2))
    if args.require_complete and not summary["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
