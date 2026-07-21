#!/usr/bin/env python3
"""Generate the complete tool-coalition manifest for exact Shapley evaluation.

For n candidate tools the output contains every one of the 2**n coalitions for
every requested seed.  A coalition is applied in two places by ``sweep.sh``:

* ``mask`` filters the proxy/OpenAPI surface and rejects masked calls;
* ``extra_tools`` plus ``hide_tools`` makes each task's menu exactly the
  coalition (and any fixed tools), matching the OctoTools and MedRAX setup.

The second restriction matters because GTA's ``tool_meta`` fallback creates
dummy actions for tools absent from the proxy.  Keeping the full metadata file
and explicitly hiding the complement also makes the empty coalition a genuine
no-tool condition without creating one metadata file per coalition.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def load_tool_names(path: Path) -> list[str]:
    with path.open() as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or not all(
        isinstance(name, str) for name in metadata
    ):
        raise ValueError(f"{path} must be a JSON object keyed by tool name")
    return list(metadata)


def _check_unique(label: str, names: Iterable[str]) -> list[str]:
    values = list(names)
    duplicates = sorted({name for name in values if values.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"duplicate {label}: {', '.join(str(item) for item in duplicates)}"
        )
    return values


def coalition_id(index: int, num_candidates: int) -> str:
    width = max(1, (num_candidates + 3) // 4)
    return f"coalition_{index:0{width}x}"


def iter_coalitions(candidate_tools: list[str]):
    """Yield (bit-mask index, tools) once each, ordered by coalition size."""
    positions = {name: index for index, name in enumerate(candidate_tools)}
    for size in range(len(candidate_tools) + 1):
        for combo in combinations(candidate_tools, size):
            index = sum(1 << positions[name] for name in combo)
            yield index, list(combo)


def build_rows(
    *,
    all_tools: list[str],
    candidate_tools: list[str],
    fixed_tools: list[str],
    seeds: list[int],
    run_prefix: str,
    toolmeta: Path,
    force_coalition_menu: bool = True,
) -> list[dict[str, Any]]:
    all_tools = _check_unique("tool names", all_tools)
    candidate_tools = _check_unique("candidate tools", candidate_tools)
    fixed_tools = _check_unique("fixed tools", fixed_tools)
    seeds = _check_unique("seeds", seeds)

    known = set(all_tools)
    unknown = sorted((set(candidate_tools) | set(fixed_tools)) - known)
    if unknown:
        raise ValueError(f"tools absent from {toolmeta}: {', '.join(unknown)}")
    overlap = sorted(set(candidate_tools) & set(fixed_tools))
    if overlap:
        raise ValueError(
            f"candidate and fixed tools must be disjoint: {', '.join(overlap)}"
        )
    if not candidate_tools:
        raise ValueError("at least one candidate tool is required")

    rows: list[dict[str, Any]] = []
    fixed_set = set(fixed_tools)
    toolmeta_abs = str(toolmeta.resolve())
    for index, coalition in iter_coalitions(candidate_tools):
        allowed_set = fixed_set | set(coalition)
        allowed = [name for name in all_tools if name in allowed_set]
        hidden = [name for name in all_tools if name not in allowed_set]
        cid = coalition_id(index, len(candidate_tools))
        for seed in seeds:
            rows.append(
                {
                    "run_id": f"{run_prefix}_{cid}_s{seed}",
                    "experiment": "exact_tool_shapley",
                    "probe_mode": "passthrough",
                    "per_tool_modes": {},
                    "phi_toolmeta": None,
                    "seed": seed,
                    "toolmeta": toolmeta_abs,
                    "mask": allowed,
                    "hide_tools": hidden,
                    "extra_tools": ",".join(allowed) if force_coalition_menu else "",
                    "eval_modes": "end",
                    "coalition_id": cid,
                    "coalition_index": index,
                    "coalition": coalition,
                    "candidate_tools": candidate_tools,
                    "fixed_tools": fixed_tools,
                    "num_candidates": len(candidate_tools),
                    "num_coalition_tools": len(coalition),
                    "menu_semantics": (
                        "exact_coalition"
                        if force_coalition_menu
                        else "natural_task_intersection"
                    ),
                }
            )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all 2^n tool coalitions for exact Shapley evaluation."
    )
    parser.add_argument("--toolmeta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-prefix", default="shapley")
    parser.add_argument(
        "--tools",
        help="comma-separated Shapley candidates (default: every tool in toolmeta)",
    )
    parser.add_argument(
        "--fixed-tools",
        default="",
        help="comma-separated tools present in every coalition",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--natural-task-menus",
        action="store_true",
        help=(
            "do not force S onto every task; expose only each task's natural "
            "resources intersected with S"
        ),
    )
    args = parser.parse_args()

    all_tools = load_tool_names(args.toolmeta)
    candidate_tools = parse_csv(args.tools) or all_tools
    fixed_tools = parse_csv(args.fixed_tools) or []
    rows = build_rows(
        all_tools=all_tools,
        candidate_tools=candidate_tools,
        fixed_tools=fixed_tools,
        seeds=args.seeds,
        run_prefix=args.run_prefix,
        toolmeta=args.toolmeta,
        force_coalition_menu=not args.natural_task_menus,
    )
    count = write_jsonl(args.out, rows)
    coalitions = 1 << len(candidate_tools)
    print(
        f"wrote {count} runs ({coalitions} coalitions x {len(args.seeds)} seeds) "
        f"for {len(candidate_tools)} candidate tools to {args.out}"
    )


if __name__ == "__main__":
    main()
