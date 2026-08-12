#!/usr/bin/env python3
"""Summarize pool_subset_experiment.py output: does the likelihood-selected
top-K subset beat showing the model the whole pool (and beat a random
same-size subset, to control for "smaller menu helps regardless of which
tools are removed")?
"""
import argparse
import json
import statistics as st
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cond_stats(rows, cond):
    entries = [r[cond] for r in rows if cond in r and 'skipped' not in r[cond]]
    n = len(entries)
    if n == 0:
        return dict(n=0)
    correct = sum(e['correct'] for e in entries)
    gt_in_menu = sum(e['gt_in_menu'] for e in entries)
    # accuracy conditional on GT actually being in the shown menu -- separates
    # "selector/random dropped the right tool" from "model had it and still
    # got it wrong"
    with_gt = [e for e in entries if e['gt_in_menu']]
    acc_given_gt_present = (sum(e['correct'] for e in with_gt) / len(with_gt)
                             if with_gt else None)
    return dict(
        n=n,
        accuracy=correct / n,
        gt_in_menu_rate=gt_in_menu / n,
        accuracy_given_gt_in_menu=acc_given_gt_present,
        mean_menu_size=st.mean(e['menu_size'] for e in entries),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.results) if 'error' not in r]
    n_tasks = len(rows)
    pool_sizes = [r['pool_size'] for r in rows if 'pool_size' in r]

    summary = dict(
        n_tasks=n_tasks,
        pool_size_mean=st.mean(pool_sizes) if pool_sizes else None,
        pool_size_min=min(pool_sizes) if pool_sizes else None,
        pool_size_max=max(pool_sizes) if pool_sizes else None,
        full=cond_stats(rows, 'full'),
        topk=cond_stats(rows, 'topk'),
        randomk=cond_stats(rows, 'randomk'),
    )

    # stratify by pool size -- the "gain from pruning" is expected to grow
    # with how cluttered the full menu is (GTA's mask-axis finding: gain
    # scales with how much irrelevant clutter is in the pool)
    buckets = [('small_pool_lt15', lambda p: p < 15),
               ('mid_pool_15_30', lambda p: 15 <= p < 30),
               ('large_pool_ge30', lambda p: p >= 30)]
    by_pool_size = {}
    for label, pred in buckets:
        sub = [r for r in rows if pred(r['pool_size'])]
        if not sub:
            continue
        by_pool_size[label] = dict(
            n_tasks=len(sub),
            full_accuracy=cond_stats(sub, 'full')['accuracy'],
            topk_accuracy=cond_stats(sub, 'topk')['accuracy'],
            randomk_accuracy=cond_stats(sub, 'randomk')['accuracy'],
        )
    summary['by_pool_size'] = by_pool_size

    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
