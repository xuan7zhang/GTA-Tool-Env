#!/usr/bin/env python3
"""Summarize experiments/bfcl_token_likelihood/score_candidates.py output.

Pooled metrics answering: does per-candidate token likelihood pick out the
ground-truth tool out of the task's candidate set?
"""
import argparse
import json
import statistics as st
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def auroc(scores, labels):
    """Pairwise AUROC, half credit for ties (same formula used in
    analysis/analyze_token_uncertainty_correctness.py)."""
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    xbar, ybar = st.mean(xs), st.mean(ys)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = (sum((x - xbar) ** 2 for x in xs) * sum((y - ybar) ** 2 for y in ys)) ** 0.5
    return num / den if den else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.scored) if 'error' not in r]
    n = len(rows)

    top1_acc = sum(r['top1_correct'] for r in rows) / n
    mrr = sum(r['reciprocal_rank'] for r in rows) / n
    random_baseline = sum(1.0 / r['n_candidates'] for r in rows) / n
    free_acc = sum(r['free_choice_correct'] for r in rows) / n

    all_scores, all_labels = [], []
    for r in rows:
        for c in r['candidates']:
            all_scores.append(c['mean_logprob'])
            all_labels.append(c['is_gt'])
    pooled_auroc = auroc(all_scores, all_labels)

    correct_entropy = [r['candidate_entropy_nats'] for r in rows if r['top1_correct']]
    wrong_entropy = [r['candidate_entropy_nats'] for r in rows if not r['top1_correct']]
    normalized_entropy = [r['candidate_entropy_nats'] / r['max_entropy_nats']
                           if r['max_entropy_nats'] > 0 else 0.0 for r in rows]
    top1_flags = [1.0 if r['top1_correct'] else 0.0 for r in rows]
    entropy_vs_correct_r = pearson(normalized_entropy, top1_flags)

    n_candidates = [r['n_candidates'] for r in rows]
    rank_dist = {}
    for r in rows:
        rank_dist[r['rank_of_gt']] = rank_dist.get(r['rank_of_gt'], 0) + 1

    summary = dict(
        n_tasks=n,
        n_candidates_mean=st.mean(n_candidates),
        n_candidates_min=min(n_candidates),
        n_candidates_max=max(n_candidates),
        top1_accuracy_forced_scoring=top1_acc,
        mrr_forced_scoring=mrr,
        random_baseline_accuracy=random_baseline,
        free_generation_accuracy=free_acc,
        pooled_auroc_score_vs_is_gt=pooled_auroc,
        n_scored_candidate_pairs=len(all_scores),
        mean_candidate_entropy_when_top1_correct=(
            st.mean(correct_entropy) if correct_entropy else None),
        mean_candidate_entropy_when_top1_wrong=(
            st.mean(wrong_entropy) if wrong_entropy else None),
        n_top1_correct=len(correct_entropy),
        n_top1_wrong=len(wrong_entropy),
        pearson_normalized_entropy_vs_top1_correct=entropy_vs_correct_r,
        rank_of_gt_distribution=rank_dist,
    )

    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
