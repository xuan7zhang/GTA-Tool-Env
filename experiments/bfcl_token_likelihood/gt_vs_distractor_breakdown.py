#!/usr/bin/env python3
"""Per-candidate token-likelihood breakdown: ground-truth tool vs. distractor
tools, pooled across all scored tasks. Answers "what does the likelihood
number actually look like for the tool that's right vs. the tools that
aren't" -- the group-separation evidence behind the AUROC/top-1 numbers in
the pilot report.
"""
import argparse
import json
import statistics as st


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def breakdown(rows):
    gt_lp, distractor_lp = [], []
    gt_ent, distractor_ent = [], []
    for row in rows:
        for cand in row['candidates']:
            bucket_lp = gt_lp if cand['is_gt'] else distractor_lp
            bucket_ent = gt_ent if cand['is_gt'] else distractor_ent
            bucket_lp.append(cand['mean_logprob'])
            bucket_ent.append(cand['mean_entropy_nats'])
    return dict(
        n_gt=len(gt_lp),
        n_distractor=len(distractor_lp),
        gt_mean_logprob=st.mean(gt_lp),
        gt_std_logprob=st.stdev(gt_lp),
        gt_min_logprob=min(gt_lp),
        gt_max_logprob=max(gt_lp),
        distractor_mean_logprob=st.mean(distractor_lp),
        distractor_std_logprob=st.stdev(distractor_lp),
        distractor_min_logprob=min(distractor_lp),
        distractor_max_logprob=max(distractor_lp),
        gap_nats=st.mean(gt_lp) - st.mean(distractor_lp),
        gt_mean_entropy_nats=st.mean(gt_ent),
        distractor_mean_entropy_nats=st.mean(distractor_ent),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', action='append', required=True,
                     help='One or more scored.jsonl files (repeatable); '
                          'each is reported separately.')
    args = ap.parse_args()

    for path in args.scored:
        rows = [r for r in read_jsonl(path) if 'error' not in r]
        result = breakdown(rows)
        result['source'] = path
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
