#!/usr/bin/env python3
"""Cluster BFCL v1 `live_multiple` tasks by tool-space similarity.

Each task is represented as a multi-hot binary vector over the 457 distinct
function names in the dataset (1 = this function is one of the task's
candidates). Rows are L2-normalized so ordinary (Euclidean) k-means behaves
like spherical/cosine k-means -- appropriate for sparse binary "which tools
does this task's menu contain" vectors, where what matters is which tools
overlap, not raw Euclidean distance in a 457-dim {0,1} space.

Motivation: this investigation's forced-scoring / pool-subset experiments
rank a task's candidate pool online, per task, at inference time. Clustering
answers a different, complementary question: do many tasks share close to
the same tool subset, such that an offline "which domain is this task in ->
here's its cached candidate subset" lookup could replace most of the online
per-task scoring calls? A big domain cluster (e.g. movie ticketing, 60
tasks nearly all drawing from the same 4-5 `Movies_1_*` functions) is a
candidate for that; an isolated task with a unique tool combination is not
and still needs the online path.
"""
import argparse
import json
import collections

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_matrix(tasks):
    vocab = sorted({fn['name'] for t in tasks for fn in t['function']})
    index = {name: i for i, name in enumerate(vocab)}
    matrix = np.zeros((len(tasks), len(vocab)), dtype=np.float32)
    for row, task in enumerate(tasks):
        for fn in task['function']:
            matrix[row, index[fn['name']]] = 1.0
    return matrix, vocab


def sweep_k(Xn, values):
    rows = []
    for k in values:
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xn)
        sil = silhouette_score(Xn, km.labels_, sample_size=min(len(Xn), 2000),
                                random_state=0)
        rows.append(dict(k=k, silhouette=float(sil), inertia=float(km.inertia_)))
    return rows


def summarize_clusters(tasks, answers, labels, k):
    clusters = collections.defaultdict(list)
    for i, task in enumerate(tasks):
        clusters[int(labels[i])].append(task)

    summaries = []
    for cluster_id in sorted(clusters, key=lambda c: -len(clusters[c])):
        cluster_tasks = clusters[cluster_id]
        tool_counter = collections.Counter()
        gt_counter = collections.Counter()
        for task in cluster_tasks:
            for fn in task['function']:
                tool_counter[fn['name']] += 1
            gt_name = next(iter(answers[task['id']]['ground_truth'][0].keys()))
            gt_counter[gt_name] += 1
        sample_queries = []
        for task in cluster_tasks[:3]:
            turns = (task['question'][0] if isinstance(task['question'][0], list)
                     else task['question'])
            user_text = next((t['content'] for t in turns if t.get('role') == 'user'), '')
            sample_queries.append(user_text[:90])
        summaries.append(dict(
            cluster=cluster_id,
            n_tasks=len(cluster_tasks),
            n_distinct_gt_tools=len(gt_counter),
            top_tools=[name for name, _ in tool_counter.most_common(5)],
            sample_queries=sample_queries,
            task_ids=[t['id'] for t in cluster_tasks],
        ))
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--k', type=int, default=40)
    ap.add_argument('--sweep', action='store_true',
                     help='print silhouette/inertia for a range of K before clustering')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    from pathlib import Path
    data_dir = Path(args.data_dir)
    tasks = read_jsonl(data_dir / 'questions.json')
    answers = {a['id']: a for a in read_jsonl(data_dir / 'answers.json')}
    tasks = [t for t in tasks if t['id'] in answers]

    X, vocab = build_matrix(tasks)
    Xn = normalize(X)
    print(f'{len(tasks)} tasks x {len(vocab)} distinct tool names')

    if args.sweep:
        for row in sweep_k(Xn, [10, 20, 30, 40, 50, 60, 80]):
            print(f"K={row['k']:3d}  silhouette={row['silhouette']:.3f}  "
                  f"inertia={row['inertia']:.1f}")

    km = KMeans(n_clusters=args.k, random_state=0, n_init=10).fit(Xn)
    summaries = summarize_clusters(tasks, answers, km.labels_, args.k)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(f'wrote {len(summaries)} clusters -> {args.out}')


if __name__ == '__main__':
    main()
