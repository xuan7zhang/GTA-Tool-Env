#!/usr/bin/env python3
"""Cluster BFCL v1 `live_multiple` tasks by tool-space similarity.

Each task is represented as a multi-hot binary vector over the 457 distinct
function names in the dataset (1 = this function is one of the task's
candidates). Rows are L2-normalized so Euclidean k-means behaves like
spherical/cosine k-means -- appropriate for sparse binary "which tools does
this task's menu contain" vectors, where what matters is which tools
overlap, not raw Euclidean distance in a 457-dim {0,1} space.

Two clustering methods:

- **kmeans** (original): forces every task into one of K clusters. Produces
  mostly-coherent domain clusters, but tasks whose tool set barely overlaps
  with anything else still get dumped into whichever centroid is nearest,
  even when the fit is poor -- one such "leftover bin" (K=40: 122 tasks, 97
  distinct ground-truth answers) motivated the second method.
- **dbscan** (fix): density-based, explicitly labels isolated tasks as noise
  (cluster -1) instead of forcing them into a bucket. On this dataset it
  cuts the incoherent-task count from 140 (kmeans K=40) to 32, at the cost
  of an honest 14% "no good cluster" bucket instead of zero.

Motivation: this investigation's forced-scoring / pool-subset experiments
rank a task's candidate pool online, per task, at inference time. Clustering
answers a different, complementary question: do many tasks share close to
the same tool subset, such that an offline "which domain is this task in ->
here's its cached candidate subset" lookup could replace most of the online
per-task scoring calls? A big, coherent cluster (e.g. movie ticketing) is a
candidate for that; a noise point / incoherent cluster member is not and
still needs the online path.
"""
import argparse
import json
import collections
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
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


def sweep_eps(D, n_tasks, values, min_samples=4):
    rows = []
    for eps in values:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(D)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        rows.append(dict(eps=eps, n_clusters=n_clusters, n_noise=n_noise,
                          noise_pct=100 * n_noise / n_tasks))
    return rows


def coherence_label(n_tasks, n_distinct_gt):
    ratio = n_distinct_gt / n_tasks
    if ratio > 0.5:
        return ratio, 'incoherent'
    if ratio > 0.15:
        return ratio, 'mixed'
    return ratio, 'coherent'


def summarize_clusters(tasks, answers, labels):
    clusters = collections.defaultdict(list)
    for i, task in enumerate(tasks):
        clusters[int(labels[i])].append(task)

    summaries = []
    # -1 (DBSCAN noise) sorted last regardless of size
    order = sorted(clusters, key=lambda c: (c == -1, -len(clusters[c])))
    for cluster_id in order:
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
        ratio, coherence = (None, 'noise') if cluster_id == -1 else \
            coherence_label(len(cluster_tasks), len(gt_counter))
        summaries.append(dict(
            cluster=cluster_id,
            n_tasks=len(cluster_tasks),
            n_distinct_gt_tools=len(gt_counter),
            coherence_ratio=ratio,
            coherence=coherence,
            top_tools=[name for name, _ in tool_counter.most_common(5)],
            sample_queries=sample_queries,
            task_ids=[t['id'] for t in cluster_tasks],
        ))
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--method', choices=['kmeans', 'dbscan'], default='kmeans')
    ap.add_argument('--k', type=int, default=40, help='kmeans only')
    ap.add_argument('--eps', type=float, default=0.3, help='dbscan only (cosine distance)')
    ap.add_argument('--min-samples', type=int, default=4, help='dbscan only')
    ap.add_argument('--sweep', action='store_true',
                     help='print a parameter sweep before clustering '
                          '(silhouette/inertia for kmeans, cluster/noise counts for dbscan)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tasks = read_jsonl(data_dir / 'questions.json')
    answers = {a['id']: a for a in read_jsonl(data_dir / 'answers.json')}
    tasks = [t for t in tasks if t['id'] in answers]

    X, vocab = build_matrix(tasks)
    Xn = normalize(X)
    print(f'{len(tasks)} tasks x {len(vocab)} distinct tool names')

    if args.method == 'kmeans':
        if args.sweep:
            for row in sweep_k(Xn, [10, 20, 30, 40, 50, 60, 80]):
                print(f"K={row['k']:3d}  silhouette={row['silhouette']:.3f}  "
                      f"inertia={row['inertia']:.1f}")
        km = KMeans(n_clusters=args.k, random_state=0, n_init=10).fit(Xn)
        labels = km.labels_
    else:
        D = cosine_distances(Xn)
        if args.sweep:
            for row in sweep_eps(D, len(tasks), [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
                                  args.min_samples):
                print(f"eps={row['eps']:.2f}  n_clusters={row['n_clusters']:3d}  "
                      f"n_noise={row['n_noise']:4d} ({row['noise_pct']:.1f}%)")
        db = DBSCAN(eps=args.eps, min_samples=args.min_samples,
                     metric='precomputed').fit(D)
        labels = db.labels_

    summaries = summarize_clusters(tasks, answers, labels)

    cat_counts = collections.Counter(s['coherence'] for s in summaries)
    task_counts = collections.Counter()
    for s in summaries:
        task_counts[s['coherence']] += s['n_tasks']
    print(f"\ncoherence roll-up: clusters={dict(cat_counts)}  tasks={dict(task_counts)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(f'wrote {len(summaries)} clusters -> {args.out}')


if __name__ == '__main__':
    main()
