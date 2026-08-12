#!/usr/bin/env python3
"""Per-cluster tool subspace, built with token likelihood (not the failed
"cache every historical GT answer" approach from cluster_cache_experiment.py).

For a cluster of tasks that share similar tool usage (from
cluster_tasks_by_tool_space.py), build ONE shared subspace for the whole
cluster:

  1. Pool = union of every candidate tool ever shown to any task in the
     cluster (not just GT answers -- the full native candidate lists).
  2. For every task in the cluster, forced-score every tool in the pool
     (same method validated in score_candidates.py / pool_subset_experiment.py
     -- no ground truth used).
  3. For each tool, average its score across the cluster's tasks, computed
     leave-one-out per evaluated task (task T's own subspace is built from
     the OTHER tasks' scores only, so nothing about T's own query influences
     the subspace it's tested against -- avoids any accusation of
     overfitting the subspace to the task being scored, even though the
     forced-scoring itself never touches ground truth either way).
  4. Rank tools by LOO-averaged score, keep top-K -> that task's subspace
     (in practice nearly identical across all tasks in the cluster, since
     only one row differs).

Real unconstrained generation compares native_full (task's own BFCL
candidate list) vs. this likelihood-filtered cluster subspace, and both
against the earlier (failed) cluster_cache result for the same tasks.
"""
import argparse
import json
import math
import statistics as st
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def function_block(fn, idx):
    params = json.dumps(fn.get('parameters', {}), ensure_ascii=False)
    return (f"Function {idx}: {fn['name']}\n"
            f"Description: {fn.get('description', '')}\n"
            f"Parameters: {params}")


def build_menu_prefix(user_text, tools):
    functions = '\n\n'.join(function_block(fn, i + 1) for i, fn in enumerate(tools))
    return (
        'You have access to the following functions. Given the user request, '
        'decide which single function should be called next.\n\n'
        f'{functions}\n\n'
        f'User request: {user_text}\n\n'
        'Thought: I will pick the function that matches this request.\n'
        'Action:')


def _finite(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def best_match(text, names):
    if not text:
        return None
    text = text.strip()
    best = None
    for name in names:
        short = name.split('.')[-1]
        if name in text:
            if best is None or len(name) > len(best):
                best = name
        elif short in text and sum(1 for n in names if n.split('.')[-1] == short) == 1:
            if best is None:
                best = name
    return best


class Client:
    def __init__(self, api_base, model, top_k=20, timeout=120):
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.top_k = top_k
        self.timeout = timeout
        self._session = requests.Session()

    def score_continuation(self, prefix, continuation):
        full_text = prefix + continuation
        resp = self._session.post(
            f'{self.api_base}/completions',
            json=dict(model=self.model, prompt=full_text, max_tokens=0,
                      echo=True, logprobs=self.top_k),
            timeout=self.timeout)
        resp.raise_for_status()
        choice = resp.json()['choices'][0]
        logprobs = choice['logprobs']
        boundary = len(prefix)
        tail = [
            _finite(lp, float('nan'))
            for offset, lp in zip(logprobs['text_offset'], logprobs['token_logprobs'])
            if offset is not None and offset >= boundary
        ]
        finite = [v for v in tail if math.isfinite(v)]
        return sum(finite) / len(finite) if finite else None

    def free_generate(self, prefix, max_tokens=24):
        resp = self._session.post(
            f'{self.api_base}/completions',
            json=dict(model=self.model, prompt=prefix, max_tokens=max_tokens,
                      temperature=0.0, stop=['\n']),
            timeout=self.timeout)
        if resp.status_code == 400:
            return None
        resp.raise_for_status()
        return resp.json()['choices'][0]['text'].strip()


def user_text_of(task):
    turns = task['question'][0] if isinstance(task['question'][0], list) else task['question']
    return next((t['content'] for t in turns if t.get('role') == 'user'), '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--cluster-id', type=int, required=True)
    ap.add_argument('--api-base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--scoring-workers', type=int, default=10)
    ap.add_argument('--max-tasks', type=int, default=None, help='smoke-test cap')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    all_tasks = {t['id']: t for t in read_jsonl(data_dir / 'questions.json')}
    answers = {a['id']: a for a in read_jsonl(data_dir / 'answers.json')}

    cluster_entries = json.loads(Path(args.clusters).read_text())
    entry = next(c for c in cluster_entries if c['cluster'] == args.cluster_id)
    task_ids = entry['task_ids']
    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]
    tasks = [all_tasks[tid] for tid in task_ids]
    print(f'cluster {args.cluster_id}: {len(tasks)} tasks')

    pool = {}
    for task in tasks:
        for fn in task['function']:
            pool.setdefault(fn['name'], fn)
    pool_names = sorted(pool.keys())
    print(f'native pool (union of candidate tools in this cluster): {len(pool_names)} tools')

    client = Client(args.api_base, args.model)

    # Step 1: score every (task, tool) pair against the full cluster pool.
    # Flat job list submitted to ONE thread pool -- avoids the overhead (and
    # the effective concurrency cap) of spinning up a fresh nested
    # ThreadPoolExecutor per task, which serialized scoring far below the
    # intended concurrency in an earlier version of this script.
    lock = threading.Lock()
    score_matrix = {tid: {} for tid in [t['id'] for t in tasks]}
    prefixes = {t['id']: build_menu_prefix(user_text_of(t), list(pool.values()))
                for t in tasks}
    jobs = [(t['id'], name) for t in tasks for name in pool_names]
    done = 0
    total_jobs = len(jobs)

    print(f'scoring {total_jobs} (task, tool) pairs against the cluster pool '
          f'with {args.scoring_workers} workers...')
    with ThreadPoolExecutor(max_workers=args.scoring_workers) as pool_exec:
        futures = {
            pool_exec.submit(client.score_continuation, prefixes[tid], ' ' + name): (tid, name)
            for tid, name in jobs
        }
        for future in as_completed(futures):
            tid, name = futures[future]
            score = future.result()
            if score is not None:
                score_matrix[tid][name] = score
            with lock:
                done += 1
                if done % 500 == 0 or done == total_jobs:
                    print(f'  scored [{done}/{total_jobs}]')

    # Step 2: for each tool, sum of scores across all tasks that scored it
    # (needed for leave-one-out averages).
    tool_sum = {name: 0.0 for name in pool_names}
    tool_count = {name: 0 for name in pool_names}
    for scores in score_matrix.values():
        for name, score in scores.items():
            tool_sum[name] += score
            tool_count[name] += 1

    # Step 3: build each task's leave-one-out subspace and run real generation.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _run_generation(task):
        task_id = task['id']
        gt_name = next(iter(answers[task_id]['ground_truth'][0].keys()))
        user_text = user_text_of(task)

        loo_avg = {}
        own_scores = score_matrix.get(task_id, {})
        for name in pool_names:
            n = tool_count[name] - (1 if name in own_scores else 0)
            if n <= 0:
                continue
            total = tool_sum[name] - own_scores.get(name, 0.0)
            loo_avg[name] = total / n
        ranked = sorted(loo_avg.items(), key=lambda x: x[1], reverse=True)
        subspace_names = [name for name, _ in ranked[:args.topk]]
        subspace_tools = [pool[name] for name in subspace_names]

        result = dict(task_id=task_id, gt_name=gt_name,
                       subspace_names=subspace_names,
                       gt_in_subspace=(gt_name in subspace_names))

        prefix_full = build_menu_prefix(user_text, task['function'])
        text_full = client.free_generate(prefix_full)
        names_full = [fn['name'] for fn in task['function']]
        choice_full = best_match(text_full, names_full) if text_full else None
        result['native_full'] = dict(menu_size=len(names_full), free_text=text_full,
                                      choice=choice_full, correct=(choice_full == gt_name))

        prefix_sub = build_menu_prefix(user_text, subspace_tools)
        text_sub = client.free_generate(prefix_sub)
        choice_sub = best_match(text_sub, subspace_names) if text_sub else None
        result['likelihood_subspace'] = dict(
            menu_size=len(subspace_names), free_text=text_sub, choice=choice_sub,
            correct=(choice_sub == gt_name))
        return result

    print('running real generation (native_full vs likelihood_subspace)...')
    done = 0
    with out_path.open('w', encoding='utf-8') as handle, \
            ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        futures = {pool_exec.submit(_run_generation, task): task['id'] for task in tasks}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                record = dict(task_id=task_id, error=str(exc))
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            with lock:
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f'  generated [{done}/{len(tasks)}]')

    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
