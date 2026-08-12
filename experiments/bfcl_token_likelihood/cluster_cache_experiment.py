#!/usr/bin/env python3
"""Does the offline cluster-cache idea from BFCL_TOOL_SPACE_CLUSTERING_20260811.md
actually work, measured with real generation?

For every task in a DBSCAN cluster (not noise), build a **leave-one-out**
cache: the union of ground-truth tool names from every OTHER task in the
same cluster (excluding the task itself, so nothing leaks). This simulates
"we've seen this domain's historical tasks, a new one just arrived" -- if a
cluster's own GT tool never appears among its other members, the cache
legitimately misses it, same as a real deployed lookup would.

Two conditions, real unconstrained generation (temperature 0):

  native_full    -- the task's own BFCL-provided candidate list (all tasks)
  cluster_cache  -- ONLY the leave-one-out cluster cache tools (non-noise
                    tasks only) -- the task's own candidate list is not
                    shown at all; this tests whether knowing the domain
                    alone (no per-task lookup, no online scoring) is enough

Blended two-tier system accuracy = cluster_cache accuracy on clustered
tasks + native_full accuracy on noise tasks (the honest fallback for tasks
with no cluster to draw a cache from), compared against "always show
native_full" as the no-optimization baseline.
"""
import argparse
import json
import collections
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
    def __init__(self, api_base, model, timeout=120):
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()

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


def load_clusters(path):
    """task_id -> cluster_id (int, -1 = noise)."""
    entries = json.loads(Path(path).read_text())
    mapping = {}
    for entry in entries:
        for task_id in entry['task_ids']:
            mapping[task_id] = entry['cluster']
    return mapping


def build_tool_schema_index(tasks):
    """First-seen schema per distinct function name, across the whole dataset."""
    index = {}
    for task in tasks:
        for fn in task['function']:
            index.setdefault(fn['name'], fn)
    return index


def build_loo_caches(tasks, answers, task_to_cluster):
    """cluster_id -> {task_id -> set(GT tool names from every OTHER task in
    the same cluster)}. Noise (-1) tasks get no cache entry."""
    by_cluster = collections.defaultdict(list)
    for task in tasks:
        cluster = task_to_cluster.get(task['id'])
        if cluster is not None and cluster != -1:
            by_cluster[cluster].append(task)

    caches = {}
    for cluster, cluster_tasks in by_cluster.items():
        gt_by_task = {
            t['id']: next(iter(answers[t['id']]['ground_truth'][0].keys()))
            for t in cluster_tasks
        }
        for t in cluster_tasks:
            others = [gt_by_task[o['id']] for o in cluster_tasks if o['id'] != t['id']]
            caches[t['id']] = set(others)
    return caches


def run_task(client, task, gt_name, native_full_tools, cache_names, tool_index):
    turns = task['question'][0] if isinstance(task['question'][0], list) else task['question']
    user_text = next((t['content'] for t in turns if t.get('role') == 'user'), '')

    result = dict(task_id=task['id'], gt_name=gt_name)

    prefix_full = build_menu_prefix(user_text, native_full_tools)
    text_full = client.free_generate(prefix_full)
    names_full = [fn['name'] for fn in native_full_tools]
    choice_full = best_match(text_full, names_full) if text_full else None
    result['native_full'] = dict(
        menu_size=len(names_full), free_text=text_full, choice=choice_full,
        correct=(choice_full == gt_name))

    if cache_names is not None:
        cache_tools = [tool_index[name] for name in cache_names if name in tool_index]
        if cache_tools:
            prefix_cache = build_menu_prefix(user_text, cache_tools)
            text_cache = client.free_generate(prefix_cache)
            names_cache = [fn['name'] for fn in cache_tools]
            choice_cache = best_match(text_cache, names_cache) if text_cache else None
            result['cluster_cache'] = dict(
                menu_size=len(names_cache), gt_in_cache=(gt_name in names_cache),
                free_text=text_cache, choice=choice_cache,
                correct=(choice_cache == gt_name))
        else:
            result['cluster_cache'] = dict(menu_size=0, gt_in_cache=False,
                                            free_text=None, choice=None, correct=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--api-base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tasks = read_jsonl(data_dir / 'questions.json')
    answers = {a['id']: a for a in read_jsonl(data_dir / 'answers.json')}
    tasks = [t for t in tasks if t['id'] in answers]

    task_to_cluster = load_clusters(args.clusters)
    tool_index = build_tool_schema_index(tasks)
    caches = build_loo_caches(tasks, answers, task_to_cluster)

    client = Client(args.api_base, args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    done = 0
    total = len(tasks)

    def _run(task):
        gt_name = next(iter(answers[task['id']]['ground_truth'][0].keys()))
        cache_names = caches.get(task['id'])  # None for noise tasks
        return run_task(client, task, gt_name, task['function'], cache_names, tool_index)

    with out_path.open('w', encoding='utf-8') as handle, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, task): task['id'] for task in tasks}
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
                if done % 50 == 0 or done == total:
                    print(f'[{done}/{total}] {task_id}', flush=True)

    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
