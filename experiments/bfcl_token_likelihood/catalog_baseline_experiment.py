#!/usr/bin/env python3
"""The "real" baseline: can the model find the right tool when it's buried
in a large, mostly-irrelevant sample drawn from the WHOLE tool catalog
(457 distinct tools across all of BFCL v1 live_multiple), not just a single
cluster's 24 tools?

457 tool schemas don't fit in this model's 8192-token context at all (tested
directly: even 50 tools alone overflows). So instead of literally cramming
"everything" into one prompt (impossible), the baseline is: for each task,
draw N=40 tools at random from the full 457-tool catalog (the largest size
that reliably fits in context with room for the query and generation),
FORCING the task's own ground-truth tool into the draw (otherwise this tests
sampling luck, not selection quality). Repeated with several random seeds
per task and averaged, so no single draw's difficulty/ease dominates.

Two conditions, real unconstrained generation:

  catalog_baseline      -- all N=40 catalog-drawn tools shown
  likelihood_topk       -- only the top-K (forced-scoring, no GT used) of
                            those 40 shown

This is the catalog-scale generalization of pool_subset_experiment.py's
full-vs-topk-vs-randomk design (which drew "clutter" from only ~7
round-robin-batched tasks, not the whole 457-tool catalog).

Concurrency: everything is a FLAT job list submitted to ONE ThreadPoolExecutor
per phase (scoring, then generation) -- earlier scripts in this investigation
that nested a nested per-item ThreadPoolExecutor inside an outer one lost
most of their intended parallelism (or, worse, multiplied it far past what
the shared vLLM server could handle and made it time out). Never nest pools.
"""
import argparse
import json
import math
import random
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


def draw_catalog_pool(gt_name, catalog_names, n_tools, seed):
    rng = random.Random(seed)
    others = [n for n in catalog_names if n != gt_name]
    draw = rng.sample(others, min(n_tools - 1, len(others)))
    pool_names = draw + [gt_name]
    rng.shuffle(pool_names)
    return pool_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--api-base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--n-tools', type=int, default=40)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--n-tasks', type=int, default=100,
                     help='sample this many tasks, spread across the dataset')
    ap.add_argument('--seeds-per-task', type=int, default=1)
    ap.add_argument('--scoring-workers', type=int, default=12)
    ap.add_argument('--gen-workers', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    all_tasks = read_jsonl(data_dir / 'questions.json')
    answers = {a['id']: a for a in read_jsonl(data_dir / 'answers.json')}
    all_tasks = [t for t in all_tasks if t['id'] in answers]

    catalog = {}
    for task in all_tasks:
        for fn in task['function']:
            catalog.setdefault(fn['name'], fn)
    catalog_names = list(catalog.keys())
    print(f'full catalog: {len(catalog_names)} distinct tools')

    stride = max(1, len(all_tasks) // args.n_tasks)
    sampled = all_tasks[::stride][:args.n_tasks]
    print(f'sampled {len(sampled)} tasks (stride {stride}) spread across the dataset')

    # Build one job per (task, seed): a fixed draw of n_tools catalog tools
    # (GT forced in), computed up front so scoring and generation both work
    # off the exact same draw.
    jobs = []
    for task in sampled:
        gt_name = next(iter(answers[task['id']]['ground_truth'][0].keys()))
        for s in range(args.seeds_per_task):
            pool_names = draw_catalog_pool(gt_name, catalog_names, args.n_tools, s)
            jobs.append(dict(task=task, gt_name=gt_name, seed=s, pool_names=pool_names,
                              user_text=user_text_of(task)))
    print(f'{len(jobs)} (task, seed) draws')

    client = Client(args.api_base, args.model)
    lock = threading.Lock()

    # ---- Phase 1: score every (job, tool) pair, one flat pool. ----
    score_matrix = {i: {} for i in range(len(jobs))}
    prefixes = {i: build_menu_prefix(job['user_text'], [catalog[n] for n in job['pool_names']])
                for i, job in enumerate(jobs)}
    score_jobs = [(i, name) for i, job in enumerate(jobs) for name in job['pool_names']]
    print(f'scoring {len(score_jobs)} (job, tool) pairs with {args.scoring_workers} workers...')
    done = 0
    with ThreadPoolExecutor(max_workers=args.scoring_workers) as pool_exec:
        futures = {
            pool_exec.submit(client.score_continuation, prefixes[i], ' ' + name): (i, name)
            for i, name in score_jobs
        }
        for future in as_completed(futures):
            i, name = futures[future]
            score = future.result()
            if score is not None:
                score_matrix[i][name] = score
            with lock:
                done += 1
                if done % 500 == 0 or done == len(score_jobs):
                    print(f'  scored [{done}/{len(score_jobs)}]')

    # ---- Phase 2: real generation, one flat pool, 2 calls per job. ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _run_generation(i):
        job = jobs[i]
        task, gt_name, pool_names = job['task'], job['gt_name'], job['pool_names']
        scores = score_matrix[i]
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        topk_names = [name for name, _ in ranked[:args.topk]]

        pool_tools = [catalog[n] for n in pool_names]
        text_baseline = client.free_generate(prefixes[i])
        choice_baseline = best_match(text_baseline, pool_names) if text_baseline else None

        topk_tools = [catalog[n] for n in topk_names]
        prefix_topk = build_menu_prefix(job['user_text'], topk_tools)
        text_topk = client.free_generate(prefix_topk)
        choice_topk = best_match(text_topk, topk_names) if text_topk else None

        return dict(
            task_id=task['id'], gt_name=gt_name, seed=job['seed'], pool_size=len(pool_names),
            rank_of_gt=next((idx + 1 for idx, (n, _) in enumerate(ranked) if n == gt_name), None),
            catalog_baseline=dict(menu_size=len(pool_names), free_text=text_baseline,
                                   choice=choice_baseline, correct=(choice_baseline == gt_name)),
            likelihood_topk=dict(menu_size=len(topk_names), gt_in_topk=(gt_name in topk_names),
                                  free_text=text_topk, choice=choice_topk,
                                  correct=(choice_topk == gt_name)),
        )

    print(f'running generation for {len(jobs)} jobs with {args.gen_workers} workers...')
    done = 0
    with out_path.open('w', encoding='utf-8') as handle, \
            ThreadPoolExecutor(max_workers=args.gen_workers) as pool_exec:
        futures = {pool_exec.submit(_run_generation, i): jobs[i]['task']['id']
                   for i in range(len(jobs))}
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
                if done % 20 == 0 or done == len(jobs):
                    print(f'  generated [{done}/{len(jobs)}]')

    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
