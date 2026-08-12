#!/usr/bin/env python3
"""Does giving the model a likelihood-selected SUBSET of a larger tool pool
beat giving it the WHOLE pool?

This is the causal question the forced-scoring pilot (score_candidates.py)
didn't test: that pilot only asked "can likelihood rank a task's own small
pre-filtered candidate list" (2-4 tools, already near ceiling). Here we
build a genuinely large, mostly-irrelevant-per-task tool POOL by unioning
many BFCL live_multiple tasks' candidate functions together (real function
reuse across live_multiple's 1053 tasks makes this natural -- 457 distinct
names across 1053 tasks), then for each task in the pool compare three
conditions with REAL unconstrained generation:

  full    -- show the model the entire pool
  topk    -- show only the top-K tools by forced-scoring likelihood
             (ranked against THIS task's query, no GT used)
  randomk -- show a random K-sized subset (same size as topk, no signal --
             controls for "smaller menu helps regardless of *which* tools
             are removed" vs "our selection specifically helps")

Matches this repo's `experiments/relevance_selector.py` / `predmask_curve.sh`
mask-axis methodology (build a no-GT predicted mask, measure the accuracy
recovery it buys vs. the full pool), with token likelihood as the selector
instead of query<->tool embedding similarity.

Pools are built with round-robin task assignment (task_idx % n_batches),
not consecutive slicing -- live_multiple groups domain-adjacent tasks
consecutively in the file (near-identical tool schemas back to back), so a
consecutive slice gives almost no distractor diversity. Round-robin scatters
that adjacency across batches instead.
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
    functions = '\n\n'.join(
        function_block(fn, i + 1) for i, fn in enumerate(tools))
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


class Client:
    def __init__(self, api_base, model, top_k=5, timeout=120):
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
            return None  # prompt too long for context -- caller skips
        resp.raise_for_status()
        return resp.json()['choices'][0]['text'].strip()


def rank_pool(client, prefix, pool_names, scoring_workers=12):
    """Score every candidate concurrently (each call is an independent HTTP
    round-trip -- I/O bound, so plain threads parallelize fine despite the
    GIL). Sequential scoring was the dominant cost in the first timed run
    (~24 pool tools x serial round-trips per task)."""
    scored = []
    with ThreadPoolExecutor(max_workers=min(scoring_workers, max(1, len(pool_names)))) as pool:
        futures = {pool.submit(client.score_continuation, prefix, ' ' + name): name
                   for name in pool_names}
        for future in as_completed(futures):
            name = futures[future]
            score = future.result()
            if score is not None:
                scored.append((name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


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


def round_robin_batches(tasks, n_batches):
    batches = [[] for _ in range(n_batches)]
    for i, task in enumerate(tasks):
        batches[i % n_batches].append(task)
    return [b for b in batches if b]


def run_task(client, task, pool, topk, rng):
    user_text = task['question'][0][0]['content']
    gt = task['_gt_name']
    pool_names = list(pool.keys())
    pool_tools = list(pool.values())

    score_prefix = build_menu_prefix(user_text, pool_tools)
    ranked = rank_pool(client, score_prefix, pool_names)
    if not ranked:
        return None
    topk_names = [name for name, _ in ranked[:topk]]
    remaining = [n for n in pool_names if n not in topk_names]
    randk_names = rng.sample(remaining, min(topk, len(remaining))) if remaining else []
    randk_names = randk_names[:topk] if len(randk_names) == topk else (
        randk_names + topk_names[:max(0, topk - len(randk_names))])

    conditions = dict(full=pool_names, topk=topk_names, randomk=randk_names)
    result = dict(task_id=task['id'], gt_name=gt, pool_size=len(pool_names),
                  rank_of_gt_in_pool=next(
                      (i + 1 for i, (n, _) in enumerate(ranked) if n == gt), None))

    for cond_name, names in conditions.items():
        tools = [pool[n] for n in names]
        gen_prefix = build_menu_prefix(user_text, tools)
        # free_generate returns None on a 400 (context overflow) as well as
        # on genuinely empty output -- both are "no usable answer", and only
        # the 'full' condition (large pool) is ever at real risk of the
        # former, so one call replaces the separate prompt_fits pre-check.
        text = client.free_generate(gen_prefix)
        choice = best_match(text, names) if text else None
        result[cond_name] = dict(
            gt_in_menu=(gt in names),
            menu_size=len(names),
            free_text=text,
            choice=choice,
            correct=(choice == gt),
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--api-base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--n-batches', type=int, default=150)
    ap.add_argument('--max-batches', type=int, default=None,
                     help='cap number of batches processed (for smoke tests)')
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    questions = read_jsonl(data_dir / 'questions.json')
    answers = {a['id']: a for a in read_jsonl(data_dir / 'answers.json')}
    skipped_no_answer = [t['id'] for t in questions if t['id'] not in answers]
    if skipped_no_answer:
        print(f'skipping {len(skipped_no_answer)} task(s) with no possible_answer entry: '
              f'{skipped_no_answer}')
    questions = [t for t in questions if t['id'] in answers]
    for task in questions:
        task['_gt_name'] = next(iter(answers[task['id']]['ground_truth'][0].keys()))

    batches = round_robin_batches(questions, args.n_batches)
    if args.max_batches:
        batches = batches[:args.max_batches]

    client = Client(args.api_base, args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for batch_idx, batch in enumerate(batches):
        pool = {}
        for task in batch:
            for fn in task['function']:
                pool.setdefault(fn['name'], fn)
        for task in batch:
            jobs.append((batch_idx, task, pool))

    lock = threading.Lock()
    done = 0
    total = len(jobs)

    def _run(job):
        batch_idx, task, pool = job
        rng = random.Random(f'{args.seed}-{task["id"]}')
        record = run_task(client, task, pool, args.topk, rng)
        if record is not None:
            record['batch_idx'] = batch_idx
        return record

    with out_path.open('w', encoding='utf-8') as handle, \
            ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
        futures = {pool_exec.submit(_run, job): job[1]['id'] for job in jobs}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                record = dict(task_id=task_id, error=str(exc))
            if record is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                handle.flush()
            with lock:
                done += 1
                if done % 20 == 0 or done == total:
                    print(f'[{done}/{total}] {task_id}', flush=True)

    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
