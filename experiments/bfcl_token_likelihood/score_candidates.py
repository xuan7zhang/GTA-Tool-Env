#!/usr/bin/env python3
"""BFCL v1 pilot: does per-candidate token likelihood/entropy recover the
ground-truth tool out of a task's candidate set?

Method ("forced-scoring"): for each task, build one shared ReAct-style prefix
ending in "Action:" (same convention this repo's GTA/lagent agent already
uses -- see GTA/opencompass/opencompass/models/token_metrics.py's phase
classifier), then for every candidate function in the task, score
P(candidate_name | prefix) by hitting the live vLLM OpenAI-compatible
/v1/completions endpoint with echo=True, max_tokens=0, logprobs=K. That
returns per-token logprobs for the whole echoed prompt (vLLM/OpenAI
completions convention), from which we slice out just the tokens belonging
to the appended candidate name.

This is the tool-subset analogue of experiments/relevance_selector.py (which
ranks candidates by query<->tool embedding similarity instead of token
likelihood) -- same "rank all candidates, score against GT" evaluation
shape, different signal.

Also scores one natural (unconstrained) generation per task as a cheap
context baseline: does the model already get this right without any forced
scoring at all?

Usage:
    python score_candidates.py \
        --data-dir runtime/bfcl_v1_pilot_data \
        --api-base http://127.0.0.1:8013/v1 \
        --model gpt-qwen3-vl-8b \
        --n-tasks 80 \
        --out runtime/bfcl_token_likelihood_pilot_20260811/scored.jsonl
"""
import argparse
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def read_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def function_block(fn, idx, display_name=None):
    params = json.dumps(fn.get('parameters', {}), ensure_ascii=False)
    return (f"Function {idx}: {display_name or fn['name']}\n"
            f"Description: {fn.get('description', '')}\n"
            f"Parameters: {params}")


def anonymized_names(task):
    """func_1..func_N in the same order as task['function'] -- description
    and params are untouched, only the literal name string is hidden. Used
    for the name-blinding ablation: if the likelihood signal survives this,
    it isn't just matching the query text against the tool's literal name."""
    return {fn['name']: f'func_{i + 1}' for i, fn in enumerate(task['function'])}


def build_prefix(task, anonymize=False):
    """Shared context every candidate is scored against. Ends in 'Action:'
    with no trailing space -- the space + candidate name is appended per
    candidate so the character offset split is unambiguous."""
    turns = task['question'][0] if isinstance(task['question'][0], list) else task['question']
    user_text = '\n'.join(t['content'] for t in turns if t.get('role') == 'user')
    name_map = anonymized_names(task) if anonymize else {}
    functions = '\n\n'.join(
        function_block(fn, i + 1, name_map.get(fn['name']))
        for i, fn in enumerate(task['function']))
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


def _position_entropy(top_logprobs_dict):
    """Same shape as token_metrics.py's per-token entropy: top-k entropy plus
    a tail-mass lower-bound correction for whatever probability mass isn't in
    the returned alternatives."""
    if not top_logprobs_dict:
        return 0.0
    captured = 0.0
    entropy_topk = 0.0
    for logprob in top_logprobs_dict.values():
        logprob = _finite(logprob, float('-inf'))
        prob = math.exp(logprob) if math.isfinite(logprob) else 0.0
        captured += prob
        if prob > 0.0:
            entropy_topk -= prob * logprob
    tail = max(0.0, 1.0 - min(1.0, captured))
    if tail > 0.0:
        entropy_topk -= tail * math.log(tail)
    return entropy_topk


class Client:
    def __init__(self, api_base, model, top_k=20, timeout=120):
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.top_k = top_k
        self.timeout = timeout
        self._session = requests.Session()

    def score_continuation(self, prefix, continuation):
        """Return (mean_logprob, sum_logprob, mean_entropy_nats, n_tokens)
        for the tokens of `continuation` appended right after `prefix`."""
        full_text = prefix + continuation
        resp = self._session.post(
            f'{self.api_base}/completions',
            json=dict(model=self.model, prompt=full_text, max_tokens=0,
                      echo=True, logprobs=self.top_k),
            timeout=self.timeout)
        resp.raise_for_status()
        choice = resp.json()['choices'][0]
        logprobs = choice['logprobs']
        offsets = logprobs['text_offset']
        token_logprobs = logprobs['token_logprobs']
        top_logprobs = logprobs['top_logprobs']
        boundary = len(prefix)
        tail_logprobs, tail_entropy = [], []
        for offset, logprob, top in zip(offsets, token_logprobs, top_logprobs):
            if offset is None or offset < boundary:
                continue
            tail_logprobs.append(_finite(logprob, float('nan')))
            tail_entropy.append(_position_entropy(top))
        if not tail_logprobs:
            return None
        finite_lp = [v for v in tail_logprobs if math.isfinite(v)]
        mean_lp = sum(finite_lp) / len(finite_lp) if finite_lp else None
        sum_lp = sum(finite_lp) if finite_lp else None
        mean_ent = sum(tail_entropy) / len(tail_entropy) if tail_entropy else None
        return dict(mean_logprob=mean_lp, sum_logprob=sum_lp,
                    mean_entropy_nats=mean_ent, n_tokens=len(tail_logprobs))

    def free_generate(self, prefix, max_tokens=24):
        """Unconstrained baseline: let the model finish 'Action:' on its own."""
        resp = self._session.post(
            f'{self.api_base}/completions',
            json=dict(model=self.model, prompt=prefix, max_tokens=max_tokens,
                      temperature=0.0, stop=['\n']),
            timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()['choices'][0]['text']


def parse_free_choice(text, candidate_names):
    """Best-effort extraction of which candidate name (if any) the free
    generation named first."""
    text = text.strip()
    best = None
    for name in candidate_names:
        idx = text.find(name)
        if idx != -1 and (best is None or idx < best[1]):
            best = (name, idx)
    return best[0] if best else None


def softmax_entropy_nats(logprobs):
    """Entropy (nats) of softmax(logprobs) treated as unnormalized logits --
    i.e. entropy over 'which candidate does the model favor', not entropy
    over individual tokens."""
    finite = [v for v in logprobs if v is not None]
    if len(finite) < 2:
        return 0.0
    m = max(finite)
    exps = [math.exp(v - m) for v in finite]
    z = sum(exps)
    probs = [e / z for e in exps]
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def score_task(client, task, answer, anonymize=False):
    prefix = build_prefix(task, anonymize=anonymize)
    gt_name = next(iter(answer['ground_truth'][0].keys()))
    name_map = anonymized_names(task) if anonymize else {}
    candidates = [fn['name'] for fn in task['function']]
    score_names = [name_map.get(name, name) for name in candidates]

    scored = []
    for real_name, score_name in zip(candidates, score_names):
        result = client.score_continuation(prefix, ' ' + score_name)
        if result is None:
            continue
        result['name'] = real_name
        result['is_gt'] = (real_name == gt_name)
        scored.append(result)
    if not scored:
        return None

    ranked = sorted(scored, key=lambda r: r['mean_logprob'], reverse=True)
    rank_of_gt = next((i + 1 for i, r in enumerate(ranked) if r['is_gt']), None)

    free_text = client.free_generate(prefix)
    free_choice_raw = parse_free_choice(free_text, score_names)
    free_choice = (candidates[score_names.index(free_choice_raw)]
                   if free_choice_raw in score_names else None)

    return dict(
        task_id=task['id'],
        gt_name=gt_name,
        n_candidates=len(candidates),
        candidates=scored,
        rank_of_gt=rank_of_gt,
        top1_correct=(rank_of_gt == 1),
        reciprocal_rank=(1.0 / rank_of_gt if rank_of_gt else 0.0),
        candidate_entropy_nats=softmax_entropy_nats(
            [r['mean_logprob'] for r in scored]),
        max_entropy_nats=math.log(len(candidates)),
        free_generation_text=free_text.strip(),
        free_choice=free_choice,
        free_choice_correct=(free_choice == gt_name),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--api-base', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--n-tasks', type=int, default=80)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out', required=True)
    ap.add_argument('--anonymize', action='store_true',
                     help='Replace function names with func_1..func_N '
                          '(description/params kept) -- name-blinding '
                          'ablation to check the signal is not just '
                          'lexical match between query and tool name.')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    questions = read_jsonl(data_dir / 'questions.json')[:args.n_tasks]
    answers = {a['id']: a for a in read_jsonl(data_dir / 'answers.json')}

    client = Client(args.api_base, args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    done = 0
    total = len(questions)

    def _run(task):
        answer = answers[task['id']]
        return score_task(client, task, answer, anonymize=args.anonymize)

    with out_path.open('w', encoding='utf-8') as handle, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, task): task['id'] for task in questions}
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
                print(f'[{done}/{total}] {task_id}', flush=True)

    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
