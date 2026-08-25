"""Per-token likelihood and entropy logging for OpenAI-compatible models.

The logger is deliberately independent of a particular serving backend.  It
accepts the OpenAI ``choices[0].logprobs`` response shape used by vLLM and
recent LMDeploy releases, and also understands the legacy completions arrays.
"""

import json
import math
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


def _finite_float(value: Any, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


class TokenMetricsLogger:
    """Append one JSONL record for every assistant generation with logprobs."""

    def __init__(self,
                 path: str,
                 top_k: int = 20,
                 vocab_size: Optional[int] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # OpenAI-compatible chat endpoints conventionally cap top_logprobs at 20.
        self.top_k = max(0, min(20, int(top_k)))
        self.vocab_size = int(vocab_size) if vocab_size else None
        self._lock = threading.Lock()
        self._task_id = None
        self._task_sequence = 0
        self._generation_index = 0
        self._tool_names = set()
        self._executed_tool_path = []

    def set_task_id(self, task_id: Any) -> None:
        """Start a new sample and reset its ReAct generation/path counters."""
        with self._lock:
            self._task_id = task_id
            self._task_sequence += 1
            self._generation_index = 0
            self._executed_tool_path = []

    def set_tool_names(self, tool_names: Iterable[str]) -> None:
        self._tool_names = {str(name) for name in tool_names}

    @staticmethod
    def _standard_tokens(payload: Any) -> list:
        if not isinstance(payload, dict):
            return []
        content = payload.get('content')
        if isinstance(content, list):
            return content

        # Legacy /v1/completions representation.
        tokens = payload.get('tokens')
        token_logprobs = payload.get('token_logprobs')
        top_logprobs = payload.get('top_logprobs')
        if not isinstance(tokens, list) or not isinstance(token_logprobs, list):
            return []
        rows = []
        for idx, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
            alternatives = top_logprobs[idx] if isinstance(
                top_logprobs, list) and idx < len(top_logprobs) else []
            if isinstance(alternatives, dict):
                alternatives = [
                    dict(token=alt_token, logprob=alt_logprob)
                    for alt_token, alt_logprob in alternatives.items()
                ]
            rows.append(
                dict(token=token,
                     logprob=logprob,
                     top_logprobs=alternatives or []))
        return rows

    def _token_row(self, position: int, item: dict) -> dict:
        sampled_logprob = _finite_float(item.get('logprob'), float('nan'))
        alternatives = item.get('top_logprobs') or []
        if isinstance(alternatives, dict):
            alternatives = [
                dict(token=token, logprob=logprob)
                for token, logprob in alternatives.items()
            ]

        top_rows = []
        captured_mass = 0.0
        entropy_topk = 0.0
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                continue
            logprob = _finite_float(alternative.get('logprob'),
                                    float('-inf'))
            probability = math.exp(logprob) if math.isfinite(logprob) else 0.0
            captured_mass += probability
            if probability > 0.0:
                entropy_topk -= probability * logprob
            top_rows.append({
                'token': alternative.get('token'),
                'logprob': logprob,
                'probability': probability,
            })

        tail_mass = max(0.0, 1.0 - min(1.0, captured_mass))
        entropy_lower = entropy_topk
        if tail_mass > 0.0:
            entropy_lower -= tail_mass * math.log(tail_mass)
        entropy_upper = None
        if self.vocab_size:
            if tail_mass > 0.0:
                unseen = max(1, self.vocab_size - len(top_rows))
                entropy_upper = (entropy_topk - tail_mass *
                                 math.log(tail_mass / unseen))
            else:
                entropy_upper = entropy_topk

        return {
            'position': position,
            'token': item.get('token'),
            'sampled_logprob': sampled_logprob,
            'sampled_probability': (math.exp(sampled_logprob)
                                    if math.isfinite(sampled_logprob) else None),
            'top_logprobs': top_rows,
            'captured_probability_mass': captured_mass,
            'tail_probability_mass': tail_mass,
            'entropy_topk_nats': entropy_topk,
            'entropy_lower_bound_nats': entropy_lower,
            'entropy_upper_bound_nats': entropy_upper,
        }

    def _classify(self, text: str) -> tuple:
        if 'Final Answer:' in text:
            phase = ('final_after_all_tools' if self._executed_tool_path else
                     'final_without_tools')
            return phase, []

        matches = re.findall(r'(?:^|\n)Action:\s*([^\r\n]+)', text)
        has_arguments = re.search(r'(?:^|\n)Action Input:\s*', text) is not None
        proposed = matches[-1].strip() if matches and has_arguments else None
        if proposed and (not self._tool_names or proposed in self._tool_names):
            return 'before_tool', [proposed]

        phase = ('unparsed_after_tool' if self._executed_tool_path else
                 'unparsed_without_tools')
        return phase, []

    def log(self, choice: dict, assistant_text: str) -> bool:
        """Persist a response. Returns False when the server sent no logprobs."""
        raw_tokens = self._standard_tokens(choice.get('logprobs'))
        if not raw_tokens:
            return False
        token_rows = [
            self._token_row(position, item)
            for position, item in enumerate(raw_tokens)
            if isinstance(item, dict)
        ]
        if not token_rows:
            return False

        with self._lock:
            phase, proposed_tools = self._classify(assistant_text)
            finite_logprobs = [
                row['sampled_logprob'] for row in token_rows
                if math.isfinite(row['sampled_logprob'])
            ]
            entry = {
                'timestamp': datetime.now().isoformat(),
                'task_id': self._task_id,
                'task_sequence': self._task_sequence,
                'generation_index': self._generation_index,
                'phase': phase,
                'executed_tool_path_before_turn':
                list(self._executed_tool_path),
                'after_tool': (self._executed_tool_path[-1]
                               if self._executed_tool_path else None),
                'proposed_tools': proposed_tools,
                'finish_reason': choice.get('finish_reason'),
                'assistant_text': assistant_text,
                'token_count': len(token_rows),
                'mean_sampled_logprob_nats': (
                    sum(finite_logprobs) / len(finite_logprobs)
                    if finite_logprobs else None),
                'mean_entropy_topk_nats': sum(
                    row['entropy_topk_nats']
                    for row in token_rows) / len(token_rows),
                'mean_entropy_lower_bound_nats': sum(
                    row['entropy_lower_bound_nats']
                    for row in token_rows) / len(token_rows),
                'mean_captured_probability_mass': sum(
                    row['captured_probability_mass']
                    for row in token_rows) / len(token_rows),
                'tokens': token_rows,
            }
            upper = [row['entropy_upper_bound_nats'] for row in token_rows]
            entry['mean_entropy_upper_bound_nats'] = (
                sum(upper) / len(upper)
                if upper and all(value is not None for value in upper) else None)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')

            self._generation_index += 1
            if phase == 'before_tool':
                self._executed_tool_path.extend(proposed_tools)
        return True
