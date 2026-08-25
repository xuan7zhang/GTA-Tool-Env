"""GTA-Atomic (original 229-task GTA) eval config, environment-variable driven.

This is gta2-envlab's canonical config (symlinked from gta2-envlab/configs/).
Everything a sweep varies comes from env vars, so one config file serves all
(env_variant, probe_mode, seed) runs:

  GTA_LLM_URL      OpenAI-compatible chat endpoint    (default local lmdeploy)
  GTA_MODEL_NAME   served model name                  (default qwen2.5-7b-instruct)
  GTA_TOOLSERVER   tool server (usually the PROXY)    (default http://127.0.0.1:16281)
  GTA_TOOLMETA     toolmeta.json path — use envgen variants for masked/rewritten envs
  GTA_EVAL_MODES   comma list of {step,end}           (default both)
  GTA_MAX_TURN     ReAct max turns                    (default 10)
  GTA_TOKEN_METRICS_LOG  optional JSONL path for sampled-token logprobs
  GTA_LOGPROB_TOP_K      returned alternatives/token  (default 20)
  GTA_LOGPROB_VOCAB_SIZE optional vocab size for entropy upper bound

Metrics: step -> InstAcc/ToolAcc/ArgAcc/SummAcc; end -> AnsAcc + P/O/L/C F1.
"""
import json
import os

from lagent.agents import ReAct

from opencompass.models import OpenAI
from opencompass.models.lagent import LagentAgent
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.datasets.gta_bench import GTABenchDataset, GTABenchEvaluator

_reader_cfg = dict(
    input_columns=["dialogs", "resources"],
    output_column="gt_answer",
    train_split='test',
    test_split='test')

_modes = os.getenv('GTA_EVAL_MODES', 'step,end').split(',')
_stop = os.getenv('GTA_STOP', '["\\nResponse:", "<|im_end|>"]')
try:
    _stop = json.loads(_stop)
except json.JSONDecodeError:
    pass

datasets = []
if 'step' in _modes:
    datasets.append(dict(
        abbr="gta_bench_step",
        type=GTABenchDataset,
        path="data/gta_dataset",
        reader_cfg=_reader_cfg,
        infer_cfg=dict(
            prompt_template=dict(type=PromptTemplate, template="""{questions}"""),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=AgentInferencer, infer_mode='every_with_gt'),
        ),
        eval_cfg=dict(evaluator=dict(type=GTABenchEvaluator, mode='every_with_gt')),
    ))
if 'end' in _modes:
    datasets.append(dict(
        abbr="gta_bench_end",
        type=GTABenchDataset,
        path="data/gta_dataset",
        reader_cfg=_reader_cfg,
        infer_cfg=dict(
            prompt_template=dict(type=PromptTemplate, template="""{questions}"""),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=AgentInferencer, infer_mode='every'),
        ),
        eval_cfg=dict(evaluator=dict(type=GTABenchEvaluator, mode='every')),
    ))

models = [
    dict(
        abbr=os.getenv('GTA_MODEL_NAME', 'qwen2.5-7b-instruct'),
        type=LagentAgent,
        agent_type=ReAct,
        max_turn=int(os.getenv('GTA_MAX_TURN', '10')),
        llm=dict(
            type=OpenAI,
            path=os.getenv('GTA_MODEL_NAME', 'qwen2.5-7b-instruct'),
            key='EMPTY',
            openai_api_base=os.getenv(
                'GTA_LLM_URL', 'http://127.0.0.1:12580/v1/chat/completions'),
            query_per_second=int(os.getenv('GTA_QPS', '2')),
            max_seq_len=int(os.getenv('GTA_MAX_SEQ_LEN', '32768')),
            retry=5,
            stop=_stop,
            token_metrics_log=os.getenv('GTA_TOKEN_METRICS_LOG') or None,
            logprob_top_k=int(os.getenv('GTA_LOGPROB_TOP_K', '20')),
            logprob_vocab_size=(
                int(os.getenv('GTA_LOGPROB_VOCAB_SIZE'))
                if os.getenv('GTA_LOGPROB_VOCAB_SIZE') else None),
        ),
        tool_server=os.getenv('GTA_TOOLSERVER', 'http://127.0.0.1:16281'),
        tool_meta=os.getenv('GTA_TOOLMETA', 'data/gta_dataset/toolmeta.json'),
        batch_size=int(os.getenv('GTA_BATCH', '8')),
    ),
]

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=1),
    runner=dict(
        type=LocalRunner,
        max_num_workers=int(os.getenv('GTA_INFER_WORKERS', '1')),
        task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=1),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLEvalTask)),
)

# OpenCompass re-dumps this config; a bare module in the namespace serializes
# as `os=<module ...>` and breaks re-parsing. Drop it after use.
_modes = None
_stop = None
del json
del os
