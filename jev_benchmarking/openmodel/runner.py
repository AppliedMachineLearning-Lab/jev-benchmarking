"""Run an open-weight model over benchmark tasks and write Jev-format responses to JSONL (resumable, shardable).

Each Jev request (one state, k questions) becomes k prompts sharing the same state; every question is
answered independently, as in Jev. The output lines are imported into the response cache with
`scripts/import_responses.py`, after which all evaluation code works unchanged.
"""

import hashlib
import json
import logging
import math
import time
from collections.abc import Iterator
from pathlib import Path

from jev_benchmarking.cache import request_key
from jev_benchmarking.openmodel.prompts import render, to_answer
from jev_benchmarking.tasks.base import Example, Task

log = logging.getLogger(__name__)


def model_tag(model_id: str) -> str:
    return f"hf:{model_id}"


def shard_of(key: str, num_shards: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % num_shards


def done_keys(out_dir: Path) -> set[str]:
    """Keys already written by any shard (so re-sharding never redoes work)."""
    keys = set()
    for f in out_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                try:
                    keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass  # a line cut off by a killed job is simply redone
    return keys


def iter_items(tasks: list[Task], split: str, limit: int | None, tag: str, shard: int, num_shards: int, skip: set[str]) -> Iterator[tuple[str, Example, str]]:
    seen = set(skip)
    for task in tasks:
        if task.splits.get(split) is None:
            continue
        for e in task.examples(split, limit=limit):
            key = request_key(tag, e.state, e.questions)
            if key in seen or shard_of(key, num_shards) != shard:
                continue
            seen.add(key)  # identical requests (duplicate rows) are scored once
            yield task.name, e, key


def _normalize(logprobs: list[float]) -> tuple[list[float], float]:
    m = max(logprobs)
    w = [math.exp(lp - m) for lp in logprobs]
    z = sum(w)
    return [x / z for x in w], math.exp(m) * z  # renormalized probabilities, total mass on the codes


def run(scorer, model_id: str, tasks: list[Task], split: str, limit: int | None, out_dir: Path,
        shard: int = 0, num_shards: int = 1, chunk_examples: int = 512) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = model_tag(model_id)
    out_path = out_dir / f"shard{shard:03d}-of-{num_shards:03d}.jsonl"
    items = iter_items(tasks, split, limit, tag, shard, num_shards, done_keys(out_dir))
    written = 0
    with out_path.open("a") as out:
        while True:
            chunk = [x for _, x in zip(range(chunk_examples), items)]
            if not chunk:
                break
            t0 = time.monotonic()
            prompts, codes, where = [], [], []
            rendered = []
            for ci, (task_name, e, key) in enumerate(chunk):
                per_q = {}
                for qid, q in e.questions.items():
                    rq = render(e.state, q, scorer.is_single_token)
                    per_q[qid] = rq
                    prompts.append(scorer.prompt_ids(rq.text))
                    codes.append(rq.codes)
                    where.append((ci, qid))
                rendered.append(per_q)
            scores = scorer.score(prompts, codes)
            answers = [dict() for _ in chunk]
            mass = [dict() for _ in chunk]
            tokens = [0] * len(chunk)
            for (ci, qid), res in zip(where, scores):
                probs, valid = _normalize(res.logprobs)
                q = chunk[ci][1].questions[qid]
                answers[ci][qid] = to_answer(rendered[ci][qid], probs, q)
                mass[ci][qid] = valid
                tokens[ci] += res.n_prompt_tokens
            elapsed = time.monotonic() - t0
            for ci, (task_name, e, key) in enumerate(chunk):
                out.write(json.dumps({
                    "key": key, "task": task_name, "uid": e.uid,
                    "latency_s": elapsed / len(chunk),
                    "response": {
                        "model": tag, "answers": answers[ci],
                        "usage": {"input_tokens": tokens[ci], "output_tokens": 0},
                        "diagnostics": {"valid_mass": mass[ci]},
                    },
                }, ensure_ascii=False) + "\n")
            out.flush()
            written += len(chunk)
            log.info("shard %d/%d: +%d examples (%d prompts, %d tokens) in %.1fs, %d total",
                     shard, num_shards, len(chunk), len(prompts), sum(tokens), elapsed, written)
    return written
