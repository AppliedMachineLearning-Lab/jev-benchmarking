"""Who answers a step request. Every backend returns the API's `answers` dict (question id -> answer).

All backends are cache-first: the response is looked up by the hash of (model, state, questions) in the
same SQLite cache the single-step suite uses, one database per model. A deterministic episode therefore
replays from the cache, and the Jev database stays the spend ledger.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jev_benchmarking.cache import ResponseCache, request_key
from jev_benchmarking.config import MODEL, TOTAL_BUDGET_USD, USD_PER_INPUT_TOKEN
from jev_benchmarking.runner import OVERHEAD_TOKENS, CHARS_PER_TOKEN


class BudgetExceeded(Exception):
    """The next request would push spend past the run cap or the ledger's total budget."""


@dataclass
class BackendStats:
    requests: int = 0
    cached: int = 0
    input_tokens: int = 0
    latency_s: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * USD_PER_INPUT_TOKEN


def estimate_tokens(state: Any, questions: dict) -> int:
    chars = len(json.dumps(state, ensure_ascii=False)) + len(json.dumps(questions, ensure_ascii=False))
    return OVERHEAD_TOKENS + int(chars / CHARS_PER_TOKEN)


class Backend:
    """Subclasses implement `_answer`; `ask` adds caching and bookkeeping."""

    model: str = ""
    pays: bool = False  # True only for the TypeSafe API

    def __init__(self, cache: ResponseCache | None = None, max_cost_usd: float = 0.0):
        self.cache = cache
        self.max_cost_usd = max_cost_usd
        self.stats = BackendStats()
        self._ledger_usd: float | None = None

    def ask(self, state: Any, questions: dict) -> tuple[dict, dict]:
        """Return (answers, meta). meta: key, cached, input_tokens, latency_s."""
        key = request_key(self.model, state, questions)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.stats.cached += 1
                tokens = int((hit.get("usage") or {}).get("input_tokens") or 0)
                return hit["answers"], {"key": key, "cached": True, "input_tokens": tokens, "latency_s": 0.0}
        if self.pays:
            est = estimate_tokens(state, questions)
            if self.stats.requests:  # never assume less than the average real request so far
                est = max(est, -(-self.stats.input_tokens // self.stats.requests))
            if (self.stats.input_tokens + est) * USD_PER_INPUT_TOKEN > self.max_cost_usd:
                raise BudgetExceeded(f"run cap ${self.max_cost_usd:.2f} reached (${self.stats.cost_usd:.4f} spent)")
            if self._ledger_usd is None:  # read the ledger once; afterwards add this run's own spend
                self._ledger_usd = self.cache.spent_usd() if self.cache is not None else 0.0
            if self._ledger_usd + (self.stats.input_tokens + est) * USD_PER_INPUT_TOKEN > TOTAL_BUDGET_USD:
                raise BudgetExceeded(f"total budget ${TOTAL_BUDGET_USD:.2f} in the ledger reached")
        t0 = time.monotonic()
        body = self._answer(state, questions)
        latency = time.monotonic() - t0
        tokens = int((body.get("usage") or {}).get("input_tokens") or 0)
        self.stats.requests += 1
        self.stats.input_tokens += tokens
        self.stats.latency_s += latency
        if self.cache is not None:
            self.cache.put(key, "sequential", body, estimate_tokens(state, questions), latency)
        return body["answers"], {"key": key, "cached": False, "input_tokens": tokens, "latency_s": latency}

    def _answer(self, state: Any, questions: dict) -> dict:
        raise NotImplementedError


class JevBackend(Backend):
    """TypeSafe API (synchronous; a sequential episode cannot pipeline its own steps).

    Episodes are independent, so throughput comes from running several episodes in parallel processes
    (`scripts/run_sequential.py --shard i --num-shards n`), not from concurrency inside an episode.
    """

    pays = True

    def __init__(self, cache: ResponseCache | None = None, max_cost_usd: float = 0.5, model: str = MODEL, transport=None):
        super().__init__(cache, max_cost_usd)
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        from jev_benchmarking.config import load_api_key
        from jev_benchmarking.runner import RawResponse

        self.model = model
        self._raw = RawResponse
        self.client = TypeSafeClient(
            api_key=load_api_key(), model=model, timeout=60.0,
            retry=RetryPolicy(max_retries=6, backoff_max=30.0, timeout=180.0), transport=transport,
        )

    def _answer(self, state: Any, questions: dict) -> dict:
        return self.client.system_one(state, questions, response_model=self._raw).model_dump()


class OpenModelBackend(Backend):
    """Open-weight LLM scored exactly as in the single-step suite: one forward pass per question, the
    next-token distribution restricted to the option codes (see `openmodel/`)."""

    def __init__(self, model_id: str, cache: ResponseCache | None = None, device: str = "cuda", **scorer_kw):
        super().__init__(cache)
        from jev_benchmarking.openmodel.runner import model_tag
        from jev_benchmarking.openmodel.scorer import HFScorer

        self.model = model_tag(model_id)
        self.scorer = HFScorer(model_id, device=device, **scorer_kw)

    def _answer(self, state: Any, questions: dict) -> dict:
        from jev_benchmarking.openmodel.prompts import render, to_answer
        from jev_benchmarking.openmodel.runner import _normalize

        rendered = {qid: render(state, q, self.scorer.is_single_token) for qid, q in questions.items()}
        qids = list(rendered)
        results = self.scorer.score([self.scorer.prompt_ids(rendered[q].text) for q in qids], [rendered[q].codes for q in qids])
        answers, mass, tokens = {}, {}, 0
        for qid, res in zip(qids, results):
            probs, valid = _normalize(res.logprobs)
            answers[qid] = to_answer(rendered[qid], probs, questions[qid])
            mass[qid] = valid
            tokens += res.n_prompt_tokens
        return {"model": self.model, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0},
                "diagnostics": {"valid_mass": mass}}


class FunctionBackend(Backend):
    """A Python policy in the API's response format, for baselines (random, expert, basic strategy) and
    tests. `fn(state, questions) -> answers`. Not cached by default: it costs nothing."""

    def __init__(self, name: str, fn: Callable[[Any, dict], dict], cache: ResponseCache | None = None):
        super().__init__(cache)
        self.model = f"scripted:{name}"
        self.fn = fn

    def _answer(self, state: Any, questions: dict) -> dict:
        return {"model": self.model, "answers": self.fn(state, questions), "usage": {"input_tokens": 0}}


# ----------------------------------------------------------------------------------------------------
# Helpers to build answers in the API's format (used by scripted policies and tests).


def choice_answer(probs: dict[str, float]) -> dict:
    z = sum(probs.values())
    probs = {k: v / z for k, v in probs.items()}
    k = len(probs)
    top = max(probs, key=probs.get)
    conf = max(0.0, min(1.0, (k * probs[top] - 1) / (k - 1))) if k > 1 else 1.0
    return {"type": "choice", "choice": top, "probabilities": probs, "confidence": conf}


def noul_answer(p: float) -> dict:
    return {"type": "noul", "noul": float(p)}
