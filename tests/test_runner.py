"""Runner failure handling against a fake API (httpx2 MockTransport): nothing here touches the network."""

import asyncio
import json

import httpx2
import pytest
from typesafe_sdk import RetryPolicy

from jev_benchmarking.cache import ResponseCache
from jev_benchmarking.runner import CIRCUIT_BREAKER, FatalAPIError, run_examples
from jev_benchmarking.tasks.base import Example, noul

NO_RETRY = RetryPolicy(max_retries=0)


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


def examples(n: int) -> list[Example]:
    return [Example(str(i), "", {"text": f"t{i}"}, {"answer": noul("Is it?")}, {"answer": True}) for i in range(n)]


def ok_body(tokens: int = 100) -> dict:
    return {"model": "jev-1.13.0", "answers": {"answer": {"type": "noul", "noul": 0.9}}, "usage": {"input_tokens": tokens, "output_tokens": 5}}


def transport(handler):
    return httpx2.MockTransport(handler)


def run(cache, exs, handler, **kw):
    kw.setdefault("max_cost_usd", 1.0)
    return asyncio.run(
        run_examples("t", exs, cache, rpm=60_000, concurrency=4, transport=transport(handler), retry=NO_RETRY, **kw)
    )


def test_resume_only_sends_missing(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["state"]["text"])
        return httpx2.Response(200, json=ok_body())

    exs = examples(10)
    assert run(cache, exs[:4], handler).sent == 4
    stats = run(cache, exs, handler)
    assert (stats.sent, stats.cached) == (6, 4)
    assert len(calls) == 10  # nothing paid twice


def test_422_is_remembered_but_other_4xx_stay_retryable(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")

    def handler(request):
        text = json.loads(request.content)["state"]["text"]
        if text == "t0":
            return httpx2.Response(422, json={"detail": "state too long"})
        if text == "t1":
            return httpx2.Response(400, json={"detail": "something odd"})
        return httpx2.Response(200, json=ok_body())

    stats = run(cache, examples(4), handler)
    assert (stats.sent, stats.permanent, stats.failed) == (2, 1, 1)
    # Second pass: the 422 is not resent, the 400 is.
    stats = run(cache, examples(4), handler)
    assert (stats.cached, stats.permanent, stats.failed, stats.sent) == (2, 1, 1, 0)


@pytest.mark.parametrize("status", [401, 402, 403])
def test_fatal_status_stops_the_run(tmp_path, status):
    cache = ResponseCache(tmp_path / "c.db")
    with pytest.raises(FatalAPIError):
        run(cache, examples(50), lambda r: httpx2.Response(status, json={"detail": "no"}))
    assert cache.spent_usd() == 0


def test_circuit_breaker_on_persistent_failures(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(500, json={"detail": "down"})

    with pytest.raises(FatalAPIError, match="consecutive failures"):
        run(cache, examples(500), handler)
    assert len(calls) < CIRCUIT_BREAKER + 10  # stopped early instead of hammering 500 requests


def test_budget_cap_stops_sending(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")
    # Each request's estimate is ~300 tokens; a cap of ~3 requests' worth must stop well before 20.
    stats = run(cache, examples(20), lambda r: httpx2.Response(200, json=ok_body(300)), max_cost_usd=1000 * 0.042 / 1e6)
    assert stats.sent <= 3
    assert stats.skipped_budget >= 17
