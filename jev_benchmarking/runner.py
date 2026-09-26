"""Async request runner: cache-first, rate-limited, with a hard spend cap."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm
from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError,
)

from jev_benchmarking.cache import ResponseCache, request_key
from jev_benchmarking.config import MODEL, USD_PER_INPUT_TOKEN, load_api_key
from jev_benchmarking.tasks.base import Example

log = logging.getLogger(__name__)

# Fixed per-request overhead observed in the docs' examples (~280 tokens for a one-line state and one question).
OVERHEAD_TOKENS = 300
CHARS_PER_TOKEN = 3.5


class RawResponse(BaseModel):
    """Keep the response body verbatim so the cache holds exactly what the API returned."""

    model_config = ConfigDict(extra="allow")


def estimate_tokens(example: Example) -> int:
    """Deliberately pessimistic estimate; calibrate against `scripts/usage.py` after a pilot."""
    chars = len(json.dumps(example.state, ensure_ascii=False)) + len(json.dumps(example.questions, ensure_ascii=False))
    return OVERHEAD_TOKENS + int(chars / CHARS_PER_TOKEN)


@dataclass
class RunStats:
    cached: int = 0
    sent: int = 0
    failed: int = 0
    skipped_budget: int = 0
    input_tokens: int = 0
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * USD_PER_INPUT_TOKEN


class _RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self.next_slot = time.monotonic()
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = self.next_slot - now
            self.next_slot = max(now, self.next_slot) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class _AuthFailure(Exception):
    pass


async def run_examples(
    task_name: str,
    examples: list[Example],
    cache: ResponseCache,
    *,
    max_cost_usd: float,
    rpm: int,
    concurrency: int,
    retry_errors: bool = False,
) -> RunStats:
    """Send every uncached example once. Stops scheduling when the run's spend would exceed `max_cost_usd`."""
    stats = RunStats()
    keys = [request_key(MODEL, ex.state, ex.questions) for ex in examples]
    have = cache.get_many(keys)
    known_errors = set() if retry_errors else cache.error_keys()

    todo = []
    seen = set()
    for ex, key in zip(examples, keys):
        if key in have:
            stats.cached += 1
        elif key in known_errors:
            stats.failed += 1
        elif key not in seen:  # identical requests (e.g. duplicated rows) are paid once
            seen.add(key)
            todo.append((ex, key))
    if not todo:
        return stats

    limiter = _RateLimiter(rpm)
    sem = asyncio.Semaphore(concurrency)
    reserved = 0  # estimated tokens of in-flight requests
    budget_tokens = max_cost_usd / USD_PER_INPUT_TOKEN
    lock = asyncio.Lock()
    auth_failed = asyncio.Event()
    bar = tqdm(total=len(todo), desc=task_name, unit="req", leave=False)

    async with AsyncTypeSafeClient(
        api_key=load_api_key(),
        model=MODEL,
        timeout=60.0,
        retry=RetryPolicy(max_retries=6, backoff_max=30.0, timeout=180.0),
    ) as client:

        async def one(ex: Example, key: str) -> None:
            nonlocal reserved
            est = estimate_tokens(ex)
            async with sem:
                if auth_failed.is_set():
                    return
                async with lock:
                    if stats.input_tokens + reserved + est > budget_tokens:
                        stats.skipped_budget += 1
                        return
                    reserved += est
                try:
                    await limiter.wait()
                    t0 = time.monotonic()
                    resp = await client.system_one(ex.state, ex.questions, response_model=RawResponse)
                    latency = time.monotonic() - t0
                    body = resp.model_dump()
                    cache.put(key, task_name, body, est, latency)
                    stats.sent += 1
                    stats.input_tokens += int((body.get("usage") or {}).get("input_tokens") or 0)
                except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError) as e:
                    auth_failed.set()
                    raise _AuthFailure(f"authentication failed (HTTP {e.status})") from None
                except TypeSafeAPIError as e:
                    stats.failed += 1
                    stats.errors[str(e.status)] = stats.errors.get(str(e.status), 0) + 1
                    # 4xx means the request itself is bad: remember it so it isn't resent (and re-billed).
                    if 400 <= (e.status or 0) < 500 and e.status not in (408, 429):
                        cache.put_error(key, task_name, e.status, str(e))
                    log.warning("%s %s: HTTP %s %s", task_name, ex.uid, e.status, str(e)[:300])
                except Exception as e:  # connection errors/timeouts after retries: leave uncached, retry next run
                    stats.failed += 1
                    stats.errors[type(e).__name__] = stats.errors.get(type(e).__name__, 0) + 1
                    log.warning("%s %s: %s %s", task_name, ex.uid, type(e).__name__, str(e)[:300])
                finally:
                    async with lock:
                        reserved -= est
                    bar.update()

        try:
            await asyncio.gather(*(one(ex, key) for ex, key in todo))
        finally:
            bar.close()
    return stats
