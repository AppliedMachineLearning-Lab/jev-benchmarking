"""Async request runner: cache-first, rate-limited, with a hard spend cap."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict
from tqdm import tqdm
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeAPIError

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
    failed: int = 0  # retryable failures, left uncached for the next pass
    permanent: int = 0  # requests the API rejects as invalid (413/422); remembered, never resent
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


class FatalAPIError(Exception):
    """Stop the whole run: bad key, no credit, or the API keeps failing."""


FATAL_STATUSES = {401, 402, 403}
# Only these mean "this exact request is invalid" (e.g. state over the context limit), so it is safe to
# remember them and never resend. Anything else (e.g. a 400 for an empty balance) must stay retryable.
PERMANENT_STATUSES = {413, 422}
# The API reports an over-long request as a 400 with this error type (observed on jev-1.13.0).
PERMANENT_ERROR_TYPES = ("max_tokens_exceeded",)


def _is_permanent(status: int, message: str) -> bool:
    return status in PERMANENT_STATUSES or (status == 400 and any(t in message for t in PERMANENT_ERROR_TYPES))
CIRCUIT_BREAKER = 25  # consecutive failures before giving up on the run
PROGRESS_EVERY = 60.0  # seconds between progress log lines


async def run_examples(
    task_name: str,
    examples: list[Example],
    cache: ResponseCache,
    *,
    max_cost_usd: float,
    rpm: int,
    concurrency: int,
    retry_errors: bool = False,
    transport=None,  # tests inject an httpx2 MockTransport here
    retry: RetryPolicy | None = None,
) -> RunStats:
    """Send every uncached example once. Stops scheduling when the run's spend would exceed `max_cost_usd`.

    Safe to interrupt at any point: each response is committed as it arrives, and a rerun only sends
    what is still missing. Raises `FatalAPIError` when continuing is pointless.
    """
    stats = RunStats()
    keys = [request_key(MODEL, ex.state, ex.questions) for ex in examples]
    have = cache.get_many(keys)
    known_errors = set() if retry_errors else cache.error_keys()

    queue: asyncio.Queue = asyncio.Queue()
    seen = set()
    for ex, key in zip(examples, keys):
        if key in have:
            stats.cached += 1
        elif key in known_errors:
            stats.permanent += 1
        elif key not in seen:  # identical requests (e.g. duplicated rows) are paid once
            seen.add(key)
            queue.put_nowait((ex, key))
    total = queue.qsize()
    if not total:
        return stats

    limiter = _RateLimiter(rpm)
    budget_tokens = max_cost_usd / USD_PER_INPUT_TOKEN
    reserved = 0  # estimated tokens of in-flight requests
    consecutive_failures = 0
    fatal: list[str] = []
    stop = asyncio.Event()
    bar = tqdm(total=total, desc=task_name, unit="req", leave=False, disable=None)  # silent when not a TTY
    t_start = last_log = time.monotonic()

    def fail(label: str, message: str) -> None:
        nonlocal consecutive_failures
        stats.failed += 1
        stats.errors[label] = stats.errors.get(label, 0) + 1
        consecutive_failures += 1
        if consecutive_failures >= CIRCUIT_BREAKER and not stop.is_set():
            fatal.append(f"{CIRCUIT_BREAKER} consecutive failures, last: {message[:300]}")
            stop.set()

    async with AsyncTypeSafeClient(
        api_key=load_api_key(),
        model=MODEL,
        timeout=60.0,
        retry=retry or RetryPolicy(max_retries=6, backoff_max=30.0, timeout=180.0),
        transport=transport,
    ) as client:

        async def worker() -> None:
            nonlocal reserved, consecutive_failures, last_log
            while not stop.is_set():
                try:
                    ex, key = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                est = estimate_tokens(ex)
                if stats.input_tokens + reserved + est > budget_tokens:
                    stats.skipped_budget += 1
                    continue
                reserved += est
                try:
                    await limiter.wait()
                    if stop.is_set():
                        return
                    t0 = time.monotonic()
                    resp = await client.system_one(ex.state, ex.questions, response_model=RawResponse)
                    body = resp.model_dump()
                    cache.put(key, task_name, body, est, time.monotonic() - t0)
                    stats.sent += 1
                    stats.input_tokens += int((body.get("usage") or {}).get("input_tokens") or 0)
                    consecutive_failures = 0
                except TypeSafeAPIError as e:
                    status = e.status or 0
                    if status in FATAL_STATUSES:
                        fatal.append(f"HTTP {status}: {str(e)[:300]}")
                        stop.set()
                        return
                    if _is_permanent(status, str(e)):
                        cache.put_error(key, task_name, status, str(e))
                        stats.permanent += 1
                        log.info("%s %s: HTTP %s (invalid request, remembered, not resent) %s", task_name, ex.uid, status, str(e)[:200])
                    else:
                        fail(str(status), f"HTTP {status}: {e}")
                        log.warning("%s %s: HTTP %s %s", task_name, ex.uid, status, str(e)[:300])
                except Exception as e:  # connection errors/timeouts after retries: left uncached for the next pass
                    fail(type(e).__name__, f"{type(e).__name__}: {e}")
                    log.warning("%s %s: %s %s", task_name, ex.uid, type(e).__name__, str(e)[:300])
                finally:
                    reserved -= est
                    bar.update()
                now = time.monotonic()
                if now - last_log >= PROGRESS_EVERY:
                    last_log = now
                    rate = stats.sent / (now - t_start) * 60
                    log.info(
                        "%s: %d/%d sent, %d failed, %.0f req/min, $%.4f this task",
                        task_name, stats.sent, total, stats.failed, rate, stats.cost_usd,
                    )  # fmt: skip

        try:
            await asyncio.gather(*(worker() for _ in range(concurrency)))
        finally:
            bar.close()
    if fatal:
        raise FatalAPIError(f"{task_name}: {fatal[0]}")
    return stats
