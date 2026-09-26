"""Spend and latency per task from the cache ledger, plus how good the token estimate was."""

from jev_benchmarking.cache import ResponseCache
from jev_benchmarking.config import TOTAL_BUDGET_USD, USD_PER_INPUT_TOKEN


def main() -> None:
    cache = ResponseCache()
    rows = cache.usage_by_task()
    print(f"{'task':<22} {'requests':>8} {'input tok':>12} {'tok/req':>8} {'actual/est':>10} {'USD':>9} {'latency':>8}")
    for task, n, tok, est, lat in rows:
        print(f"{task:<22} {n:>8} {tok:>12,} {tok / n:>8.0f} {tok / max(1, est):>10.2f} {tok * USD_PER_INPUT_TOKEN:>9.4f} {lat:>7.2f}s")
    print(f"\nTotal spent: ${cache.spent_usd():.4f} of ${TOTAL_BUDGET_USD:.2f} budget")
    errors = cache.errors_by_task()
    if errors:
        print("\nRemembered 4xx errors (not resent unless --retry-errors):")
        for task, status, n, msg in errors:
            print(f"  {task:<22} HTTP {status} x{n}: {msg[:200]}")


if __name__ == "__main__":
    main()
