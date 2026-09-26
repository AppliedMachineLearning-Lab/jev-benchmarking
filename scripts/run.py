"""Send benchmark requests to Jev (cache-first, with a hard spend cap).

Examples:
    python scripts/run.py --list
    python scripts/run.py ag_news boolq --split dev --limit 20 --dry-run
    python scripts/run.py all --split dev --limit 20 --max-cost 0.05
    python scripts/run.py all --max-cost 10 --until-done --yes      # full run; rerun the same command to resume

Resumable by construction: every response is committed to cache/responses.db as it arrives and requests
are keyed by content, so after a crash, Ctrl-C or reboot the same command continues where it stopped.
"""

import argparse
import asyncio
import fcntl
import json
import logging
import sys
import time

from jev_benchmarking.cache import ResponseCache, request_key
from jev_benchmarking.config import CACHE_DB, DEFAULT_RPM, MODEL, TOTAL_BUDGET_USD, USD_PER_INPUT_TOKEN
from jev_benchmarking.runner import FatalAPIError, estimate_tokens, run_examples
from jev_benchmarking.tasks import TASKS, get_tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", nargs="*", help="task names, or 'all'")
    p.add_argument("--list", action="store_true", help="list tasks and exit")
    p.add_argument("--split", choices=["eval", "dev"], default="eval", help="dev = train/validation split for prompt work")
    p.add_argument("--limit", type=int, help="max examples per task (deterministic, nested samples)")
    p.add_argument("--subsets", nargs="+", help="restrict to these configs (languages, subjects, ...)")
    p.add_argument("--max-cost", type=float, default=0.02, help="USD cap for this run (default: 0.02)")
    p.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--dry-run", action="store_true", help="build requests and estimate cost, send nothing")
    p.add_argument("--show", type=int, default=0, metavar="N", help="print the first N requests per task")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--retry-errors", action="store_true", help="resend requests that previously failed with 413/422")
    p.add_argument("--until-done", action="store_true", help="repeat passes until nothing is left or a pass makes no progress")
    p.add_argument("--max-passes", type=int, default=5)
    p.add_argument("--pause", type=float, default=120.0, help="seconds between passes (lets transient outages clear)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for t in TASKS.values():
            splits = ", ".join(f"{k}={v}" for k, v in t.splits.items() if v)
            print(f"{t.name:<22} {t.hf_path:<45} [{splits}]  {t.description}")
        return
    if not args.tasks:
        raise SystemExit("Name tasks to run (or 'all'); see --list.")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("jev_benchmarking").setLevel(logging.INFO)

    cache = ResponseCache()
    plan = []
    total_est = 0
    for task in get_tasks(args.tasks):
        if task.splits.get(args.split) is None:
            print(f"- {task.name}: no '{args.split}' split, skipped")
            continue
        try:
            examples = task.examples(args.split, limit=args.limit, subsets=args.subsets)
        except Exception as e:  # e.g. gated dataset without access
            print(f"- {task.name}: could not load ({type(e).__name__}: {str(e)[:200]})")
            continue
        keys = {request_key(MODEL, e.state, e.questions): e for e in examples}
        have = cache.get_many(list(keys))
        todo = [e for k, e in keys.items() if k not in have]
        est = sum(estimate_tokens(e) for e in todo)
        total_est += est
        plan.append((est, task, examples))
        print(f"- {task.name:<22} {len(examples):>6} examples, {len(todo):>6} uncached, est. {est:>10,} tok (${est * USD_PER_INPUT_TOKEN:.4f})")
        if task.skipped_configs:
            print(f"  !! {len(task.skipped_configs)} config(s) skipped: {', '.join(task.skipped_configs)}")
        for e in examples[: args.show]:
            print(json.dumps({"uid": e.uid, "state": e.state, "questions": e.questions, "gold": {k: sorted(v) if isinstance(v, frozenset) else v for k, v in e.gold.items()}}, ensure_ascii=False, indent=1)[:3000])

    spent = cache.spent_usd()
    est_usd = total_est * USD_PER_INPUT_TOKEN
    cap = min(args.max_cost, TOTAL_BUDGET_USD - spent)
    print(f"\nEstimated cost of uncached requests: ${est_usd:.4f}  |  run cap: ${args.max_cost:.4f}")
    print(f"Spent so far (cache ledger): ${spent:.4f} of ${TOTAL_BUDGET_USD:.2f} total budget (JEV_TOTAL_BUDGET_USD)")
    if args.dry_run or total_est == 0:
        return
    if cap <= 0:
        raise SystemExit("Total budget exhausted; raise JEV_TOTAL_BUDGET_USD to continue.")
    if est_usd > cap:
        print(f"NOTE: the estimate exceeds the cap; the run stops once ${cap:.4f} is spent.")
    if not args.yes and input("Send requests? [y/N] ").strip().lower() != "y":
        raise SystemExit("Aborted.")

    # Two concurrent runs would both send (and pay for) the same uncached requests.
    lock_file = open(CACHE_DB.parent / "run.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another run.py is already sending requests (cache/run.lock). Wait for it or stop it first.") from None

    plan.sort(key=lambda item: item[0])  # cheapest first: partial runs still cover as many tasks as possible
    remaining = cap
    passes = args.max_passes if args.until_done else 1
    try:
        for n_pass in range(1, passes + 1):
            print(f"\n=== pass {n_pass}/{passes} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===", flush=True)
            sent = failed = 0
            for _, task, examples in plan:
                if remaining <= 0:
                    print(f"- {task.name}: skipped, run cap reached", flush=True)
                    continue
                stats = asyncio.run(
                    run_examples(
                        task.name, examples, cache,
                        max_cost_usd=remaining, rpm=args.rpm, concurrency=args.concurrency, retry_errors=args.retry_errors,
                    )
                )  # fmt: skip
                remaining -= stats.cost_usd
                sent += stats.sent
                failed += stats.failed
                if stats.sent or stats.failed or stats.skipped_budget or (n_pass == 1 and stats.permanent):
                    msg = f"- {task.name:<22} sent {stats.sent}, cached {stats.cached}, failed {stats.failed}"
                    if stats.permanent:
                        msg += f", invalid (413/422, not resent) {stats.permanent}"
                    if stats.skipped_budget:
                        msg += f", SKIPPED (cap) {stats.skipped_budget}"
                    if stats.errors:
                        msg += f", errors {stats.errors}"
                    print(f"{time.strftime('%H:%M:%S')} {msg}, {stats.input_tokens:,} tok (${stats.cost_usd:.4f})", flush=True)
            print(f"pass {n_pass}: sent {sent}, failed {failed}, ledger total ${cache.spent_usd():.4f}", flush=True)
            if not args.until_done or remaining <= 0:
                break
            if failed == 0:
                print("Nothing left to send.", flush=True)
                break
            if sent == 0:
                print("No progress in this pass; stopping. Rerun later to retry the failures.", flush=True)
                break
            time.sleep(args.pause)
    except FatalAPIError as e:
        print(f"\nFATAL: {e}\nEverything received so far is saved; rerun the same command to resume.", flush=True)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted. Everything received so far is saved; rerun the same command to resume.", flush=True)
        sys.exit(130)
    finally:
        print(f"\nRun spend: ${cap - remaining:.4f}; ledger total: ${cache.spent_usd():.4f}", flush=True)
        cache.close()


if __name__ == "__main__":
    main()
