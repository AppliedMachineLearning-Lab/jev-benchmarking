"""Import open-model responses (JSONL from run_open_model.py) into the response cache.

    python scripts/import_responses.py runs/gemma/*.jsonl

Each model gets its own database (cache/responses.<org>__<name>.db); the Jev database is never touched.
Idempotent: re-importing replaces rows with the same key.
"""

import argparse
import json
from pathlib import Path

from jev_benchmarking.cache import ResponseCache
from jev_benchmarking.config import cache_db_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    args = p.parse_args()
    caches: dict[str, ResponseCache] = {}
    counts: dict[str, int] = {}
    bad = 0
    for f in args.files:
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1  # e.g. the last line of a job killed mid-write
                continue
            model = row["response"]["model"]
            if model.startswith("jev-"):
                raise SystemExit(f"refusing to import a Jev response from {f}: Jev responses come only from run.py")
            if model not in caches:
                caches[model] = ResponseCache(model=model)
            caches[model].put(row["key"], row["task"], row["response"], est_tokens=0, latency_s=row.get("latency_s", 0.0))
            counts[model] = counts.get(model, 0) + 1
    for model, cache in caches.items():
        print(f"imported {counts[model]} responses of {model} into {cache_db_path(model)}")
        cache.close()
    print(f"skipped {bad} unreadable lines")


if __name__ == "__main__":
    main()
