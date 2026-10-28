"""Merge the per-shard response caches of the sequential runs into one database per model.

    python scripts/merge_sequential_caches.py runs/alfworld-seqcache cache/sequential

The Slurm jobs write one cache per shard (<src>/<run>/shardNN/responses.<org>__<name>.db), since SQLite's WAL
is not safe across nodes. This writes <out>/responses.<org>__<name>.db, which run_sequential.py reads with
JEV_CACHE_DB=<out>/responses.db. Each shard is copied (with its -wal/-shm files) before it is opened, so the
sources stay untouched. A key found in several shards is kept once; differing answers for it are reported.
Finally, every request in the episode logs (results/sequential/<env>/<org>__<name>.*.jsonl) must be in the
merged database.
"""

import argparse
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def merge(shards: list[Path], out: Path) -> tuple[set[str], int, int]:
    """Copy all rows of `shards` into `out`; returns the merged keys, rows read and conflicting keys."""
    schema = sqlite3.connect(shards[0]).execute("SELECT sql FROM sqlite_master WHERE type = 'table'").fetchall()
    dst = sqlite3.connect(out)
    for (sql,) in schema:
        dst.execute(sql)
    answers: dict[str, dict] = {}
    rows = conflicts = 0
    for db in shards:
        with tempfile.TemporaryDirectory() as tmp:
            for f in db.parent.glob(db.name + "*"):
                shutil.copy2(f, tmp)
            src = sqlite3.connect(Path(tmp) / db.name)
            if src.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise SystemExit(f"{db}: integrity check failed")
            for row in src.execute("SELECT * FROM responses ORDER BY created_at"):
                rows += 1
                key, answer = row[0], json.loads(row[3])["answers"]
                if key in answers:
                    conflicts += answers[key] != answer
                    continue
                answers[key] = answer
                dst.execute(f"INSERT INTO responses VALUES ({','.join('?' * len(row))})", row)
            for row in src.execute("SELECT * FROM errors"):
                dst.execute(f"INSERT OR IGNORE INTO errors VALUES ({','.join('?' * len(row))})", row)
            src.close()
    dst.commit()
    dst.execute("VACUUM")
    dst.close()
    return set(answers), rows, conflicts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="directory with <run>/shardNN/responses.<slug>.db")
    p.add_argument("out", type=Path, help="output directory, e.g. cache/sequential")
    p.add_argument("--results", type=Path, default=ROOT / "results" / "sequential", help="episode logs to check")
    args = p.parse_args()

    by_slug: dict[str, list[Path]] = {}
    for db in sorted(args.src.glob("*/shard*/responses.*.db")):
        by_slug.setdefault(db.name.removeprefix("responses.").removesuffix(".db"), []).append(db)
    if not by_slug:
        raise SystemExit(f"no shard caches under {args.src}")
    args.out.mkdir(parents=True, exist_ok=True)
    for slug, shards in by_slug.items():
        out = args.out / f"responses.{slug}.db"
        if out.exists():
            raise SystemExit(f"{out} exists; remove it to rebuild")
        keys, rows, conflicts = merge(shards, out)
        steps = [s["request"] for f in args.results.glob(f"*/{slug}.*.jsonl") for line in f.open()
                 for s in json.loads(line)["steps"]]
        missing = sum(k not in keys for k in steps)
        print(f"{slug}: {len(shards)} shards, {rows} rows, {len(keys)} unique keys, {conflicts} conflicting, "
              f"{len(steps)} episode steps, {missing} missing -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        if missing:
            raise SystemExit(f"{missing} requests of the episode logs are not in {out}")


if __name__ == "__main__":
    main()
