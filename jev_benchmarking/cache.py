"""SQLite cache of raw API responses. Every paid request lands here, so it doubles as the spend ledger."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from jev_benchmarking.config import CACHE_DB, USD_PER_INPUT_TOKEN

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key           TEXT PRIMARY KEY,
    task          TEXT NOT NULL,
    model         TEXT NOT NULL,
    response      TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    est_tokens    INTEGER NOT NULL,
    latency_s     REAL NOT NULL,
    created_at    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS errors (
    key        TEXT PRIMARY KEY,
    task       TEXT NOT NULL,
    status     INTEGER,
    message    TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def request_key(model: str, state: Any, questions: dict) -> str:
    # No sort_keys: option order inside a Choice is part of the request and may affect the answer.
    payload = json.dumps({"model": model, "state": state, "questions": questions}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    def __init__(self, path: Path = CACHE_DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        # WAL: every response is durable once committed, a killed run can't corrupt the file, and
        # evaluate.py/usage.py can read while a run is writing.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)

    def close(self) -> None:
        """Fold the WAL back into the main file so the tracked responses.db is self-contained."""
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_many(self, keys: list[str]) -> dict[str, dict]:
        out = {}
        for i in range(0, len(keys), 500):
            chunk = keys[i : i + 500]
            marks = ",".join("?" * len(chunk))
            for key, resp in self.db.execute(f"SELECT key, response FROM responses WHERE key IN ({marks})", chunk):
                out[key] = json.loads(resp)
        return out

    def put(self, key: str, task: str, response: dict, est_tokens: int, latency_s: float) -> None:
        usage = response.get("usage") or {}
        self.db.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                task,
                response.get("model", ""),
                json.dumps(response, ensure_ascii=False),
                int(usage.get("input_tokens") or 0),
                int(usage.get("output_tokens") or 0),
                est_tokens,
                latency_s,
                time.time(),
            ),
        )
        self.db.execute("DELETE FROM errors WHERE key = ?", (key,))
        self.db.commit()

    def error_keys(self) -> set[str]:
        return {k for (k,) in self.db.execute("SELECT key FROM errors")}

    def put_error(self, key: str, task: str, status: int | None, message: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO errors VALUES (?, ?, ?, ?, ?)", (key, task, status, message[:2000], time.time())
        )
        self.db.commit()

    def spent_usd(self) -> float:
        (tokens,) = self.db.execute("SELECT COALESCE(SUM(input_tokens), 0) FROM responses").fetchone()
        return tokens * USD_PER_INPUT_TOKEN

    def usage_by_task(self, since: float = 0.0) -> list[tuple]:
        """(task, requests, input tokens, estimated tokens, mean latency) for responses created at/after `since`."""
        return self.db.execute(
            """SELECT task, COUNT(*), SUM(input_tokens), SUM(est_tokens), AVG(latency_s)
               FROM responses WHERE created_at >= ? GROUP BY task ORDER BY SUM(input_tokens) DESC""",
            (since,),
        ).fetchall()

    def errors_by_task(self) -> list[tuple]:
        return self.db.execute(
            "SELECT task, status, COUNT(*), MIN(message) FROM errors GROUP BY task, status ORDER BY task"
        ).fetchall()
