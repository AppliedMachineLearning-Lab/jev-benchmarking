"""Paths, model/pricing constants and API-key resolution."""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = Path(os.environ.get("JEV_CACHE_DB", ROOT / "cache" / "responses.db"))
RESULTS_DIR = ROOT / "results"

# Pin the versioned model rather than `jev-latest`, so an alias move can't silently change results.
MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")

# https://docs.typesafe.ai/models: charged per input token, output tokens are free.
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000

# Hard ceiling on the total spend recorded in the cache DB across all runs.
TOTAL_BUDGET_USD = float(os.environ.get("JEV_TOTAL_BUDGET_USD", "0.50"))

# Documented limit is 1,200 req/min; stay below it.
DEFAULT_RPM = 1000

API_KEY_FILE = Path(
    os.environ.get("TYPESAFE_API_KEY_FILE", Path.home() / ".config" / "typesafe-ai" / "keys" / "jevbenchmarking")
)

_ASSIGNMENT = re.compile(r"^(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.*)$")


def load_api_key() -> str:
    """Return the API key from `TYPESAFE_API_KEY` or the key file. Never log the return value."""
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if key:
        return key
    if not API_KEY_FILE.is_file():
        raise RuntimeError(f"No TYPESAFE_API_KEY set and key file {API_KEY_FILE} not found.")
    lines = [line.strip() for line in API_KEY_FILE.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Key file {API_KEY_FILE} is empty.")
    # Accept either a bare key or a `NAME=value` / `export NAME="value"` line.
    match = _ASSIGNMENT.match(lines[-1])
    return (match.group(1) if match else lines[-1]).strip().strip("'\"")
