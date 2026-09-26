# jev-benchmarking

Benchmarks TypeSafe's [Jev](https://docs.typesafe.ai/introduction) (pinned to `jev-1.13.0`) on 37 open
datasets. The dataset audit and selection rationale are in [docs/datasets.md](docs/datasets.md).

## Setup

```fish
uvactivate jevbenchmarking
uv pip install -e ".[dev]"
```

The API key is read from `$TYPESAFE_API_KEY` or, if unset, from
`~/.config/typesafe-ai/keys/jevbenchmarking` (override with `$TYPESAFE_API_KEY_FILE`).
`lytang/LLM-AggreFact` is gated: accept its terms on Hugging Face with the logged-in account.

## Usage

```fish
python scripts/check_api.py                     # validate the key (free: lists models)
python scripts/run.py --list                    # all tasks
python scripts/run.py all --split dev --limit 20 --dry-run --show 1   # inspect requests + cost estimate
python scripts/run.py all --split dev --limit 20 --max-cost 0.05      # send (asks for confirmation)
python scripts/evaluate.py all --split dev --limit 20                 # metrics -> results/dev/
python scripts/usage.py                         # spend per task, estimate accuracy, 4xx errors
pytest                                          # offline tests
```

- **Splits:** `--split dev` uses a training/validation split for prompt work; `--split eval` is the
  reported split. Freeze the question wording before running `eval`.
- **Sampling:** `--limit N` takes a deterministic, hash-ordered sample spread evenly over configs
  (languages, subjects). Smaller samples are subsets of larger ones, so pilot answers are reused.
- **Cache and ledger:** every response is stored in `cache/responses.db`, keyed by the exact
  request. Nothing is ever paid for twice, and `evaluate.py` never calls the API. The same table is
  the spend ledger.
- **Spend guards:** `--max-cost` caps a run (default $0.02), and `JEV_TOTAL_BUDGET_USD`
  (default $0.50) caps the total recorded in the ledger.

## Layout

- `jev_benchmarking/tasks/`: one class per dataset (request construction + gold labels)
- `jev_benchmarking/runner.py`: async, rate-limited sender with a budget guard
- `jev_benchmarking/evaluate.py`, `metrics.py`: accuracy/F1/correlation, calibration (ECE, Brier),
  selective accuracy, bootstrap CIs
- `scripts/`: entry points
