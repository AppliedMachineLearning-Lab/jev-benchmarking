# jev-benchmarking

Benchmarks TypeSafe's [Jev](https://docs.typesafe.ai/introduction) (pinned to `jev-1.13.0`) on 37 open
datasets. The dataset audit, selection rationale and actual cost are in [docs/datasets.md](docs/datasets.md);
per-dataset metrics are in `results/eval/`.

## Setup

```fish
uvactivate jevbenchmarking
uv pip install -e ".[dev,paper]"
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
./scripts/run_full.sh                           # full eval run, detached; re-launch to resume (log in logs/)
HF_HUB_OFFLINE=1 python scripts/make_paper_assets.py   # tables/figures -> paper/ (needs .[paper]; no API calls)
pytest                                          # offline tests
```

Additional analyses (inputs for two paper tables; `make_paper_assets.py` skips a table if its responses are
missing):

```fish
python scripts/run.py go_emotions unfair_tos agb_de toxic_chat --split dev --limit 1000   # threshold tuning sample
python scripts/run.py probes                    # memorization probes (MMLU/C-Eval calculation-heavy subjects)
```

- **Splits:** `--split dev` uses a training/validation split for prompt work; `--split eval` is the
  reported split. Freeze the question wording before running `eval`.
- **Sampling:** `--limit N` takes a deterministic, hash-ordered sample spread evenly over configs
  (languages, subjects). Smaller samples are subsets of larger ones, so pilot answers are reused.
- **Cache and ledger:** every Jev response is stored in `cache/responses.db` (other models: one database
  each, `cache/responses.<org>__<model>.db`, gitignored), keyed by the exact
  request. Nothing is ever paid for twice, and `evaluate.py` never calls the API. The same table is
  the spend ledger.
- **Spend guards:** `--max-cost` caps a run (default $0.02), and `JEV_TOTAL_BUDGET_USD`
  (default $10.40) caps the total recorded in the ledger.

## Layout

- `jev_benchmarking/tasks/`: one class per dataset (request construction + gold labels);
  `probes.py` holds the memorization probes (addressable by name or `probes`, never part of `all`)
- `jev_benchmarking/runner.py`: async, rate-limited sender with a budget guard
- `jev_benchmarking/evaluate.py`, `metrics.py`: accuracy/F1/correlation, calibration (ECE, Brier),
  selective accuracy, bootstrap CIs
- `jev_benchmarking/thresholds.py`: per-question Noul thresholds tuned on dev data
- `jev_benchmarking/report/`: paper tables (LaTeX, template style) and figures (plotnine + tol-colors)
- `scripts/`: entry points
