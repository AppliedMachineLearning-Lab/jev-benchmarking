#!/usr/bin/env bash
# Full eval run, detached from the terminal (survives logout/closed sessions). Re-launching resumes:
# everything already in cache/responses.db is skipped. Watch with: tail -f logs/<latest>.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
PY="${PY:-python}"  # python of the active environment; override with PY=/path/to/python
LOG="logs/full_run_$(date +%Y%m%d_%H%M%S).log"
setsid nohup "$PY" -u scripts/run.py "${@:-all}" --split eval --until-done --yes \
    --max-cost "${MAX_COST:-10.30}" --rpm "${RPM:-1100}" >"$LOG" 2>&1 </dev/null &
echo "started pid $! -> $LOG"
