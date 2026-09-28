#!/usr/bin/env bash
# Push the code to Marvin (explicit allowlist; no responses.db, results or paper).
set -euo pipefail
REMOTE="${MARVIN_HOST:-marvin}"
REMOTE_PATH="${JEV_REMOTE_DIR:-jev-benchmarking}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ssh "$REMOTE" "mkdir -p $REMOTE_PATH/cache"
rsync -avh --delete --exclude='__pycache__/' --exclude='*.pyc' \
  jev_benchmarking scripts tests pyproject.toml README.md MANIFEST.in "$REMOTE:$REMOTE_PATH/"
# Config-name cache lets datasets load offline (config lists are otherwise fetched from the Hub).
rsync -avh cache/hf_config_names.json "$REMOTE:$REMOTE_PATH/cache/"
