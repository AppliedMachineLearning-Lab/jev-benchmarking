#!/usr/bin/env bash
# Pull an open-model run back and import it into the local response cache:
#   JEV_RUN=gemma4-e4b-eval scripts/marvin/fetch.sh
set -euo pipefail
REMOTE="${MARVIN_HOST:-marvin}"
: "${JEV_RUN:?set JEV_RUN}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p "runs/$JEV_RUN"
rsync -avh "$REMOTE:workspaces/jevbench/runs/$JEV_RUN/" "runs/$JEV_RUN/"
"${PYTHON:-$HOME/uv_environments/jevbenchmarking/bin/python}" scripts/import_responses.py runs/"$JEV_RUN"/*.jsonl
