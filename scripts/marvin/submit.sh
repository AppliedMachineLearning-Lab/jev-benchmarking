#!/usr/bin/env bash
# Submit an open-model scoring run from the laptop (via ssh). Examples:
#   JEV_RUN=gemma4-e4b-pilot JEV_SPLIT=dev JEV_LIMIT=20 SHARDS=1 PARTITION=mlgpu_devel TIME=01:00:00 scripts/marvin/submit.sh
#   JEV_RUN=gemma4-e4b-eval SHARDS=8 scripts/marvin/submit.sh
set -euo pipefail
REMOTE="${MARVIN_HOST:-marvin}"
SHARDS="${SHARDS:-8}"
: "${JEV_RUN:?set JEV_RUN (output directory name on the workspace)}"
EXPORTS="ALL,JEV_RUN=$JEV_RUN,JEV_MODEL=${JEV_MODEL:-google/gemma-4-E4B-it},JEV_TASKS=${JEV_TASKS:-all},JEV_SPLIT=${JEV_SPLIT:-eval},JEV_LIMIT=${JEV_LIMIT:-},JEV_MAX_BATCH_TOKENS=${JEV_MAX_BATCH_TOKENS:-32768}"
OPTS=(--array="0-$((SHARDS - 1))" --export="$EXPORTS")
[ -n "${PARTITION:-}" ] && OPTS+=(--partition="$PARTITION")
[ -n "${TIME:-}" ] && OPTS+=(--time="$TIME")
ssh "$REMOTE" "cd jev-benchmarking && mkdir -p \$HOME/workspaces/jevbench/logs && cd \$HOME/workspaces/jevbench/logs && sbatch ${OPTS[*]} \$HOME/jev-benchmarking/scripts/marvin/open_model.slurm"
