#!/usr/bin/env bash
# Submit an open-model scoring run from the laptop (via ssh). Examples:
#   JEV_RUN=gemma4-e4b-pilot JEV_SPLIT=dev JEV_LIMIT=20 SHARDS=1 PARTITION=mlgpu_devel TIME=01:00:00 scripts/marvin/submit.sh
#   JEV_RUN=gemma4-e4b-eval SHARDS=8 scripts/marvin/submit.sh
#   JEV_RUN=qwen38-27b-eval JEV_MODEL=Qwen/Qwen3.8-27B GPUS=2 PARTITION=mlgpu_medium TIME=24:00:00 SHARDS=8 scripts/marvin/submit.sh
# GPU_TYPE (default a40) selects the GPU model, e.g. an 80 GB type together with its PARTITION.
# GPUS>1 splits the model over the GPUs of one job (for models larger than one GPU).
set -euo pipefail
REMOTE="${MARVIN_HOST:-marvin}"
SHARDS="${SHARDS:-8}"
GPUS="${GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a40}"
DEVICE=cuda
[ "$GPUS" -gt 1 ] && DEVICE=auto
: "${JEV_RUN:?set JEV_RUN (output directory name on the workspace)}"
EXPORTS="ALL,JEV_RUN=$JEV_RUN,JEV_MODEL=${JEV_MODEL:-google/gemma-4-E4B-it},JEV_TASKS=${JEV_TASKS:-all},JEV_SPLIT=${JEV_SPLIT:-eval},JEV_LIMIT=${JEV_LIMIT:-},JEV_MAX_BATCH_TOKENS=${JEV_MAX_BATCH_TOKENS:-32768},JEV_DEVICE=$DEVICE"
OPTS=(--array="0-$((SHARDS - 1))" --export="$EXPORTS" --gres="gpu:$GPU_TYPE:$GPUS" --mem="$((64 * GPUS))G" --cpus-per-task="$((8 * GPUS))")
[ -n "${PARTITION:-}" ] && OPTS+=(--partition="$PARTITION")
[ -n "${TIME:-}" ] && OPTS+=(--time="$TIME")
ssh "$REMOTE" "cd jev-benchmarking && mkdir -p \$HOME/workspaces/jevbench/logs && cd \$HOME/workspaces/jevbench/logs && sbatch ${OPTS[*]} \$HOME/jev-benchmarking/scripts/marvin/open_model.slurm"
