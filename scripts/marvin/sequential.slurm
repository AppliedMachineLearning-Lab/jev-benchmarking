#!/usr/bin/env bash
#SBATCH --job-name=jev-sequential
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tdeusser@uni-bonn.de
#
# Play a sequential benchmark with an open-weight model. One array task = one shard of the episodes
# (assigned by episode uid). Resumable: a resubmitted shard skips episodes already in its JSONL. Submit via
#   JEV_MODEL=google/gemma-4-E4B-it SHARDS=4 SCRIPT=sequential.slurm scripts/marvin/submit.sh
#
# Environment (set by submit.sh):
#   JEV_MODEL  (default google/gemma-4-E4B-it)   JEV_TASKS (environment, default alfworld)
#   JEV_SPLIT  (default eval)                    JEV_LIMIT (default: none = all episodes)
#   JEV_RUN    name of the per-shard response caches under $JEV_WS/seqcache/
#   JEV_DEVICE (default cuda; "auto" splits a model that does not fit one GPU over all allocated GPUs)
set -euo pipefail
source "$HOME/${JEV_REMOTE_DIR:-jev-benchmarking}/scripts/marvin/common.sh"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ALFWORLD_DATA="$JEV_WS/alfworld"

MODEL="${JEV_MODEL:-google/gemma-4-E4B-it}"
ENV_NAME="${JEV_TASKS:-alfworld}"
[ "$ENV_NAME" = all ] && ENV_NAME=alfworld
SPLIT="${JEV_SPLIT:-eval}"
RUN="${JEV_RUN:?set JEV_RUN}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
LIMIT_ARG=()
[ -n "${JEV_LIMIT:-}" ] && LIMIT_ARG=(--limit "$JEV_LIMIT")
# One SQLite cache per shard: WAL locking is not safe across nodes on Lustre. Merged locally afterwards.
export JEV_CACHE_DB="$JEV_WS/seqcache/$RUN/shard$(printf %02d "$SHARD")/responses.db"

echo "=== job ${SLURM_JOB_ID:-?} shard $SHARD/$NSHARDS on $(hostname): $MODEL, env=$ENV_NAME, split=$SPLIT ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$HOME/$JEV_REMOTE_DIR"
"$JEV_VENV/bin/python" scripts/run_sequential.py "$ENV_NAME" \
    --backend "hf:$MODEL" --split "$SPLIT" "${LIMIT_ARG[@]}" \
    --shard "$SHARD" --num-shards "$NSHARDS" --device "${JEV_DEVICE:-cuda}"
