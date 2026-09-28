#!/usr/bin/env bash
#SBATCH --job-name=jev-openmodel
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tdeusser@uni-bonn.de
#
# Score benchmark tasks with an open-weight model. One array task = one shard of the requests (assigned by
# request hash, so shards are balanced across datasets). Resumable: a resubmitted shard skips everything
# already written to $JEV_RUNS/<run>/. Submit via scripts/marvin/submit.sh.
#
# Environment (set by submit.sh):
#   JEV_MODEL  (default google/gemma-4-E4B-it)   JEV_TASKS (default "all"; several: joined with "+", since sbatch --export splits on commas)
#   JEV_SPLIT  (default eval)                    JEV_LIMIT (default: none = full split)
#   JEV_RUN    output subdirectory name           JEV_MAX_BATCH_TOKENS (default 32768)
set -euo pipefail
source "$HOME/${JEV_REMOTE_DIR:-jev-benchmarking}/scripts/marvin/common.sh"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # everything was prefetched on the login node

MODEL="${JEV_MODEL:-google/gemma-4-E4B-it}"
SPLIT="${JEV_SPLIT:-eval}"
RUN="${JEV_RUN:?set JEV_RUN}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
LIMIT_ARG=()
[ -n "${JEV_LIMIT:-}" ] && LIMIT_ARG=(--limit "$JEV_LIMIT")

echo "=== job ${SLURM_JOB_ID:-?} shard $SHARD/$NSHARDS on $(hostname): $MODEL, tasks=${JEV_TASKS:-all}, split=$SPLIT ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$HOME/$JEV_REMOTE_DIR"
TASKS="${JEV_TASKS:-all}"
"$JEV_VENV/bin/python" scripts/run_open_model.py ${TASKS//+/ } \
    --model "$MODEL" --split "$SPLIT" "${LIMIT_ARG[@]}" \
    --shard "$SHARD" --num-shards "$NSHARDS" \
    --max-batch-tokens "${JEV_MAX_BATCH_TOKENS:-32768}" \
    --out "$JEV_RUNS/$RUN"
