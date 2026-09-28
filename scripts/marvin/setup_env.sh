#!/usr/bin/env bash
# One-time setup on the Marvin LOGIN node (downloads wheels only, no GPU work):
#   ssh marvin 'bash jev-benchmarking/scripts/marvin/setup_env.sh'
#
# Creates the workspace and a dedicated venv. transformers is pinned to 5.14.1, the version the
# VonNeumannBench notes found to load gemma-4 configs correctly (5.15.0 raises on gemma4's per-layer
# config); it is also the version the scorer was tested with locally.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [ ! -d "$JEV_WS" ]; then
  echo "=== allocating Lustre SSD workspace (90 days, reminder 7 days before expiry) ==="
  ws_allocate -F mlnvme -r 7 -m tdeusser@uni-bonn.de jevbench 90
  mkdir -p "$(dirname "$JEV_WS")"
  ln -sfn "$(ws_find -F mlnvme jevbench)" "$JEV_WS"
fi
mkdir -p "$HF_HUB_CACHE" "$JEV_RUNS" "$JEV_WS/logs"

echo "=== venv $JEV_VENV ==="
uv venv "$JEV_VENV" --python 3.12
uv pip install --python "$JEV_VENV/bin/python" torch "transformers==5.14.1" accelerate
uv pip install --python "$JEV_VENV/bin/python" -e "$HOME/$JEV_REMOTE_DIR"

"$JEV_VENV/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "cuda build", torch.version.cuda, "| transformers", transformers.__version__)'
