#!/usr/bin/env bash
# One-time setup of the BabyAI-Text environment on the Marvin LOGIN node (needs internet):
#   ssh marvin 'bash jev-benchmarking/scripts/marvin/setup_babyai_env.sh'
#
# Runs the repository's setup_babyai_text_env.sh in the code directory (creates .venv-babyai there) and adds
# what the open-model backend needs; transformers 5.14.1 works with the environment's numpy 1.23.1.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$HOME/$JEV_REMOTE_DIR"
bash scripts/setup_babyai_text_env.sh
uv pip install --python "$JEV_BABYAI_VENV/bin/python" "transformers==5.14.1" accelerate
"$JEV_BABYAI_VENV/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "cuda build", torch.version.cuda, "| transformers", transformers.__version__)'
