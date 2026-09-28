# Sourced by the Marvin scripts. Paths on the cluster; override via environment if needed.
export PATH="$HOME/.local/bin:$PATH"                                  # uv (already installed on Marvin)
export JEV_REMOTE_DIR="${JEV_REMOTE_DIR:-jev-benchmarking}"           # code, relative to $HOME
export JEV_WS="${JEV_WS:-$HOME/workspaces/jevbench}"                  # Lustre SSD workspace (see setup_env.sh)
export JEV_VENV="${JEV_VENV:-$HOME/.venvs/jevbench}"                  # own env, separate from VNB's vLLM env
export HF_HOME="$JEV_WS/hf_home"                                      # datasets + hub caches on Lustre
export HF_HUB_CACHE="$JEV_WS/hf_home/hub"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"                 # token stays where `hf auth login` put it
export JEV_RUNS="$JEV_WS/runs"
export PYTHONUNBUFFERED=1
