#!/usr/bin/env bash
# Environment for BabyAI-Text, which runs on the original packages of Carta et al. (GLAM repository, pinned
# commit): their forks of BabyAI and gym-minigrid and the babyai_text package. These need an old gym and
# numpy, so they get their own virtual environment with Python 3.10. Needs uv (pip install uv) and git.
#
#   bash scripts/setup_babyai_text_env.sh
#   source .venv-babyai/bin/activate
#   python scripts/run_sequential.py babyai_text --backend jev --limit 5
set -euo pipefail

GLAM=31847fb8c1db58bc8f595bea8f8cbaf26fb79dc2   # flowersteam/Grounding_LLMs_with_online_RL
SRC=.venv-babyai-src

uv venv -p 3.10 .venv-babyai
source .venv-babyai/bin/activate
if [ ! -d "$SRC" ]; then
    git clone -q --filter=blob:none --sparse https://github.com/flowersteam/Grounding_LLMs_with_online_RL.git "$SRC"
    git -C "$SRC" sparse-checkout set babyai-text
fi
git -C "$SRC" checkout -q "$GLAM"
# Dependencies of the original packages (babyai imports torch for its RL code; the levels and the bot do not
# use it), then the packages themselves as their README installs them.
uv pip install -e . rlcard pytest httpx2 "numpy==1.23.1" "gym==0.23.1" "setuptools<81" \
    torch blosc matplotlib ipython colorama termcolor tqdm
uv pip install --no-deps -e "$SRC/babyai-text/babyai" -e "$SRC/babyai-text/gym-minigrid" -e "$SRC/babyai-text"
python -W ignore -c "import gym, babyai_text; from babyai.bot import Bot; gym.make('BabyAI-MixedTrainLocal-v0'); print('BabyAI-Text environment ready')"
