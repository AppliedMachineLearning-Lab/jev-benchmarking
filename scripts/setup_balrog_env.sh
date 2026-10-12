#!/usr/bin/env bash
# Environment for BabyAI-Text and TextWorld, which run on BALROG's own code with BALROG's pinned
# dependencies (its forks of TextWorld and minigrid, gym 0.23, tatsu 5.8.3). These conflict with ALFWorld's
# TextWorld, so they get their own virtual environment with Python 3.11. Needs uv (pip install uv), git and a C
# compiler; building BALROG's TextWorld fork downloads Inform 7.
#
#   bash scripts/setup_balrog_env.sh
#   source .venv-balrog/bin/activate
#   python scripts/run_sequential.py babyai_text --backend jev --limit 5
set -euo pipefail

BALROG=b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5        # BALROG commit
TEXTWORLD=1d56f4765e7b28f04a68eff0bce5a6fe6ecd1a9f     # balrog-ai/TextWorld, as pinned by BALROG's setup.py
MINIGRID=cf73dd148e51276bd675a37e3bfb0bf2b42329b2      # BartekCupial/Minigrid, as pinned by BALROG's setup.py

uv venv -p 3.11 .venv-balrog
source .venv-balrog/bin/activate
uv pip install -e . rlcard pytest httpx2 \
    "gym==0.23" "tatsu==5.8.3" omegaconf pillow "setuptools<81" balrog-nle \
    "textworld @ git+https://github.com/balrog-ai/TextWorld.git@${TEXTWORLD}" \
    "minigrid @ git+https://github.com/BartekCupial/Minigrid.git@${MINIGRID}"
# BALROG itself without its other environments' dependencies (NetHack agents, Crafter, MiniHack, LLM clients).
uv pip install --no-deps "balrog @ git+https://github.com/balrog-ai/BALROG.git@${BALROG}"
python -W ignore -c "import balrog, textworld, minigrid; print('BALROG environment ready')"
