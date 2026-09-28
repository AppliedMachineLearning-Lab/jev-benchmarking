#!/usr/bin/env bash
# Download the model weights and every dataset into the workspace, on the LOGIN node (network I/O only),
# so that GPU jobs can run with HF_HUB_OFFLINE=1 regardless of compute-node internet access:
#   ssh marvin 'bash jev-benchmarking/scripts/marvin/prefetch.sh'
# LLM-AggreFact is gated: needs the HF token at ~/.cache/huggingface/token with access granted.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
MODEL="${JEV_MODEL:-google/gemma-4-E4B-it}"
cd "$HOME/$JEV_REMOTE_DIR"

echo "=== model $MODEL ==="
"$JEV_VENV/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"

echo "=== datasets (eval + dev splits of all tasks and probes) ==="
"$JEV_VENV/bin/python" - <<'PY'
from jev_benchmarking.tasks import ALL_TASKS
for name, task in ALL_TASKS.items():
    for split in ("eval", "dev"):
        hf_split = task.splits.get(split)
        if hf_split is None:
            continue
        n = 0
        for config in task.list_configs():
            try:
                n += len(task.load_split(config, hf_split))
            except Exception as e:  # broken configs are skipped at run time too
                print(f"  ! {name}/{config}/{hf_split}: {type(e).__name__}")
        print(f"{name:<26} {split:<4} {n:>7} rows", flush=True)
PY
