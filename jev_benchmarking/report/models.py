"""Models compared in the paper: display label -> model id, and loaders for their results."""

import json

from jev_benchmarking.config import MODEL, RESULTS_DIR, model_slug
from jev_benchmarking.tasks import TASKS

PAPER_MODELS: dict[str, str] = {
    "Jev": MODEL,
    "Gemma-4-E4B": "hf:google/gemma-4-E4B-it",
    "Qwen3.8-27B": "hf:Qwen/Qwen3.8-27B",
}


def results_dir(model: str, split: str = "eval"):
    slug = model_slug(model)
    return RESULTS_DIR / split / "models" / slug if slug else RESULTS_DIR / split


def load_model_results(model: str, split: str = "eval") -> dict[str, dict]:
    d = results_dir(model, split)
    return {t: json.loads((d / f"{t}.json").read_text()) for t in TASKS if (d / f"{t}.json").is_file()}


def available_models(split: str = "eval") -> dict[str, dict[str, dict]]:
    """Label -> results for every paper model whose results cover all tasks. Partial runs are left out; a
    handful of rejected requests are tolerated (Jev rejected one over-long BIG-bench item)."""
    out = {}
    for label, model in PAPER_MODELS.items():
        res = load_model_results(model, split)
        if len(res) == len(TASKS) and all(r["n_answered"] >= 0.999 * r["n_examples"] for r in res.values()):
            out[label] = res
    return out
