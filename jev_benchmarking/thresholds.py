"""Per-question decision thresholds for Noul answers, chosen on a training/validation sample and applied
unchanged to the evaluation split (never tuned on evaluation data)."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing F1 on (y, p); ties go to the value closest to 0.5. Without positives: 0.5."""
    if y.sum() == 0:
        return 0.5
    scores = np.array([f1_score(y, p >= t, zero_division=0) for t in GRID])
    best = np.flatnonzero(scores == scores.max())
    return float(GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]])


def tune(dev: pd.DataFrame) -> dict[str, float]:
    """One threshold per binary head from dev records (long format, see report.records)."""
    b = dev[dev["kind"] == "binary"]
    return {h: best_threshold(g["gold_num"].to_numpy(), g["prob"].to_numpy()) for h, g in b.groupby("head")}


def apply(records: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Evaluation records with a `pred_tuned` column (0/1) next to the fixed-0.5 prediction."""
    b = records[records["kind"] == "binary"].copy()
    b["pred_fixed"] = (b["prob"] >= 0.5).astype(int)
    b["pred_tuned"] = (b["prob"] >= b["head"].map(thresholds).fillna(0.5)).astype(int)
    return b


def f1_scores(b: pd.DataFrame, col: str, multilabel: bool) -> dict[str, float]:
    """Per-head F1, plus micro/macro-F1 over heads for multi-label tasks."""
    out = {f"f1/{h}": f1_score(g["gold_num"], g[col], zero_division=0) for h, g in b.groupby("head")}
    if multilabel:
        wide_y = b.pivot(index="uid", columns="head", values="gold_num")
        wide_p = b.pivot(index="uid", columns="head", values=col)
        out["micro_f1"] = f1_score(wide_y, wide_p, average="micro", zero_division=0)
        out["macro_f1"] = f1_score(wide_y, wide_p, average="macro", zero_division=0)
    return out
