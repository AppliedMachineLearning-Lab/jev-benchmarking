"""Metrics per head kind. All functions take parallel lists of gold values and raw API answers."""

import math
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

N_BINS = 15


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error with equal-width bins."""
    if len(conf) == 0:
        return math.nan
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total / len(conf))


def selective_accuracy(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy on the `coverage` fraction of examples with the highest confidence."""
    k = max(1, int(round(coverage * len(conf))))
    top = np.argsort(-conf, kind="stable")[:k]
    return float(correct[top].mean())


def aurc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    order = np.argsort(-conf, kind="stable")
    risks = np.cumsum(1 - correct[order]) / np.arange(1, len(conf) + 1)
    return float(risks.mean())


def _safe(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except ValueError:  # e.g. AUROC with a single class present
        return math.nan


def choice_metrics(golds: list, answers: list[dict]) -> dict:
    gold_sets = [g if isinstance(g, (set, frozenset, list, tuple)) else {g} for g in golds]
    preds = [a["choice"] for a in answers]
    correct = np.array([p in g for p, g in zip(preds, gold_sets)], dtype=float)
    top_p = np.array([max(a["probabilities"].values()) for a in answers])
    api_conf = np.array([a["confidence"] for a in answers])
    brier = np.mean(
        [sum((p - (k in g)) ** 2 for k, p in a["probabilities"].items()) for a, g in zip(answers, gold_sets)]
    )
    m = {
        "accuracy": float(correct.mean()),
        "ece": ece(top_p, correct),
        "brier": float(brier),
        "mean_top_prob": float(top_p.mean()),
        "mean_confidence": float(api_conf.mean()),
        "sel_acc@50": selective_accuracy(api_conf, correct, 0.5),
        "sel_acc@80": selective_accuracy(api_conf, correct, 0.8),
        "aurc": aurc(api_conf, correct),
    }
    # Macro-F1 only makes sense when every example shares one label set (classification, not per-item MC).
    label_sets = {tuple(sorted(a["probabilities"])) for a in answers}
    if len(label_sets) == 1:
        first_gold = [next(iter(sorted(g))) for g in gold_sets]
        m["macro_f1"] = float(f1_score(first_gold, preds, average="macro", zero_division=0))
    return m


def binary_metrics(golds: list[bool], answers: list[dict], threshold: float = 0.5) -> dict:
    y = np.array(golds, dtype=int)
    p = np.array([a["noul"] for a in answers], dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / max(1, (y == 1).sum())
    tnr = int(((pred == 0) & (y == 0)).sum()) / max(1, (y == 0).sum())
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float((tpr + tnr) / 2),
        "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tpr),
        "auroc": _safe(roc_auc_score, y, p),
        "auprc": _safe(average_precision_score, y, p) if y.any() else math.nan,
        "ece": ece(p, y.astype(float)),
        "brier": float(np.mean((p - y) ** 2)),
        "positive_rate": float(y.mean()),
    }


def score_metrics(golds: list[float], answers: list[dict], groups: list[str] | None = None) -> dict:
    y = np.array(golds, dtype=float)
    s = np.array([a["score"] for a in answers], dtype=float)
    argmax = np.array([int(max(a["probabilities"], key=lambda k: a["probabilities"][k])) for a in answers])
    m = {
        "spearman": _safe(lambda: stats.spearmanr(y, s).statistic),
        "pearson": _safe(lambda: stats.pearsonr(y, s).statistic),
        "kendall": _safe(lambda: stats.kendalltau(y, s).statistic),
        "mae": float(np.abs(y - s).mean()),
    }
    if np.allclose(y, np.round(y)):
        m["argmax_accuracy"] = float((argmax == y).mean())
    if groups and any(groups):
        # "Summary-level" correlation (SummEval convention): correlate within each group, then average.
        by_group = defaultdict(list)
        for i, g in enumerate(groups):
            by_group[g].append(i)
        rhos, taus = [], []
        for idx in by_group.values():
            if len(idx) > 2 and np.ptp(y[idx]) > 0 and np.ptp(s[idx]) > 0:
                rhos.append(stats.spearmanr(y[idx], s[idx]).statistic)
                taus.append(stats.kendalltau(y[idx], s[idx]).statistic)
        m["group_spearman"] = float(np.mean(rhos)) if rhos else math.nan
        m["group_kendall"] = float(np.mean(taus)) if taus else math.nan
    return m


def multilabel_metrics(gold_matrix: np.ndarray, prob_matrix: np.ndarray, threshold: float = 0.5) -> dict:
    """Rows = examples, columns = labels."""
    pred = (prob_matrix >= threshold).astype(int)
    both = [j for j in range(gold_matrix.shape[1]) if 0 < gold_matrix[:, j].sum() < len(gold_matrix)]
    aurocs = [roc_auc_score(gold_matrix[:, j], prob_matrix[:, j]) for j in both]
    auprcs = [average_precision_score(gold_matrix[:, j], prob_matrix[:, j]) for j in both]
    return {
        "micro_f1": float(f1_score(gold_matrix, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(gold_matrix, pred, average="macro", zero_division=0)),
        "exact_match": float((pred == gold_matrix).all(axis=1).mean()),
        "mean_auroc": float(np.mean(aurocs)) if aurocs else math.nan,
        "mean_auprc": float(np.mean(auprcs)) if auprcs else math.nan,
    }
