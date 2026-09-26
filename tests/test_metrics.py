import math

import numpy as np
import pytest

from jev_benchmarking.metrics import aurc, binary_metrics, choice_metrics, ece, multilabel_metrics, score_metrics, selective_accuracy


def test_ece_perfectly_calibrated_is_zero():
    conf = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert ece(conf, correct) == 0.0


def test_ece_overconfident():
    conf = np.full(10, 0.9)
    correct = np.array([1.0] * 5 + [0.0] * 5)
    assert ece(conf, correct) == pytest.approx(0.4)


def test_selective_accuracy_and_aurc():
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert selective_accuracy(conf, correct, 0.5) == 1.0
    assert selective_accuracy(conf, correct, 1.0) == 0.5
    assert aurc(conf, correct) == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)


def _choice(choice, probs, confidence=0.5):
    return {"type": "choice", "choice": choice, "probabilities": probs, "confidence": confidence}


def test_choice_metrics_with_gold_sets_and_macro_f1():
    answers = [
        _choice("a", {"a": 0.9, "b": 0.1}),
        _choice("b", {"a": 0.4, "b": 0.6}),
        _choice("a", {"a": 0.7, "b": 0.3}),
    ]
    m = choice_metrics(["a", frozenset({"a", "b"}), "b"], answers)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert "macro_f1" in m
    # brier for first example: (0.9-1)^2 + (0.1-0)^2
    assert m["brier"] > 0


def test_choice_metrics_skips_macro_f1_for_per_item_options():
    answers = [_choice("A", {"A": 1.0, "B": 0.0}), _choice("A", {"A": 1.0, "B": 0.0, "C": 0.0})]
    assert "macro_f1" not in choice_metrics(["A", "A"], answers)


def test_binary_metrics():
    answers = [{"noul": p} for p in (0.9, 0.8, 0.3, 0.1)]
    m = binary_metrics([True, False, True, False], answers)
    assert m["accuracy"] == 0.5
    assert m["auroc"] == pytest.approx(0.75)
    assert m["balanced_accuracy"] == 0.5


def test_binary_metrics_single_class_auroc_is_nan():
    m = binary_metrics([True, True], [{"noul": 0.9}, {"noul": 0.2}])
    assert math.isnan(m["auroc"])


def _score(s, probs):
    return {"type": "score", "score": s, "probabilities": probs, "confidence": 0.5}


def test_score_metrics_with_groups():
    answers = [_score(0.2, {"0": 0.8, "1": 0.2}), _score(0.9, {"0": 0.1, "1": 0.9}), _score(0.5, {"0": 0.5, "1": 0.5})] * 2
    golds = [0, 1, 1, 0, 1, 1]
    m = score_metrics(golds, answers, groups=["a", "a", "a", "b", "b", "b"])
    assert m["spearman"] > 0
    assert "argmax_accuracy" in m
    assert not math.isnan(m["group_spearman"])


def test_multilabel_metrics():
    gold = np.array([[1, 0], [0, 1], [1, 1]])
    prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    m = multilabel_metrics(gold, prob)
    assert m["exact_match"] == pytest.approx(2 / 3)
    assert 0 < m["micro_f1"] <= 1
