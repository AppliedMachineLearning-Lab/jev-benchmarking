"""Offline checks of sampling and request construction (fake rows, no downloads, no API)."""

import pytest

from jev_benchmarking.cache import ResponseCache, request_key
from jev_benchmarking.evaluate import evaluate
from jev_benchmarking.tasks import TASKS
from jev_benchmarking.tasks.base import Primary, Task, choice, ex, option_keys


class FakeTask(Task):
    name = "fake"
    configs = ("x", "y")

    def load_split(self, config, split):
        return [{"text": f"{config}-{i}", "label": i % 2} for i in range(10)]

    def build(self, row, config):
        q = choice("Which label fits `text`?", {"even": None, "odd": None})
        return ex({"text": row["text"]}, {"answer": q}, {"answer": ("even", "odd")[row["label"]]})


def test_limit_is_nested_and_balanced_over_configs():
    task = FakeTask()
    small = task.examples(limit=4)
    large = task.examples(limit=12)
    assert [e.uid for e in small] == [e.uid for e in large[:4]]
    assert {e.subset for e in small} == {"x", "y"}
    assert len(task.examples()) == 20
    assert len({e.uid for e in task.examples()}) == 20


def test_evaluate_reads_only_cached_answers(tmp_path):
    task = FakeTask()
    cache = ResponseCache(tmp_path / "c.db")
    exs = task.examples(limit=6)
    for e in exs[:4]:
        pred = e.gold["answer"]
        probs = {"even": 0.8 if pred == "even" else 0.2, "odd": 0.2 if pred == "even" else 0.8}
        body = {"model": "jev-test", "answers": {"answer": {"type": "choice", "choice": pred, "probabilities": probs, "confidence": 0.7}}, "usage": {"input_tokens": 100, "output_tokens": 5}}
        cache.put(request_key("jev-1.13.0", e.state, e.questions), task.name, body, 120, 0.1)
    res = evaluate(task, exs, cache, with_ci=False)
    assert res["n_answered"] == 4
    assert res["primary"]["value"] == 1.0
    assert cache.spent_usd() == pytest.approx(400 * 0.042 / 1e6)


def test_option_keys():
    assert option_keys(3) == ["A", "B", "C"]
    assert option_keys(30)[:2] == ["AA", "AB"]
    assert len(set(option_keys(255))) == 255


def test_registry_is_consistent():
    for name, task in TASKS.items():
        assert task.name == name
        assert task.hf_path
        assert task.heads
        assert task.primary.head == "*" or task.primary.head in task.heads, name
        assert task.splits.get("eval"), name


def test_toxigen_build_matches_heads():
    t = TASKS["toxigen"]
    row = {"text": " some text ", "toxicity_ai": 3.0, "toxicity_human": 3.0}
    e = t.build(row, "annotated")
    assert set(e.questions) == set(t.heads) == set(e.gold)
    assert e.gold == {"toxic": True, "toxicity": 2.0}


def test_multilabel_builds_have_all_heads():
    goem = TASKS["go_emotions"].build({"text": "wow", "labels": [0, 27]}, "simplified")
    assert set(goem.questions) == set(TASKS["go_emotions"].heads)
    assert goem.gold["admiration"] and goem.gold["neutral"] and not goem.gold["anger"]
    tos = TASKS["unfair_tos"].build({"text": "x", "labels": [7]}, "unfair_tos")
    assert set(tos.questions) == set(TASKS["unfair_tos"].heads)
    assert tos.gold["arbitration"] and not tos.gold["jurisdiction"]


def test_primary_star_tasks_define_aggregate():
    for task in TASKS.values():
        if task.primary == Primary("*", task.primary.metric):
            assert task.multilabel or type(task).extra_metrics is not Task.extra_metrics, task.name
