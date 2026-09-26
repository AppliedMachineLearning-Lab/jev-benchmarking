"""Score a task's examples against cached responses (never calls the API)."""

import math
from collections import defaultdict

import numpy as np

from jev_benchmarking.cache import ResponseCache, request_key
from jev_benchmarking.config import MODEL
from jev_benchmarking.metrics import binary_metrics, choice_metrics, multilabel_metrics, score_metrics
from jev_benchmarking.tasks.base import Example, Task

N_BOOTSTRAP = 500
MIN_SUBSET_N = 20  # below this, per-subset numbers are noise


def _head_metrics(task: Task, head: str, exs: list[Example], answers: list[dict]) -> dict:
    kind = task.heads[head]
    golds = [e.gold[head] for e in exs]
    got = [a[head] for a in answers]
    if kind == "choice":
        return choice_metrics(golds, got)
    if kind == "binary":
        return binary_metrics(golds, got)
    return score_metrics(golds, got, [e.group for e in exs])


def compute(task: Task, exs: list[Example], answers: list[dict]) -> dict:
    """All metrics for a set of answered examples: per head, plus multi-label aggregates."""
    heads = {h: _head_metrics(task, h, exs, answers) for h in task.heads}
    aggregate = {}
    if task.multilabel:
        binary = [h for h, k in task.heads.items() if k == "binary"]
        gold = np.array([[e.gold[h] for h in binary] for e in exs], dtype=int)
        prob = np.array([[a[h]["noul"] for h in binary] for a in answers], dtype=float)
        aggregate.update(multilabel_metrics(gold, prob))
    aggregate.update(task.extra_metrics(exs, answers, heads))
    return {"heads": heads, "aggregate": aggregate}


def primary_value(task: Task, metrics: dict) -> float:
    p = task.primary
    source = metrics["aggregate"] if p.head == "*" else metrics["heads"][p.head]
    return source.get(p.metric, math.nan)


def bootstrap_ci(task: Task, exs: list[Example], answers: list[dict], seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(exs)
    vals = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        try:
            vals.append(primary_value(task, compute(task, [exs[i] for i in idx], [answers[i] for i in idx])))
        except (ValueError, ZeroDivisionError):
            continue
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return (math.nan, math.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def evaluate(task: Task, examples: list[Example], cache: ResponseCache, with_ci: bool = True) -> dict:
    keys = [request_key(MODEL, e.state, e.questions) for e in examples]
    responses = cache.get_many(keys)
    exs, answers, models = [], [], set()
    for e, k in zip(examples, keys):
        r = responses.get(k)
        if r is not None:
            exs.append(e)
            answers.append(r["answers"])
            models.add(r.get("model", ""))
    result = {
        "task": task.name,
        "n_examples": len(examples),
        "n_answered": len(exs),
        "models": sorted(models),
        "primary": {"head": task.primary.head, "metric": task.primary.metric},
    }
    if not exs:
        return result
    metrics = compute(task, exs, answers)
    result.update(metrics)
    result["primary"]["value"] = primary_value(task, metrics)
    if with_ci and len(exs) >= 10:
        result["primary"]["ci95"] = bootstrap_ci(task, exs, answers)

    by_subset = defaultdict(list)
    for i, e in enumerate(exs):
        if e.subset:
            by_subset[e.subset].append(i)
    if len(by_subset) > 1:
        result["subsets"] = {}
        for s, idx in sorted(by_subset.items()):
            if len(idx) < MIN_SUBSET_N:
                continue
            try:
                m = compute(task, [exs[i] for i in idx], [answers[i] for i in idx])
                result["subsets"][s] = {"n": len(idx), "primary": primary_value(task, m)}
            except ValueError:
                pass
    return result
