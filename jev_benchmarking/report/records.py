"""Per-example, per-question records (long format) built from the response cache, memoized as parquet."""

import pandas as pd

from jev_benchmarking.cache import ResponseCache, request_key
from jev_benchmarking.config import MODEL, ROOT
from jev_benchmarking.tasks import TASKS

RECORDS_DIR = ROOT / "cache" / "records"


def _row(task, e, head: str, ans: dict) -> dict:
    kind = task.heads[head]
    gold = e.gold[head]
    r = {"task": task.name, "uid": e.uid, "subset": e.subset, "group": e.group, "head": head, "kind": kind}
    if kind == "choice":
        golds = gold if isinstance(gold, (set, frozenset)) else {gold}
        probs = ans["probabilities"]
        r.update(
            gold=next(iter(sorted(golds))),
            pred=ans["choice"],
            correct=float(ans["choice"] in golds),
            prob=max(probs.values()),  # top-label probability
            confidence=ans["confidence"],
            n_options=len(probs),
        )
    elif kind == "binary":
        r.update(gold=str(bool(gold)), gold_num=float(bool(gold)), prob=ans["noul"], pred=str(ans["noul"] >= 0.5))
        r["correct"] = float(r["pred"] == r["gold"])
    else:
        probs = ans["probabilities"]
        r.update(
            gold_num=float(gold),
            score=ans["score"],
            pred=max(probs, key=probs.get),
            confidence=ans["confidence"],
            n_options=len(probs),
        )
    return r


def build_records(task_name: str, split: str = "eval", cache: ResponseCache | None = None) -> pd.DataFrame:
    task = TASKS[task_name]
    cache = cache or ResponseCache()
    examples = task.examples(split)
    responses = cache.get_many([request_key(MODEL, e.state, e.questions) for e in examples])
    rows = []
    for e in examples:
        resp = responses.get(request_key(MODEL, e.state, e.questions))
        if resp is None:
            continue
        for head in task.heads:
            rows.append(_row(task, e, head, resp["answers"][head]))
    return pd.DataFrame(rows)


def load_records(task_name: str, split: str = "eval", refresh: bool = False) -> pd.DataFrame:
    path = RECORDS_DIR / split / f"{task_name}.parquet"
    if path.is_file() and not refresh:
        return pd.read_parquet(path)
    df = build_records(task_name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df
