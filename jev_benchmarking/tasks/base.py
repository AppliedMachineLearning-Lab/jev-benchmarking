"""Task abstraction: turn dataset rows into Jev requests plus gold labels.

Every task sends one request per example: the example goes in `state`, and all of the task's questions
are fanned out inside that one request. Each question id is a "head" with a kind that decides how it
is scored:

- ``choice``: gold is an option key (or a set of acceptable keys).
- ``binary``: a Noul; gold is a bool.
- ``score``: a Score; gold is a number on the same scale as the level indices.
"""

import hashlib
import json
import logging
import string
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

from datasets import get_dataset_config_names, load_dataset

from jev_benchmarking.config import ROOT

log = logging.getLogger(__name__)
_CONFIG_CACHE = ROOT / "cache" / "hf_config_names.json"


def hub_config_names(path: str) -> list[str]:
    """`get_dataset_config_names`, memoized on disk so datasets can be rebuilt with HF_HUB_OFFLINE=1
    (and without ~2 s of Hub round-trips per config)."""
    cached = json.loads(_CONFIG_CACHE.read_text()) if _CONFIG_CACHE.is_file() else {}
    if path not in cached:
        cached[path] = get_dataset_config_names(path)
        _CONFIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_CACHE.write_text(json.dumps(cached, indent=1))
    return cached[path]


HeadKind = Literal["choice", "binary", "score"]


@dataclass(frozen=True)
class Example:
    uid: str  # unique within the task, stable across runs
    subset: str  # dataset config / language / subject; "" when the dataset has none
    state: Any
    questions: dict[str, dict]
    gold: dict[str, Any]  # question id -> gold value
    group: str = ""  # for grouped metrics (e.g. SummEval per-document correlation)


@dataclass(frozen=True)
class Primary:
    """The headline number for the summary table."""

    head: str  # question id, or "*" for a metric aggregated over all heads
    metric: str


# ----------------------------------------------------------------------------------------------------
# Question constructors (raw API dicts; the cache keys on their exact JSON).


def choice(instructions: Any, criteria: dict[str, Any]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(instructions: Any, true: str | None = None, false: str | None = None) -> dict:
    q: dict = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v}
    return q


def score(instructions: Any, levels: list[Any]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


def option_keys(n: int) -> list[str]:
    """A, B, C, ... for up to 26 options, else AA, AB, ... (never numbers: they can collide with numeric
    answers, e.g. key "36" for "the element with atomic number 36")."""
    letters = string.ascii_uppercase
    if n <= 26:
        return list(letters[:n])
    return [a + b for a in letters for b in letters][:n]


def mc_criteria(options: list[str]) -> dict[str, str]:
    return dict(zip(option_keys(len(options)), options))


# ----------------------------------------------------------------------------------------------------


def _rank(*parts: str) -> str:
    return hashlib.sha256("/".join(parts).encode()).hexdigest()


class Task:
    """Subclasses set the class attributes and implement `build`."""

    name: str = ""
    hf_path: str = ""
    configs: tuple[str | None, ...] = (None,)
    splits: dict[str, str | None] = {"eval": "test", "dev": "validation"}
    heads: dict[str, HeadKind] = {"answer": "choice"}
    primary: Primary = Primary("answer", "accuracy")
    multilabel: bool = False  # binary heads form one multi-label problem (adds micro/macro-F1)
    max_per_config: int | None = None  # cap rows per config (applied before --limit)
    description: str = ""

    def build(self, row: dict, config: str | None) -> Example | list[Example] | None:
        raise NotImplementedError

    def extra_metrics(self, exs: list[Example], answers: list[dict], heads: dict[str, dict]) -> dict:
        """Task-specific metrics that go into the `aggregate` block (e.g. OOS recall, mean over heads)."""
        return {}

    def list_configs(self) -> list[str | None]:
        return list(self.configs)

    def load_split(self, config: str | None, split: str):
        return load_dataset(self.hf_path, config, split=split)

    def uid(self, config: str | None, idx: int) -> str:
        return f"{config}/{idx}" if config else str(idx)

    def examples(self, split: str = "eval", limit: int | None = None, subsets: list[str] | None = None) -> list[Example]:
        """Deterministic sample: rows are ordered by a hash of their uid, so a smaller `limit` always
        selects a subset of a larger one and pilot results stay cached for the full run."""
        hf_split = self.splits.get(split)
        if hf_split is None:
            raise ValueError(f"{self.name} has no '{split}' split")
        configs = [c for c in self.list_configs() if not subsets or c in subsets]
        if not configs:
            raise ValueError(f"{self.name}: no configs match {subsets}")

        # Load each config lazily and take rows round-robin over configs (configs also hash-ordered),
        # so `limit` spreads evenly across languages/subjects.
        ordered = sorted(configs, key=lambda c: _rank(self.name, str(c)))
        per_config: list[tuple[str | None, Iterator[Example]]] = [(c, self._iter_config(c, hf_split)) for c in ordered]
        out: list[Example] = []
        self.skipped_configs: dict[str, str] = {}
        while per_config and (limit is None or len(out) < limit):
            alive = []
            for c, it in per_config:
                if limit is not None and len(out) >= limit:
                    alive.append((c, it))
                    break
                try:
                    ex = next(it, None)
                except Exception as e:  # a single broken config must not take down the whole task
                    self.skipped_configs[str(c)] = f"{type(e).__name__}: {str(e)[:150]}"
                    log.warning("%s: skipping config %s (%s)", self.name, c, self.skipped_configs[str(c)])
                    continue
                if ex is not None:
                    out.append(ex)
                    alive.append((c, it))
            per_config = alive
        return out

    def _load_with_retry(self, config: str | None, hf_split: str, attempts: int = 3):
        for attempt in range(attempts):
            try:
                return self.load_split(config, hf_split)
            except Exception:
                if attempt == attempts - 1:
                    raise
                time.sleep(2 * (attempt + 1))  # transient Hub/DNS errors are common with many configs

    def _iter_config(self, config: str | None, hf_split: str) -> Iterator[Example]:
        ds = self._load_with_retry(config, hf_split)
        order = sorted(range(len(ds)), key=lambda i: _rank(self.name, self.uid(config, i)))
        taken = 0
        for i in order:
            built = self.build(ds[i], config)
            if built is None:
                continue
            # A row may expand into several examples (e.g. SummEval: one document, 16 summaries).
            items = [(self.uid(config, i), built)] if isinstance(built, Example) else [
                (f"{self.uid(config, i)}/{j}", b) for j, b in enumerate(built)
            ]
            for uid, b in items:
                if self.max_per_config is not None and taken >= self.max_per_config:
                    return
                taken += 1
                yield Example(uid, b.subset or config or "", b.state, b.questions, b.gold, b.group)


def ex(state: Any, questions: dict[str, dict], gold: dict[str, Any], group: str = "", subset: str = "") -> Example:
    """Shorthand used by `build`; uid is filled in by `Task`, subset defaults to the config name."""
    return Example("", subset, state, questions, gold, group)
