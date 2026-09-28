"""Render a Jev question (state + typed question) as a chat prompt for an open-weight LLM.

Every answer option gets a short code that is a single token in the model's vocabulary (A, B, ... / AA,
AB, ... / Yes, No / 0-9), so one forward pass over the prompt yields the exact probability of every
option: the next-token distribution restricted to the option codes and renormalized.
"""

import json
import re
import string
from dataclasses import dataclass
from typing import Any

# Choice option keys that carry no meaning of their own (MC letters, ARC's "1".."4", our AA, AB, ...).
_ANONYMOUS_KEY = re.compile(r"^(?:[A-Z]{1,2}|\d+)$")


@dataclass(frozen=True)
class RenderedQuestion:
    text: str  # user message content
    codes: list[str]  # answer codes in option order
    keys: list[str]  # Jev answer key per code (Choice keys, "yes"/"no", level indices as strings)
    kind: str  # "choice" | "noul" | "score"


def _fmt(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)


def choice_codes(n: int, is_single_token=lambda code: True) -> list[str]:
    """A..Z for up to 26 options, else two-letter codes (AA, AB, ...), skipping codes that are not a
    single token for the model's tokenizer."""
    letters = string.ascii_uppercase
    pool = list(letters) if n <= 26 else [a + b for a in letters for b in letters]
    codes = [c for c in pool if is_single_token(c)]
    if len(codes) < n:
        raise ValueError(f"only {len(codes)} single-token codes available for {n} options")
    return codes[:n]


def render(state: Any, question: dict, is_single_token=lambda code: True) -> RenderedQuestion:
    parts = ["Answer the question about the following state.", "State:\n" + _fmt(state), "Question:\n" + _fmt(question["instructions"])]
    kind = question["type"]
    if kind == "choice":
        keys = list(question["criteria"])
        codes = choice_codes(len(keys), is_single_token)
        lines = []
        for code, key in zip(codes, keys):
            desc = question["criteria"][key]
            if _ANONYMOUS_KEY.match(key):
                label = _fmt(desc) if desc is not None else key
            else:
                label = key if desc is None else f"{key} ({_fmt(desc)})"
            lines.append(f"{code}: {label}")
        parts.append("Options:\n" + "\n".join(lines))
        parts.append("Reply with the code of the correct option only.")
    elif kind == "noul":
        codes, keys = ["Yes", "No"], ["yes", "no"]
        crit = question.get("criteria") or {}
        meaning = [f"{word} means: {_fmt(crit[k])}" for word, k in (("Yes", "true"), ("No", "false")) if k in crit]
        if meaning:
            parts.append("\n".join(meaning))
        parts.append("Reply with Yes or No only.")
    elif kind == "score":
        levels = question["criteria"]
        if len(levels) > 10:
            raise ValueError("more than 10 levels cannot be coded as single digits")
        codes = [str(i) for i in range(len(levels))]
        keys = list(codes)
        parts.append("Levels:\n" + "\n".join(f"{i}: {_fmt(level)}" for i, level in enumerate(levels)))
        parts.append("Reply with the number of the level only.")
    else:
        raise ValueError(f"unknown question type {kind!r}")
    return RenderedQuestion("\n\n".join(parts), codes, keys, kind)


def confidence(probs: list[float]) -> float:
    """Jev's documented confidence for a distribution over K options: (K * p_max - 1) / (K - 1), clipped."""
    k = len(probs)
    return max(0.0, min(1.0, (k * max(probs) - 1) / (k - 1))) if k > 1 else 1.0


def to_answer(rq: RenderedQuestion, probs: list[float], question: dict) -> dict:
    """Probabilities over rq.codes (renormalized) -> an answer in Jev's response format."""
    if rq.kind == "noul":
        return {"type": "noul", "noul": probs[0]}
    dist = dict(zip(rq.keys, probs))
    best = max(dist, key=dist.get)
    if rq.kind == "choice":
        return {"type": "choice", "choice": best, "probabilities": dist, "confidence": confidence(probs)}
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(probs)),
        "legend": {str(i): level for i, level in enumerate(question["criteria"])},
        "probabilities": dist,
        "confidence": confidence(probs),
    }
