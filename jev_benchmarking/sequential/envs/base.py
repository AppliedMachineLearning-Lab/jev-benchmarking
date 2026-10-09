"""Environment interface for sequential benchmarks.

An environment plays one episode at a time. At each step it exposes a `Decision`: the request (state and
typed questions) plus which question picks the action and how its option keys map to actions. The runner
asks a backend, executes the chosen action, and repeats until the episode ends.

Every step request carries the action question: a Choice over the currently legal actions. Optionally
(`ask_success = True`, off by default) it also carries a success question, a Noul asking whether the
episode will end in success. It is answered independently, never used to act, and only serves to check
calibration against the episode's outcome.
"""

from dataclasses import dataclass, field
from typing import Any

ACTION = "action"
SUCCESS = "success"


@dataclass(frozen=True)
class EpisodeSpec:
    uid: str  # stable id, e.g. "unseen/pick_and_place.../trial_..." or "seed-000123"
    subset: str = ""  # task type, difficulty, etc. for grouped metrics
    payload: Any = None  # whatever the env needs to reset (game file, seed)


@dataclass
class Decision:
    state: Any
    questions: dict[str, dict]
    actions: dict[str, str]  # option key of the action question -> action executed in the env
    oracle: dict = field(default_factory=dict)  # privileged info logged for metrics, never sent to the model


@dataclass
class StepResult:
    done: bool
    success: bool = False
    reward: float = 0.0
    feedback: str = ""


class SequentialEnv:
    name: str = ""
    max_steps: int = 50
    ask_success: bool = False  # add the optional success question to every request
    description: str = ""

    def episodes(self, split: str = "eval", limit: int | None = None) -> list[EpisodeSpec]:
        raise NotImplementedError

    def reset(self, spec: EpisodeSpec) -> None:
        raise NotImplementedError

    def decision(self) -> Decision | None:
        """The next decision, or None if the episode ended without needing one (e.g. a dealt blackjack)."""
        raise NotImplementedError

    def step(self, action: str) -> StepResult:
        raise NotImplementedError

    def outcome(self) -> StepResult:
        """Final result; called once the episode is over (by `done` or by hitting `max_steps`)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def _only_asked(questions: dict, answers: dict) -> dict:
    """Scripted policies answer every question they know; keep only those in the request."""
    return {k: v for k, v in answers.items() if k in questions}
