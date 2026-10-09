"""ALFWorld (text): household tasks played through TextWorld; one Choice over the admissible commands per step.

Splits follow the ALFWorld papers: `eval` = valid_unseen (134 games, unseen rooms, "out of distribution"),
`dev` = valid_seen (140 games). An episode succeeds if the task is completed within `max_steps` (50, as in
ReAct and most ALFWorld agent papers).

The request state holds the task, the initial room description (which lists every place in the room
and is never repeated later), the current observation and the full history of commands and their
observations since the start of the episode (at most 50 entries). The model chooses among the commands TextWorld lists as
admissible in the current state ("help" removed), so every answer is executable. The handcoded ALFWorld
expert runs alongside and its next command is logged as `oracle.expert` (never sent to the model) for
per-step agreement.

Setup: `pip install "alfworld[full]"` and `alfworld-download` (about 2 GB into $ALFWORLD_DATA).
"""

import glob
import hashlib
import os
import re
import sys

from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS, Decision, EpisodeSpec, SequentialEnv, StepResult, _only_asked
from jev_benchmarking.tasks.base import choice, mc_criteria, noul, option_keys

SPLITS = {"eval": "valid_unseen", "dev": "valid_seen", "unseen": "valid_unseen", "seen": "valid_seen"}
INTRO = re.compile(r"^-= Welcome to TextWorld, ALFRED! =-\s*", re.M)

ACTION_INSTRUCTIONS = (
    "You are controlling an agent in a household. Choose the command that makes the most progress toward "
    "completing the task, given the current observation and everything that happened so far."
)


def _patch_textworld_for_py313() -> None:
    """TextWorld's grammar evaluates templates with `locals().update(...)`, which no longer affects `eval`
    under Python 3.13 (PEP 667). Pass the variables explicitly instead. No-op on older Pythons."""
    if sys.version_info < (3, 13):
        return
    from textworld.envs.pddl import textgen

    def derive(self, context=None):
        context = context or self.context
        return [textgen.TerminalSymbol(eval(self.expression, {}, dict(context["variables"])))]

    textgen.EvalSymbol.derive = derive


class ALFWorld(SequentialEnv):
    name = "alfworld"
    max_steps = 50
    history_window: int | None = None  # None = full history (the episode is capped at max_steps anyway)
    description = "Text household tasks; Choice over admissible commands; success within 50 steps."

    def __init__(self, data_dir: str | None = None, with_expert: bool = True, ask_success: bool = False):
        _patch_textworld_for_py313()
        from alfworld.info import ALFWORLD_DATA

        self.data_dir = data_dir or ALFWORLD_DATA
        self.with_expert = with_expert
        self.ask_success = ask_success
        self.env = None

    def episodes(self, split: str = "eval", limit: int | None = None) -> list[EpisodeSpec]:
        root = os.path.join(self.data_dir, "json_2.1.1", SPLITS[split])
        files = sorted(glob.glob(os.path.join(root, "*", "*", "game.tw-pddl")))
        if not files:
            raise FileNotFoundError(f"no ALFWorld games under {root}; run `alfworld-download`")
        specs = []
        for f in files:
            rel = os.path.relpath(os.path.dirname(f), root)  # "<task>-<obj>-...-<scene>/trial_..."
            task_type = rel.split("-")[0]
            specs.append(EpisodeSpec(f"{SPLITS[split]}/{rel}", task_type, f))
        # Hash order (as in the single-step suite), so `--limit` takes a mixed sample of task types and a
        # smaller limit is always a subset of a larger one. Alphabetical order would pick one task type.
        specs.sort(key=lambda sp: hashlib.sha256(f"alfworld/{sp.uid}".encode()).hexdigest())
        return specs[:limit] if limit is not None else specs

    def reset(self, spec: EpisodeSpec) -> None:
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredExpert, AlfredExpertType, AlfredInfos

        self.close()
        extras = ["gamefile"] + (["expert_plan"] if self.with_expert else [])
        infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=self.with_expert, extras=extras)
        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        if self.with_expert:
            wrappers.append(AlfredExpert(AlfredExpertType.HANDCODED))
        env_id = textworld.gym.register_games([spec.payload], infos, batch_size=1, asynchronous=False,
                                              max_episode_steps=self.max_steps, wrappers=wrappers)
        self.env = textworld.gym.make(env_id)
        obs, info = self.env.reset()
        text = INTRO.sub("", obs[0]).strip()
        intro, _, task = text.partition("Your task is to:")
        self.task = task.strip()
        self.observation = intro.strip()
        self.initial_observation = self.observation
        self.info = info
        self.history: list[dict] = []
        self.won = False
        self.done = False

    def _expert(self) -> str | None:
        plan = (self.info.get("extra.expert_plan") or [None])[0]
        return plan[0] if plan else None

    def decision(self) -> Decision | None:
        if self.done:
            return None
        commands = [c for c in self.info["admissible_commands"][0] if c != "help"]
        t = len(self.history)
        left = self.max_steps - t
        state = {
            "task": self.task,
            "initial_observation": self.initial_observation,
            "current_observation": self.observation,
            "history": list(self.history) if self.history_window is None else self.history[-self.history_window :],  # a copy, not the live list
            "steps_taken": t,
            "steps_left": left,
        }
        questions = {ACTION: choice(ACTION_INSTRUCTIONS, mc_criteria(commands))}
        if self.ask_success:
            questions[SUCCESS] = noul(f"Will the task be completed within the remaining {left} steps?")
        actions = dict(zip(option_keys(len(commands)), commands))
        return Decision(state, questions, actions, {"expert": self._expert()} if self.with_expert else {})

    def step(self, action: str) -> StepResult:
        obs, _, done, info = self.env.step([action])
        self.info = info
        self.observation = obs[0].strip()
        self.history.append({"command": action, "observation": self.observation})
        self.won = bool(info["won"][0])
        self.done = bool(done[0])
        return StepResult(self.done, self.won, float(self.won), self.observation)

    def outcome(self) -> StepResult:
        return StepResult(self.done, self.won, float(self.won))

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


# ----------------------------------------------------------------------------------------------------
# Scripted reference policies. They need the expert's command, which only the env knows, so the env
# exposes it through `oracle`; `ExpertBackend` in the CLI reads it from the last decision.


def random_policy(state: dict, questions: dict) -> dict:
    from jev_benchmarking.sequential.backends import choice_answer, noul_answer

    keys = list(questions[ACTION]["criteria"])
    return _only_asked(questions, {ACTION: choice_answer({k: 1.0 for k in keys}), SUCCESS: noul_answer(0.1)})


class ExpertBackend:
    """Follows the handcoded ALFWorld expert (an upper reference, not a model). Built by the CLI."""

    @staticmethod
    def make(cache=None):
        from jev_benchmarking.sequential.backends import FunctionBackend, choice_answer, noul_answer

        holder: dict = {}

        def fn(state: dict, questions: dict) -> dict:
            d = holder["decision"]
            expert = d.oracle.get("expert")
            keys = list(d.actions)
            probs = {k: float(d.actions[k] == expert) for k in keys}
            if not any(probs.values()):  # expert has no plan or proposes a non-admissible command
                probs = {k: 1.0 for k in keys}
            return _only_asked(questions, {ACTION: choice_answer(probs), SUCCESS: noul_answer(0.9)})

        backend = FunctionBackend("expert", fn, cache)
        backend.observe = lambda d: holder.__setitem__("decision", d)
        return backend


SCRIPTED = {"random": random_policy}
