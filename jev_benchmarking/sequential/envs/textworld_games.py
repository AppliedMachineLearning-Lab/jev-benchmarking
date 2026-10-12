"""TextWorld (Côté et al., 2018) with its own game generators, on the official TextWorld package.

The games are made with TextWorld's `tw-make` tool, which ships three challenges from the literature:
  tw-coin_collector    a chain of rooms with a coin at the far end (Yuan et al., 2018),
  tw-treasure_hunter   find a given object, possibly in locked containers or behind locked doors,
  tw-cooking           read the cookbook, gather and process the ingredients, prepare and eat the meal (the
                       game type of the First TextWorld Problems competition).
Each challenge is used at three difficulty settings, with 10 games (seeds 1..10) each, 90 games in total:
  coin collector   levels 10, 110, 210: a quest of 10 rooms without distractor rooms, with distractor rooms
                   off the chain, and with distractor rooms placed randomly (TextWorld's level definition);
  treasure hunter  levels 5, 15, 25: TextWorld's easy, medium and hard modes (5, 10 and 20 rooms);
  cooking          1 ingredient in 1 room; 2 ingredients that need cutting and cooking, in 6 rooms with
                   closed containers and doors; 3 such ingredients in 9 rooms. (The limited-inventory skill,
                   --drop, is not used: TextWorld's solver finds no plan in those games.)
The games are generated once with the official generator and cached; the same TextWorld version and seed give
the same game. With `tw-make`'s defaults, the objective of coin collector and treasure hunter games spells out
every step of the walkthrough ("First, move south. And then, take a trip east. ..."), which turns them into
instruction following. These two challenges are therefore generated through the same code path as `tw-make`
with TextWorld's grammar option `only_last_action` set, which states only the goal ("pick up the coin from the
floor of the silent cubicle"); the world, the quest and the solver's moves are unchanged. Cooking games use
`tw-make`'s defaults; their objective never lists the steps. The request is built as for ALFWorld (objective, first observation, current room,
inventory, full history of commands and feedback, steps left), and the options are the commands TextWorld
admits in the current state. The reference action is the first command of the winning sequence TextWorld
computes from the current state (`policy_commands`); the solver knows the whole game, including where unseen
objects are and the recipe before the cookbook is read.
"""

import os
import re
import subprocess
from pathlib import Path

from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS, Decision, EpisodeSpec, SequentialEnv, StepResult, _only_asked
from jev_benchmarking.tasks.base import choice, mc_criteria, noul, option_keys

STEP_BUDGET = 50  # as for ALFWorld
GOAL_ONLY_OBJECTIVE = True  # coin collector and treasure hunter: state the goal, not the walkthrough
SEEDS = range(1, 11)
SETTINGS = {  # name -> tw-make arguments
    "coin_collector/level10": ["tw-coin_collector", "--level", "10"],
    "coin_collector/level110": ["tw-coin_collector", "--level", "110"],
    "coin_collector/level210": ["tw-coin_collector", "--level", "210"],
    "treasure_hunter/level5": ["tw-treasure_hunter", "--level", "5"],
    "treasure_hunter/level15": ["tw-treasure_hunter", "--level", "15"],
    "treasure_hunter/level25": ["tw-treasure_hunter", "--level", "25"],
    "cooking/easy": ["tw-cooking", "--recipe", "1", "--take", "1", "--go", "1"],
    "cooking/medium": ["tw-cooking", "--recipe", "2", "--take", "2", "--go", "6", "--open", "--cook", "--cut"],
    "cooking/hard": ["tw-cooking", "--recipe", "3", "--take", "3", "--go", "9", "--open", "--cook", "--cut"],
}

ACTION_Q_TEXT = "Which command brings the player closest to completing the objective?"
SUCCESS_Q = noul("Will the player complete the objective before running out of steps?")


def games_dir() -> Path:
    return Path(os.environ.get("TEXTWORLD_GAMES", Path.home() / ".cache" / "jev-benchmarking" / "textworld"))


def game_file(setting: str, seed: int) -> str:
    """Generate the game once and return its path.

    Cooking games are made with the `tw-make` command at its defaults. Coin collector and treasure hunter games
    are made exactly as `tw-make` makes them (the challenge's own `make` function, then `compile_game`), with
    one documented generator setting changed: `options.grammar.only_last_action = True`, so that the objective
    states the goal ("pick up the coin from the floor of the ...") instead of every step of the walkthrough.
    """
    args = list(SETTINGS[setting])
    goal_only = args[0] != "tw-cooking" and GOAL_ONLY_OBJECTIVE
    path = games_dir() / setting / f"seed{seed:02d}{'.goal' if goal_only else ''}.z8"
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if args[0] == "tw-cooking":
        args += ["--recipe-seed", str(seed)]  # a different recipe per game, as in the FTWP games
        subprocess.run(["tw-make", *args, "--seed", str(seed), "--output", str(path), "--silent", "-f"], check=True)
        return str(path)
    import textworld
    import textworld.challenges
    from textworld.generator import compile_game

    from jev_benchmarking.sequential.envs import alfworld  # noqa: F401  (TextWorld fix for Python 3.13)

    options = textworld.GameOptions()  # as in tw-make
    options.seeds = seed
    options.path = str(path)
    options.file_ext = ".z8"
    options.force_recompile = True
    options.grammar.only_last_action = GOAL_ONLY_OBJECTIVE
    _, make_game, _ = textworld.challenges.CHALLENGES[args[0]]
    compile_game(make_game(settings={"level": int(args[2])}, options=options), options)
    return str(path)


class TextWorldGames(SequentialEnv):
    name = "textworld"
    description = "TextWorld, official generators (coin collector, treasure hunter, cooking), 3 levels x 10 games; solver reference."

    def __init__(self, ask_success: bool = False):
        from jev_benchmarking.sequential.envs import alfworld  # noqa: F401  (TextWorld fix for Python 3.13)

        self.ask_success = ask_success
        self.max_steps = STEP_BUDGET
        self.env = None

    def episodes(self, split: str = "eval", limit: int | None = None) -> list[EpisodeSpec]:
        """All 90 games, interleaved across the nine settings so that any prefix covers all of them."""
        out = [EpisodeSpec(f"{s}/seed{seed:02d}", s.split("/")[0], {"setting": s, "seed": seed})
               for seed in SEEDS for s in SETTINGS]
        return out[:limit] if limit is not None else out

    def reset(self, spec: EpisodeSpec) -> None:
        import textworld

        if self.env is not None:
            self.env.close()
        infos = textworld.EnvInfos(objective=True, description=True, inventory=True, admissible_commands=True,
                                   policy_commands=True, won=True, lost=True, score=True, max_score=True)
        self.env = textworld.start(game_file(spec.payload["setting"], spec.payload["seed"]), request_infos=infos)
        self.gs = self.env.reset()
        self.first_observation = self._strip(self.gs.feedback)
        self.history: list[dict] = []
        self.t = 0
        self.result: StepResult | None = None

    @staticmethod
    def _strip(text: str) -> str:
        """Game text without the ASCII-art title banner and the trailing prompt and status line."""
        lines = (text or "").split("\n")
        banner = [k for k, ln in enumerate(lines) if "$$" in ln]
        if banner:
            lines = lines[banner[-1] + 1:]
        for k, ln in enumerate(lines):
            if ln.startswith(">"):
                lines = lines[:k]
                break
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

    def state(self) -> dict:
        return {
            "objective": (self.gs.objective or "").strip(),
            "first_observation": self.first_observation,
            "current_room": (self.gs.description or "").strip(),
            "inventory": (self.gs.inventory or "").strip(),
            "history": list(self.history),
            "steps_taken": self.t,
            "steps_left": self.max_steps - self.t,
        }

    def decision(self) -> Decision | None:
        if self.result is not None or self.gs.done:
            return None
        commands = list(self.gs.admissible_commands)
        policy = list(self.gs.policy_commands or [])
        oracle = {"expert": policy[0]} if policy and policy[0] in commands else {}
        q = choice(ACTION_Q_TEXT, mc_criteria(commands))
        questions = {ACTION: q, SUCCESS: SUCCESS_Q} if self.ask_success else {ACTION: q}
        return Decision(self.state(), questions, dict(zip(option_keys(len(commands)), commands)), oracle)

    def step(self, action: str) -> StepResult:
        self.gs, _, done = self.env.step(action)
        self.t += 1
        feedback = self._strip(self.gs.feedback)
        self.history.append({"command": action, "feedback": feedback})
        if done or self.gs.won or self.gs.lost:
            won = bool(self.gs.won)
            progress = 1.0 if won else (self.gs.score / self.gs.max_score if self.gs.max_score else 0.0)
            self.result = StepResult(True, won, float(progress), feedback)
            return self.result
        return StepResult(False, feedback=feedback)

    def outcome(self) -> StepResult:
        if self.result is not None:
            return self.result
        progress = self.gs.score / self.gs.max_score if self.gs.max_score else 0.0
        return StepResult(False, False, float(progress), "No steps left.")

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def random_policy(state, questions):
    from jev_benchmarking.sequential.backends import choice_answer

    keys = list(questions[ACTION]["criteria"])
    return _only_asked(questions, {ACTION: choice_answer({k: 1.0 for k in keys})})


SCRIPTED = {"random": random_policy}
