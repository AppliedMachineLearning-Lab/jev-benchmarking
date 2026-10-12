"""BabyAI-Text as in BALROG, on current minigrid, with the BabyAI bot as reference.

BALROG (Paglieri et al., 2025) evaluates LLM agents on BabyAI-Text (Carta et al., 2023): the level
`BabyAI-MixedTrainLocal-v0` restricted to one of five instruction types (go to, pick up, open, put next to,
pick up then go to), with no distractor objects (`num_dists=0`), the six actions turn left, turn right, go
forward, pick up, drop, toggle, and a text description of what the agent sees in its 7x7 egocentric view
("a red ball 1 step left and 2 steps forward"). The original code depends on a fork of gym-minigrid for old
gym versions; this module ports the level generator and the English description function to current
minigrid (`minigrid.envs.babyai`) without changing their logic. Missions that BabyAI-Text holds out for
testing (yellow box, red key, red door, green ball, grey door) are excluded, as in the training level BALROG
uses.

The reference action is that of minigrid's BabyAI bot, replanned at every step from the current state after
the action actually taken. The bot knows the positions of all objects, including ones the agent has not
seen yet, so on search steps its action is the best action under full information.
"""

import contextlib
import hashlib
import io

import gymnasium as gym
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.envs.babyai.core.levelgen import LevelGen
from minigrid.envs.babyai.core.roomgrid_level import RejectSampling
from minigrid.envs.babyai.core.verifier import AfterInstr, BeforeInstr, GoToInstr, ObjDesc, OpenInstr, PickupInstr, PutNextInstr
from minigrid.utils.baby_ai_bot import BabyAIBot

from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS, Decision, EpisodeSpec, SequentialEnv, StepResult, _only_asked
from jev_benchmarking.tasks.base import choice, noul

ACTIONS = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]  # minigrid action ids 0..5
ACTION_HELP = {  # BALROG's descriptions, except that pick up takes the object in front (not "below you")
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object in front of you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object in front of you",
}
TASKS = ["goto", "pickup", "open", "putnext", "pick up seq go to"]
EXCLUDED = ["yellow box", "red key", "red door", "green ball", "grey door"]  # held out by BabyAI-Text

RULES = (
    "BabyAI. The agent is in a grid world and must follow the mission. It sees a 7 by 7 area in front of it: "
    "positions are given in steps forward and steps to the left or right of the agent, as it is facing now. "
    "Turning changes which way the agent faces without moving it. 'go forward' moves one step in the facing "
    "direction unless a wall, a closed door or an object blocks the way. 'pick up' takes the object directly "
    "in front of the agent, which can carry one object at a time; 'drop' puts it down in front of the agent; "
    "'toggle' opens or closes the door in front of the agent (a locked door needs the key of its colour). "
    "The episode ends when the mission is completed or when no steps are left."
)

ACTION_Q = choice(
    "Which action brings the agent closest to completing the mission?",
    dict(ACTION_HELP),
)
SUCCESS_Q = noul("Will the agent complete the mission before running out of steps?")


class MixedLocal(LevelGen):
    """Port of BabyAI-Text's `Level_MixedTrainLocal` with the instruction type fixed per episode."""

    def __init__(self, action: str, num_dists: int = 0, room_size: int = 8, **kwargs):
        self.action = action
        cols = 2 if action == "open" else 1
        super().__init__(room_size=room_size, num_rows=1, num_cols=cols, num_dists=num_dists,
                         action_kinds=[action], instr_kinds=["action", "seq1"], locations=False,
                         unblocking=False, implicit_unlock=False, **kwargs)

    def _excluded(self) -> bool:
        surface = self.instrs.surface(self)
        return any(s in surface for s in EXCLUDED)

    def _pick(self, objs, not_door=True):
        obj = self._rand_elem(objs)
        while not_door and obj.type == "door":
            obj = self._rand_elem(objs)
        return obj

    def _other(self, objs, first):
        """A second object; it may share the first one's description only if that description is ambiguous."""
        second = self._rand_elem(objs)
        if (first.type, first.color) == (second.type, second.color):
            matches, _ = ObjDesc(first.type, first.color).find_matching_objs(self)
            if len(matches) < 2:
                while (first.type, first.color) == (second.type, second.color):
                    second = self._rand_elem(objs)
        return second

    def gen_mission(self):
        if self.action == "open":
            color = self._rand_elem(["yellow", "green", "blue", "purple"])  # red and grey excluded
            self._add_locked_room(color)
            self.connect_all()
            for j in range(self.num_rows):
                for i in range(self.num_cols):
                    if self.get_room(i, j) is not self.locked_room:
                        self.add_distractors(i, j, num_distractors=self.num_dists, all_unique=False)
            while True:
                self.place_agent()
                if self.room_from_pos(*self.agent_pos) is not self.locked_room:
                    break
            if not self.check_objs_reachable(raise_exc=False):
                raise RejectSampling("unreachable objects")
            self.instrs = OpenInstr(ObjDesc("door", self._rand_elem([None, color])))
        else:
            self.place_agent()
            extra = {"goto": 1, "pickup": 1, "putnext": 2, "pick up seq go to": 2}[self.action]
            objs = self.add_distractors(num_distractors=self.num_dists + extra, all_unique=False)
            if not self.check_objs_reachable(raise_exc=False):
                raise RejectSampling("unreachable objects")
            if self.action == "goto":
                obj = self._rand_elem(objs)
                self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))
            elif self.action == "pickup":
                obj = self._pick(objs)
                self.instrs = PickupInstr(ObjDesc(obj.type, obj.color))
            elif self.action == "putnext":
                a = self._pick(objs)
                b = self._other(objs, a)
                self.instrs = PutNextInstr(ObjDesc(a.type, a.color), ObjDesc(b.type, b.color))
            else:
                a = self._pick(objs)
                b = self._other(objs, a)
                pick, go = PickupInstr(ObjDesc(a.type, a.color)), GoToInstr(ObjDesc(b.type, b.color))
                self.instrs = BeforeInstr(pick, go) if self._rand_elem(["Before", "After"]) == "Before" else AfterInstr(go, pick)
        if self._excluded():
            raise RejectSampling("held-out mission")

    def _add_locked_room(self, color):
        while True:
            i, j = self._rand_int(0, self.num_cols), self._rand_int(0, self.num_rows)
            door_idx = self._rand_int(0, 4)
            self.locked_room = self.get_room(i, j)
            if self.locked_room.neighbors[door_idx] is None:
                continue
            door, _ = self.add_door(i, j, door_idx, color=color, locked=True)
            break
        while True:
            i, j = self._rand_int(0, self.num_cols), self._rand_int(0, self.num_rows)
            if self.get_room(i, j) is self.locked_room:
                continue
            self.add_object(i, j, "key", door.color)
            break


IDX_TO_STATE = {0: "open", 1: "closed", 2: "locked"}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}


def describe(env) -> list[str]:
    """BabyAI-Text's English description of the agent's view (port of its `gen_obs` descriptions)."""
    env = env.unwrapped
    grid, vis_mask = env.gen_obs_grid()
    image = grid.encode(vis_mask)
    out = []
    if env.carrying is not None:
        out.append(f"You carry a {env.carrying.color} {env.carrying.type}")
    ax, ay = env.agent_view_size // 2, env.agent_view_size - 1  # the agent's position in its own view

    def wall(axis_cells, label):
        for k, (i, j) in enumerate(axis_cells, start=1):
            code = image[i][j][0]
            if code in (0, 1):
                continue
            if code == 2:
                out.append(f"You see a wall {k} step{'s' if k > 1 else ''} {label}")
            return

    wall([(ax, j) for j in range(ay - 1, -1, -1)], "forward")
    wall([(i, ay) for i in range(ax - 1, -1, -1)], "left")
    wall([(i, ay) for i in range(ax + 1, image.shape[0])], "right")

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            obj = image[i][j]
            if obj[0] in (0, 1, 2) or (i, j) == (ax, ay):
                continue
            dx, dy = i - ax, ay - j
            parts = []
            if dx == 0:
                parts.append((dy, "forward"))
            elif dy == 0:
                parts.append((abs(dx), "right" if dx > 0 else "left"))
            else:
                parts.append((abs(dx), "right" if dx > 0 else "left"))
                parts.append((dy, "forward"))
            if obj[0] == 4:
                state = IDX_TO_STATE[obj[2]]
                what = f"You see {'an' if state == 'open' else 'a'} {state} {IDX_TO_COLOR[obj[1]]} door "
            else:
                what = f"You see a {IDX_TO_COLOR[obj[1]]} {IDX_TO_OBJECT[obj[0]]} "
            out.append(what + " and ".join(f"{n} step{'s' if n > 1 else ''} {d}" for n, d in parts))
    return out


def _seed(uid: str) -> int:
    return int(hashlib.sha256(f"babyai/{uid}".encode()).hexdigest(), 16) % (2**31)


class BabyAIText(SequentialEnv):
    name = "babyai_text"
    description = "BabyAI-Text (BALROG setup: MixedTrainLocal, 5 instruction types, no distractors); BabyAI bot reference."
    SEED_OFFSET = {"eval": 0, "dev": 10_000_000}
    DEFAULT_N = {"eval": 250, "dev": 50}
    history_window = None  # full history

    def __init__(self, ask_success: bool = False, num_dists: int = 0):
        self.ask_success = ask_success
        self.num_dists = num_dists
        self.max_steps = 128
        self.env = None

    def episodes(self, split: str = "eval", limit: int | None = None) -> list[EpisodeSpec]:
        """Cycles through the five instruction types, so any prefix (`--limit`) covers all of them."""
        n = limit if limit is not None else self.DEFAULT_N[split]
        base = self.SEED_OFFSET[split]
        return [EpisodeSpec(f"{split}-{TASKS[i % 5].replace(' ', '_')}-{i:05d}", TASKS[i % 5].replace(" ", "_"),
                            {"task": TASKS[i % 5], "seed": base + i}) for i in range(n)]

    def reset(self, spec: EpisodeSpec) -> None:
        if self.env is not None:
            self.env.close()
        self.env = MixedLocal(spec.payload["task"], num_dists=self.num_dists)
        with contextlib.redirect_stdout(io.StringIO()):  # minigrid prints every rejected level sample
            self.env.reset(seed=spec.payload["seed"])
        self.max_steps = self.env.max_steps
        self.mission = self.env.mission
        self.bot = BabyAIBot(self.env)
        self.last_action: int | None = None
        self.view = describe(self.env)
        self.history: list[dict] = []
        self.t = 0
        self.result: StepResult | None = None

    def _reference(self) -> str | None:
        try:
            a = int(self.bot.replan(self.last_action))
        except Exception:  # the bot gives up in rare states it cannot plan from
            return None
        return ACTIONS[a] if a < len(ACTIONS) else None  # 'done' (6) is not an option

    def state(self) -> dict:
        hist = self.history if self.history_window is None else self.history[-self.history_window:]
        return {
            "rules": RULES,
            "mission": self.mission,
            "current_view": [d.replace("You see ", "") for d in self.view] or ["nothing"],
            "history": list(hist),
            "steps_taken": self.t,
            "steps_left": self.max_steps - self.t,
        }

    def decision(self) -> Decision | None:
        if self.result is not None:
            return None
        ref = self._reference()
        oracle = {"expert": ref} if ref else {}
        questions = {ACTION: ACTION_Q, SUCCESS: SUCCESS_Q} if self.ask_success else {ACTION: ACTION_Q}
        return Decision(self.state(), questions, {a: a for a in ACTIONS}, oracle)

    def step(self, action: str) -> StepResult:
        a = ACTIONS.index(action)
        _, reward, terminated, truncated, _ = self.env.step(a)
        self.last_action = a
        self.t += 1
        self.view = describe(self.env)
        seen = [d.replace("You see ", "") for d in self.view] or ["nothing"]
        self.history.append({"action": action, "view_after": seen})
        feedback = "; ".join(seen)
        if terminated or truncated:
            success = reward > 0
            self.result = StepResult(True, success, 1.0 if success else 0.0,
                                     feedback + (" Mission completed." if success else " No steps left." if truncated else ""))
            return self.result
        return StepResult(False, feedback=feedback)

    def outcome(self) -> StepResult:
        return self.result or StepResult(False)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


# ----------------------------------------------------------------------------------------------------
# Scripted policies (FunctionBackend); the bot is played through the CLI's expert backend, which reads the
# reference action from the oracle.


def random_policy(state, questions):
    from jev_benchmarking.sequential.backends import choice_answer

    return _only_asked(questions, {ACTION: choice_answer({a: 1 / 6 for a in ACTIONS})})


SCRIPTED = {"random": random_policy}
