"""FrozenLake on Gymnasium's implementation, with exact action values computed alongside.

The game is Gymnasium's `FrozenLake-v1` (Towers et al., 2024) with slippery ice, the standard setting: the
agent walks on a grid from the start S to the goal G over frozen tiles F and must avoid holes H. Each move
goes in the intended direction with probability 1/3 and in each of the two perpendicular directions with
probability 1/3; a move into the edge leaves the agent in place. The episode ends in a hole (failure), at
the goal (success), or at the step limit Gymnasium registers for the map size (100 steps for 4x4, 200
for 8x8). Each episode uses its own map from Gymnasium's `generate_random_map` (frozen-tile probability
0.8, a path from S to G guaranteed), seeded by the episode id, so the model cannot rely on a memorised map.

Gymnasium simulates every step; this module only renders the state as text and computes the reference.
Because the transition probabilities are known, the probability of reaching the goal within the remaining
steps can be computed exactly for every state and action by finite-horizon dynamic programming on
Gymnasium's own transition table. The optimal actions are those with the highest probability; several
actions are often tied (for example two moves that both slide along a wall). These values are logged with
every step (`oracle`) and never sent to the model.
"""

from functools import lru_cache

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS, Decision, EpisodeSpec, SequentialEnv, StepResult, _only_asked
from jev_benchmarking.tasks.base import choice, noul

MOVES = ["left", "down", "right", "up"]  # Gymnasium's action order 0..3
STEP_LIMIT = {4: 100, 8: 200}  # Gymnasium's registered limits for FrozenLake-v1 and FrozenLake8x8-v1
TIE = 1e-9

RULES = (
    "FrozenLake. The agent walks on a frozen lake drawn as a grid of tiles: S is the start, F is frozen "
    "ice that is safe to stand on, H is a hole and G is the goal. Rows are numbered from 0 at the top and "
    "columns from 0 at the left. 'up' decreases the row, 'down' increases it, 'left' decreases the column "
    "and 'right' increases it. The ice is slippery: the agent moves in the chosen direction with "
    "probability 1/3 and in each of the two perpendicular directions with probability 1/3 (for example, "
    "choosing 'up' moves up, left or right, each with probability 1/3). A move into the edge of the grid "
    "leaves the agent where it is. Stepping onto H ends the episode in failure, reaching G ends it in "
    "success, and the episode also fails when no steps are left. In the map with the agent, A marks the "
    "agent's tile."
)

ACTION_Q = choice(
    "Which move gives the highest probability of reaching the goal G before falling into a hole or running out of steps?",
    {"left": "move left", "down": "move down", "right": "move right", "up": "move up"},
)
SUCCESS_Q = noul("Will the agent reach the goal G before falling into a hole or running out of steps?")


@lru_cache(maxsize=4096)
def _model(desc: tuple[str, ...]):
    """Gymnasium's transition table for a map: P[s][a] = [(prob, next_state, reward, terminated), ...]."""
    env = gym.make("FrozenLake-v1", desc=list(desc), is_slippery=True)
    P = env.unwrapped.P
    env.close()
    return P


@lru_cache(maxsize=4096)
def q_table(desc: tuple[str, ...], horizon: int) -> np.ndarray:
    """Q[k, s, a]: probability of reaching G when taking `a` in `s` with k steps left (k = 1..horizon),
    then acting optimally. Terminal tiles never get a decision."""
    P = _model(desc)
    n = len(desc) * len(desc[0])
    flat = "".join(desc)
    v = np.zeros(n)  # V with 0 steps left: success only if already at G (G is terminal, so never queried)
    Q = np.zeros((horizon + 1, n, 4))
    for k in range(1, horizon + 1):
        for s in range(n):
            if flat[s] in "GH":
                continue
            for a in range(4):
                Q[k, s, a] = sum(p * (1.0 if flat[s2] == "G" else 0.0 if flat[s2] == "H" else v[s2]) for p, s2, _, _ in P[s][a])
        v = Q[k].max(axis=1)
        v[[i for i, t in enumerate(flat) if t == "G"]] = 1.0
        v[[i for i, t in enumerate(flat) if t == "H"]] = 0.0
    return Q


def state_values(desc: tuple[str, ...], row: int, col: int, steps_left: int, horizon: int) -> dict:
    q = q_table(desc, horizon)[steps_left, row * len(desc[0]) + col]
    best = float(q.max())
    optimal = [MOVES[a] for a in range(4) if q[a] >= best - TIE]
    return {"q": {MOVES[a]: float(q[a]) for a in range(4)}, "p_win": best, "optimal": optimal, "best": optimal[0]}

# ----------------------------------------------------------------------------------------------------
# Prompt variants, used only by the diagnostic probe (scripts/probe_frozenlake.py); episodes use "base".
#   base:          the request above.
#   safety:        the question adds the strategy (avoid moves that can slip into a hole).
#   outcomes:      each option lists where the move can land, so no consequence has to be worked out.
#   deterministic: non-slippery ice (moves go where chosen); the reference is the shortest safe path.

VARIANTS = ("base", "safety", "outcomes", "deterministic")

SAFETY_Q = choice(
    "Which move gives the highest probability of reaching the goal G before falling into a hole or running out "
    "of steps? Falling into a hole ends the game, and every move can slip sideways. Prefer a move for which no "
    "possible slip lands on a hole, even if it leads away from the goal for now.",
    {"left": "move left", "down": "move down", "right": "move right", "up": "move up"},
)

RULES_DETERMINISTIC = (
    "FrozenLake. The agent walks on a frozen lake drawn as a grid of tiles: S is the start, F is frozen "
    "ice that is safe to stand on, H is a hole and G is the goal. Rows are numbered from 0 at the top and "
    "columns from 0 at the left. 'up' decreases the row, 'down' increases it, 'left' decreases the column "
    "and 'right' increases it. The ice is not slippery: the agent always moves in the chosen direction. A "
    "move into the edge of the grid leaves the agent where it is. Stepping onto H ends the episode in "
    "failure and reaching G ends it in success. In the map with the agent, A marks the agent's tile."
)

DETERMINISTIC_Q = choice(
    "Which move brings the agent to the goal G in the fewest moves without ever stepping onto a hole?",
    {"left": "move left", "down": "move down", "right": "move right", "up": "move up"},
)

DELTA = {"left": (0, -1), "down": (1, 0), "right": (0, 1), "up": (-1, 0)}
PERPENDICULAR = {"left": ("down", "up"), "down": ("left", "right"), "right": ("down", "up"), "up": ("left", "right")}


def landing(desc: tuple[str, ...], row: int, col: int, direction: str) -> tuple[int, int]:
    dr, dc = DELTA[direction]
    r, c = row + dr, col + dc
    return (r, c) if 0 <= r < len(desc) and 0 <= c < len(desc[0]) else (row, col)


def outcome_text(desc: tuple[str, ...], row: int, col: int, move: str) -> str:
    """Where a slippery move can land, merged by tile, e.g. '2/3: stay at row 0, column 0 (S); 1/3: row 0,
    column 1 (F)'."""
    counts: dict[tuple[int, int], int] = {}
    for d in (move, *PERPENDICULAR[move]):
        counts[landing(desc, row, col, d)] = counts.get(landing(desc, row, col, d), 0) + 1
    names = {"S": "start", "F": "frozen", "H": "HOLE", "G": "GOAL"}
    parts = []
    for (r, c), k in counts.items():
        where = f"stay at row {r}, column {c}" if (r, c) == (row, col) else f"row {r}, column {c}"
        parts.append(f"{k}/3: {where} ({names[desc[r][c]]})")
    return f"move {move}; lands on " + "; ".join(parts)


@lru_cache(maxsize=4096)
def distances(desc: tuple[str, ...]) -> dict[tuple[int, int], int]:
    """Shortest number of deterministic moves to G from every tile, never stepping onto H (BFS from G)."""
    n, m = len(desc), len(desc[0])
    goal = next((r, c) for r in range(n) for c in range(m) if desc[r][c] == "G")
    dist, frontier = {goal: 0}, [goal]
    while frontier:
        nxt = []
        for r, c in frontier:
            for dr, dc in DELTA.values():
                pr, pc = r - dr, c - dc  # a tile from which moving (dr, dc) reaches (r, c)
                if 0 <= pr < n and 0 <= pc < m and (pr, pc) not in dist and desc[pr][pc] != "H":
                    dist[(pr, pc)] = dist[(r, c)] + 1
                    nxt.append((pr, pc))
        frontier = nxt
    return dist


def deterministic_values(desc: tuple[str, ...], row: int, col: int) -> dict:
    """Reference for non-slippery ice: the moves that start a shortest safe path to G."""
    dist = distances(desc)
    q = {}
    for mv in MOVES:
        r, c = landing(desc, row, col, mv)
        # moves needed to reach G after this move (negated); a hole is worst, then a tile without a safe path, then
        # staying in place against the edge
        if desc[r][c] == "H":
            q[mv] = -1000.0
        elif (r, c) == (row, col):
            q[mv] = -99.0
        elif (r, c) not in dist:
            q[mv] = -500.0
        else:
            q[mv] = -float(dist[(r, c)] + 1)
    best = max(q.values())
    optimal = [mv for mv in MOVES if q[mv] >= best - TIE] if (row, col) in dist else []
    return {"q_steps": q, "optimal": optimal, "best": optimal[0] if optimal else None,
            "distance": dist.get((row, col))}


def build_request(desc: tuple[str, ...], row: int, col: int, steps_taken: int, max_steps: int,
                  variant: str = "base", ask_success: bool = False) -> tuple[dict, dict, dict]:
    """(state, questions, oracle) for one decision under a prompt variant."""
    rows = list(desc)
    rows[row] = rows[row][:col] + "A" + rows[row][col + 1:]
    state = {
        "rules": RULES_DETERMINISTIC if variant == "deterministic" else RULES,
        "map": list(desc),
        "map_with_agent": rows,
        "agent_row": row,
        "agent_column": col,
        "agent_tile": desc[row][col],
        "steps_taken": steps_taken,
        "steps_left": max_steps - steps_taken,
    }
    if variant == "deterministic":
        q, oracle = DETERMINISTIC_Q, deterministic_values(desc, row, col)
    else:
        q = {"base": ACTION_Q, "safety": SAFETY_Q}.get(variant)
        if variant == "outcomes":
            q = choice(ACTION_Q["instructions"], {mv: outcome_text(desc, row, col, mv) for mv in MOVES})
        if q is None:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        oracle = state_values(desc, row, col, max_steps - steps_taken, max_steps)
    questions = {ACTION: q, SUCCESS: SUCCESS_Q} if ask_success else {ACTION: q}
    return state, questions, oracle


class FrozenLake(SequentialEnv):
    name = "frozenlake"
    description = "Gymnasium FrozenLake-v1 (slippery), random 4x4 and 8x8 maps; exact success probabilities by DP."
    SIZES = (4, 8)
    SEED_OFFSET = {"eval": 0, "dev": 10_000_000}
    DEFAULT_N = {"eval": 500, "dev": 100}

    def __init__(self, ask_success: bool = False):
        self.ask_success = ask_success
        self.max_steps = max(STEP_LIMIT.values())
        self.env = None

    def episodes(self, split: str = "eval", limit: int | None = None) -> list[EpisodeSpec]:
        """Alternates 4x4 and 8x8, so any prefix (`--limit`) has both sizes."""
        n = limit if limit is not None else self.DEFAULT_N[split]
        base = self.SEED_OFFSET[split]
        out = []
        for i in range(n):
            size = self.SIZES[i % len(self.SIZES)]
            out.append(EpisodeSpec(f"{split}-{size}x{size}-{i:05d}", f"{size}x{size}", {"size": size, "seed": base + i}))
        return out

    def reset(self, spec: EpisodeSpec) -> None:
        size, seed = spec.payload["size"], spec.payload["seed"]
        self.desc = tuple(generate_random_map(size=size, p=0.8, seed=seed))
        self.max_steps = STEP_LIMIT[size]
        if self.env is not None:
            self.env.close()
        self.env = gym.make("FrozenLake-v1", desc=list(self.desc), is_slippery=True, max_episode_steps=self.max_steps)
        self.s, _ = self.env.reset(seed=seed)
        self.t = 0
        self.result: StepResult | None = None

    @property
    def pos(self) -> tuple[int, int]:
        return divmod(int(self.s), len(self.desc[0]))

    def state(self) -> dict:
        r, c = self.pos
        return build_request(self.desc, r, c, self.t, self.max_steps)[0]

    def decision(self) -> Decision | None:
        if self.result is not None:
            return None
        r, c = self.pos
        state, questions, oracle = build_request(self.desc, r, c, self.t, self.max_steps, "base", self.ask_success)
        oracle = dict(oracle, row=r, col=c, steps_left=self.max_steps - self.t)
        return Decision(state, questions, {m: m for m in MOVES}, oracle)

    def step(self, action: str) -> StepResult:
        self.s, reward, terminated, truncated, _ = self.env.step(MOVES.index(action))
        self.t += 1
        r, c = self.pos
        tile = self.desc[r][c]
        where = f"Chose {action}; now at row {r}, column {c} ({tile})."
        if terminated or truncated:
            success = tile == "G"
            end = "Reached the goal." if success else "Fell into a hole." if tile == "H" else "No steps left."
            self.result = StepResult(True, success, float(reward), f"{where} {end}")
            return self.result
        return StepResult(False, feedback=where)

    def outcome(self) -> StepResult:
        return self.result or StepResult(False)

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


# ----------------------------------------------------------------------------------------------------
# Scripted reference policies (FunctionBackend).


def _oracle_of(state: dict) -> dict:
    desc = tuple(state["map"])
    horizon = STEP_LIMIT[len(desc)]
    return state_values(desc, state["agent_row"], state["agent_column"], state["steps_left"], horizon)


def optimal_policy(state: dict, questions: dict) -> dict:
    """Always the first optimal move (deterministic), with the exact success probability: the reference."""
    from jev_benchmarking.sequential.backends import choice_answer, noul_answer

    v = _oracle_of(state)
    return _only_asked(questions, {ACTION: choice_answer({m: float(m == v["best"]) for m in MOVES}),
                                   SUCCESS: noul_answer(v["p_win"])})


def random_policy(state: dict, questions: dict) -> dict:
    from jev_benchmarking.sequential.backends import choice_answer, noul_answer

    return _only_asked(questions, {ACTION: choice_answer({m: 0.25 for m in MOVES}), SUCCESS: noul_answer(0.5)})


SCRIPTED = {"optimal": optimal_policy, "random": random_policy}
