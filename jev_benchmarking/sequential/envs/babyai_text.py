"""BabyAI-Text with the original code of Carta et al. (ICML 2023, "Grounding Large Language Models in
Interactive Environments with Online Reinforcement Learning"; GLAM repository, MIT license).

Everything comes from the original packages at a pinned commit of the GLAM repository
(`scripts/setup_babyai_text_env.sh`): the level `BabyAI-MixedTrainLocal-v0` with its default settings
(one 8x8 room, eight distractor objects; a second, locked room for the open task), the text descriptions
of the agent's view (`info["descriptions"]`), the six actions of their experiments (turn left, turn right, go
forward, pick up, drop, toggle), their prompt (`generate_prompt` of experiments/agents/base_agent.py, with
the last three observations and two actions, as in their configuration `nbr_obs: 3`), and the BabyAI bot
(`babyai.bot.Bot`) as reference, replanned at every step after the action actually taken.

The level draws its instruction type (go to, pick up, open, put next to, pick up then go to) when the
environment is constructed, from an unseeded generator. To obtain a reproducible set of episodes with all
five types, each episode constructs environments until one has the wanted type, then seeds it; the room,
objects and mission then depend only on the seed.

The original packages need an old gym and numpy, so this task runs in its own environment.
"""

from collections import deque

from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS, Decision, EpisodeSpec, SequentialEnv, StepResult, _only_asked
from jev_benchmarking.tasks.base import choice, noul

GLAM_COMMIT = "31847fb8c1db58bc8f595bea8f8cbaf26fb79dc2"
LEVEL = "BabyAI-MixedTrainLocal-v0"
ACTIONS = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]  # BabyAI action ids 0..5
KINDS = ["goto", "pickup", "open", "putnext", "pick up seq go to"]
NBR_OBS = 3

ACTION_Q_TEXT = "Which action should the agent take next?"
SUCCESS_Q = noul("Will the agent reach its goal before running out of steps?")


def generate_prompt(goal, subgoals, deque_obs, deque_actions):
    """Verbatim copy of BaseAgent.generate_prompt (GLAM, experiments/agents/base_agent.py)."""
    ldo = len(deque_obs)
    lda = len(deque_actions)

    head_prompt = "Possible action of the agent:"
    for sg in subgoals:
        head_prompt += " {},".format(sg)
    head_prompt = head_prompt[:-1]

    g = " \n Goal of the agent: {}".format(goal)
    obs = ""
    for i in range(ldo):
        obs += " \n Observation {}: ".format(i)
        for d_obs in deque_obs[i]:
            obs += "{}, ".format(d_obs)
        obs += "\n Action {}: ".format(i)
        if i < lda:
            obs += "{}".format(deque_actions[i])
    return head_prompt + g + obs


class BabyAIText(SequentialEnv):
    name = "babyai_text"
    description = "BabyAI-Text, original code of Carta et al. (MixedTrainLocal, 8 distractors, their prompt); BabyAI bot reference."
    DEFAULT_N = {"eval": 250, "dev": 50}
    SEED_OFFSET = {"eval": 0, "dev": 100_000}

    def __init__(self, ask_success: bool = False):
        self.ask_success = ask_success
        self.env = None
        self.max_steps = 128

    def episodes(self, split: str = "eval", limit: int | None = None) -> list[EpisodeSpec]:
        """The five instruction types in turn (any prefix covers all five), one seed per episode."""
        n = limit if limit is not None else self.DEFAULT_N[split]
        return [EpisodeSpec(f"{split}-{KINDS[i % 5].replace(' ', '_')}-seed{self.SEED_OFFSET[split] + i:06d}",
                            KINDS[i % 5].replace(" ", "_"), {"kind": KINDS[i % 5], "seed": self.SEED_OFFSET[split] + i})
                for i in range(n)]

    def reset(self, spec: EpisodeSpec) -> None:
        import gym
        import babyai_text  # noqa: F401  (registers the levels)
        from babyai.bot import Bot

        if self.env is not None:
            self.env.close()
        while True:  # the instruction type is fixed at construction (see module docstring)
            env = gym.make(LEVEL)
            if env.unwrapped.action_kinds[0] == spec.payload["kind"]:
                break
            env.close()
        self.env = env
        self.env.seed(spec.payload["seed"])
        self.obs, self.infos = self.env.reset()
        self.max_steps = self.env.unwrapped.max_steps
        self.bot = Bot(self.env.unwrapped)
        self.last_action = None
        self.obs_queue = deque([], maxlen=NBR_OBS)
        self.acts_queue = deque([], maxlen=NBR_OBS - 1)
        self.obs_queue.append(self.infos["descriptions"])
        self.t = 0
        self.result: StepResult | None = None

    def state(self) -> dict:
        """The prompt of Carta et al., exactly as their agents receive it."""
        return {"prompt": generate_prompt(self.obs["mission"], ACTIONS, self.obs_queue, self.acts_queue)}

    def _reference(self) -> str | None:
        try:
            a = int(self.bot.replan(self.last_action))
        except Exception:  # the bot cannot plan from some states it was not built for
            return None
        return ACTIONS[a] if a < len(ACTIONS) else None  # 'done' is not one of the six actions

    def decision(self) -> Decision | None:
        if self.result is not None:
            return None
        ref = self._reference()
        q = choice(ACTION_Q_TEXT, {a: a for a in ACTIONS})
        questions = {ACTION: q, SUCCESS: SUCCESS_Q} if self.ask_success else {ACTION: q}
        return Decision(self.state(), questions, {a: a for a in ACTIONS}, {"expert": ref} if ref else {})

    def step(self, action: str) -> StepResult:
        a = ACTIONS.index(action)
        self.acts_queue.append(action)
        self.obs, reward, done, self.infos = self.env.step(a)
        self.last_action = a
        self.t += 1
        self.obs_queue.append(self.infos["descriptions"])
        feedback = ", ".join(self.infos["descriptions"])
        if done:
            self.result = StepResult(True, reward > 0, 1.0 if reward > 0 else 0.0, feedback)
            return self.result
        return StepResult(False, feedback=feedback)

    def outcome(self) -> StepResult:
        return self.result or StepResult(False, feedback="No steps left.")

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def random_policy(state, questions):
    from jev_benchmarking.sequential.backends import choice_answer

    return _only_asked(questions, {ACTION: choice_answer({a: 1.0 for a in ACTIONS})})


SCRIPTED = {"random": random_policy}
