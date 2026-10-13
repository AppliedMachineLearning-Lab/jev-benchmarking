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

Diagnostic variant (`babyai_text_instructions`, not part of the main results): the same episodes with the
instruction prompt that BALROG (Paglieri et al., 2025) gives zero-shot LLMs on this level added to the
state, copied verbatim (`balrog/environments/babyai_text/__init__.py`, commit b7afe79): the goal, a short
description of each action, and two tips, one of which says not to repeat an action whose observation does
not change. The prompt of Carta et al. was written for agents trained with reinforcement learning and
explains neither the actions nor the rules; the variant tests whether Jev's results depend on that.
"""

from collections import deque

from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS, Decision, EpisodeSpec, SequentialEnv, StepResult, _only_asked
from jev_benchmarking.tasks.base import choice, noul

GLAM_COMMIT = "31847fb8c1db58bc8f595bea8f8cbaf26fb79dc2"
LEVEL = "BabyAI-MixedTrainLocal-v0"
ACTIONS = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]  # BabyAI action ids 0..5
KINDS = ["goto", "pickup", "open", "putnext", "pick up seq go to"]
NBR_OBS = 3
BALROG_COMMIT = "b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5"
BALROG_ACTIONS = {  # verbatim from BALROG
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object below you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object in front of you",
}

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


def balrog_instruction_prompt(mission):
    """Verbatim copy of get_instruction_prompt (BALROG, balrog/environments/babyai_text/__init__.py)."""
    action_strings = ",\n".join(f"{action}: {description}" for action, description in BALROG_ACTIONS.items())

    instruction_prompt = f"""
You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

{action_strings}.

In a moment I will present you an observation.

Tips:
- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.

PLAY!
""".strip()

    return instruction_prompt


class BabyAIText(SequentialEnv):
    name = "babyai_text"
    description = "BabyAI-Text, original code of Carta et al. (MixedTrainLocal, 8 distractors, their prompt); BabyAI bot reference."
    DEFAULT_N = {"eval": 250, "dev": 50}
    SEED_OFFSET = {"eval": 0, "dev": 100_000}

    def __init__(self, ask_success: bool = False, instructions: bool = False):
        self.ask_success = ask_success
        self.instructions = instructions  # diagnostic variant, see module docstring
        if instructions:
            self.name = "babyai_text_instructions"
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
        """The prompt of Carta et al., exactly as their agents receive it (in the diagnostic variant preceded by
        BALROG's instruction prompt)."""
        prompt = generate_prompt(self.obs["mission"], ACTIONS, self.obs_queue, self.acts_queue)
        if self.instructions:
            return {"instructions": balrog_instruction_prompt(self.obs["mission"]), "prompt": prompt}
        return {"prompt": prompt}

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
