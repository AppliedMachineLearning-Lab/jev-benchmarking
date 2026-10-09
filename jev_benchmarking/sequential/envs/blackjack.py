"""Blackjack with exact values: the one sequential task where the true probabilities are known.

Rules (stated in every request): cards are drawn with replacement (an infinite deck), so the cards seen so
far say nothing about the next one. The player sees their cards and the dealer's face-up card and may only
hit or stand. A player total over 21 loses at once. After the player stands, the dealer draws until their
total is 17 or more and stands on every 17, soft or hard. A dealer total over 21 loses; otherwise the higher
total wins and equal totals tie. A two-card 21 (blackjack) is played out like any other 21, without a bonus.

Because the deck is infinite, the value of every state can be computed exactly by dynamic programming:
the expected return of hitting and standing, the optimal action, and the probability of winning the hand
from here when playing optimally. These are logged with every step (`oracle`), never sent to the model,
and give a ground truth for calibration that an outcome-only benchmark does not have.

Each episode is one hand. Cards come from a stream seeded by the episode id, so every model plays the same
hands (common random numbers), and an answer only changes which cards of the stream are drawn by whom.
"""

import random
from functools import lru_cache

from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS, Decision, EpisodeSpec, SequentialEnv, StepResult, _only_asked
from jev_benchmarking.tasks.base import choice, noul

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
VALUE = {r: (1 if r == "A" else 10 if r in ("10", "J", "Q", "K") else int(r)) for r in RANKS}
# Probability of each card value 1..10 for one draw from an infinite deck.
P_VALUE = {v: (4 / 13 if v == 10 else 1 / 13) for v in range(1, 11)}

RULES = (
    "Blackjack against a dealer. Cards are drawn with replacement from an infinite deck, so earlier cards do "
    "not change the chances of the next card. Number cards count their number, J, Q and K count 10, an ace "
    "counts 11 unless that would bring the total over 21, in which case it counts 1. The player may hit (take "
    "one more card) or stand (stop). A player total over 21 loses immediately. After the player stands, the "
    "dealer reveals the hidden card and draws until the dealer total is 17 or more, standing on every 17. A "
    "dealer total over 21 loses; otherwise the higher total wins and equal totals tie. There is no doubling, "
    "splitting, insurance or blackjack bonus."
)

ACTION_Q = choice(
    "Which move gives the player the best expected result in this hand?",
    {"hit": "take one more card", "stand": "stop and let the dealer play"},
)
SUCCESS_Q = noul("Will the player win this hand against the dealer? A tie does not count as a win.")


def best_total(hard: int, ace: bool) -> int:
    return hard + 10 if ace and hard + 10 <= 21 else hard


def hand(cards: list[str]) -> tuple[int, bool]:
    """(hard total with aces as 1, contains an ace)."""
    return sum(VALUE[c] for c in cards), any(c == "A" for c in cards)


# ----------------------------------------------------------------------------------------------------
# Exact values (infinite deck).


@lru_cache(maxsize=None)
def dealer_final(hard: int, ace: bool) -> tuple[tuple[int, float], ...]:
    """Distribution of the dealer's final total from a dealer hand (hard, ace); 22 stands for a bust."""
    if hard > 21:
        return ((22, 1.0),)
    total = best_total(hard, ace)
    if total >= 17:
        return ((total, 1.0),)
    dist: dict[int, float] = {}
    for v, p in P_VALUE.items():
        for final, q in dealer_final(hard + v, ace or v == 1):
            dist[final] = dist.get(final, 0.0) + p * q
    return tuple(sorted(dist.items()))


@lru_cache(maxsize=None)
def stand_value(total: int, upcard: int) -> tuple[float, float]:
    """(expected return, P(win)) of standing on `total` against a dealer showing `upcard` (1 = ace)."""
    if total > 21:
        return -1.0, 0.0
    win = lose = 0.0
    for final, p in dealer_final(upcard, upcard == 1):
        if final == 22 or final < total:
            win += p
        elif final > total:
            lose += p
    return win - lose, win


@lru_cache(maxsize=None)
def values(hard: int, ace: bool, upcard: int) -> dict:
    """Exact values of a player state under optimal play from here (optimal = highest expected return)."""
    if hard > 21:
        return {"q_hit": -1.0, "q_stand": -1.0, "v": -1.0, "p_win": 0.0, "p_win_hit": 0.0, "p_win_stand": 0.0, "best": "stand"}
    q_stand, w_stand = stand_value(best_total(hard, ace), upcard)
    q_hit = w_hit = 0.0
    for v, p in P_VALUE.items():
        nxt = values(hard + v, ace or v == 1, upcard)
        q_hit += p * nxt["v"]
        w_hit += p * nxt["p_win"]
    best = "hit" if q_hit > q_stand else "stand"
    return {
        "q_hit": q_hit, "q_stand": q_stand, "v": max(q_hit, q_stand),
        "p_win": w_hit if best == "hit" else w_stand,
        "p_win_hit": w_hit, "p_win_stand": w_stand, "best": best,
    }


# ----------------------------------------------------------------------------------------------------


class Blackjack(SequentialEnv):
    name = "blackjack"
    max_steps = 12  # more hits than this always bust
    description = "Hit or stand; exact action values and win probabilities by dynamic programming."
    SEED_OFFSET = {"eval": 0, "dev": 10_000_000}
    DEFAULT_N = {"eval": 5000, "dev": 500}

    def __init__(self, ask_success: bool = False):
        self.ask_success = ask_success

    def episodes(self, split: str = "eval", limit: int | None = None) -> list[EpisodeSpec]:
        n = limit if limit is not None else self.DEFAULT_N[split]
        base = self.SEED_OFFSET[split]
        return [EpisodeSpec(f"{split}-{i:06d}", "", base + i) for i in range(n)]

    def reset(self, spec: EpisodeSpec) -> None:
        self.rng = random.Random(spec.payload)
        draw = self._draw
        self.player = [draw(), draw()]
        self.dealer = [draw(), draw()]  # dealer[0] face up, dealer[1] hidden
        self.over = False
        self.result: StepResult | None = None
        if best_total(*hand(self.player)) == 21:  # two-card 21: nothing to decide
            self._dealer_plays()

    def _draw(self) -> str:
        return self.rng.choice(RANKS)

    def state(self) -> dict:
        hard, ace = hand(self.player)
        return {
            "rules": RULES,
            "player_cards": list(self.player),
            "player_total": best_total(hard, ace),
            "player_total_is_soft": ace and hard + 10 <= 21,
            "dealer_face_up_card": self.dealer[0],
        }

    def decision(self) -> Decision | None:
        if self.over:
            return None
        hard, ace = hand(self.player)
        oracle = dict(values(hard, ace, VALUE[self.dealer[0]]))
        oracle.update(player_total=best_total(hard, ace), soft=ace and hard + 10 <= 21, upcard=self.dealer[0])
        questions = {ACTION: ACTION_Q, SUCCESS: SUCCESS_Q} if self.ask_success else {ACTION: ACTION_Q}
        return Decision(self.state(), questions, {"hit": "hit", "stand": "stand"}, oracle)

    def step(self, action: str) -> StepResult:
        if action == "hit":
            card = self._draw()
            self.player.append(card)
            total = best_total(*hand(self.player))
            if total > 21:
                self.over = True
                self.result = StepResult(True, False, -1.0, f"Player draws {card}: {total}, bust.")
                return self.result
            if total == 21:  # nothing left to gain by hitting
                return self._dealer_plays(f"Player draws {card}: 21. ")
            return StepResult(False, feedback=f"Player draws {card}: {total}.")
        return self._dealer_plays()

    def _dealer_plays(self, prefix: str = "") -> StepResult:
        while best_total(*hand(self.dealer)) < 17:
            self.dealer.append(self._draw())
        p, d = best_total(*hand(self.player)), best_total(*hand(self.dealer))
        reward = 1.0 if d > 21 or p > d else 0.0 if p == d else -1.0
        self.over = True
        self.result = StepResult(True, reward > 0, reward, f"{prefix}Dealer has {' '.join(self.dealer)} = {d}; player {p}.")
        return self.result

    def outcome(self) -> StepResult:
        return self.result or StepResult(False)


# ----------------------------------------------------------------------------------------------------
# Scripted reference policies (FunctionBackend).


def _oracle_of(state: dict) -> dict:
    total, soft = state["player_total"], state["player_total_is_soft"]
    hard, ace = (total - 10, True) if soft else (total, False)
    return values(hard, ace, VALUE[state["dealer_face_up_card"]])


def optimal_policy(state: dict, questions: dict) -> dict:
    """Optimal play with exact win probabilities: the calibrated reference."""
    from jev_benchmarking.sequential.backends import choice_answer, noul_answer

    v = _oracle_of(state)
    return _only_asked(questions, {ACTION: choice_answer({"hit": float(v["best"] == "hit"), "stand": float(v["best"] == "stand")}),
                                   SUCCESS: noul_answer(v["p_win"])})


def never_bust_policy(state: dict, questions: dict) -> dict:
    """Hit below 12 (cannot bust), stand otherwise; reports the base rate of winning as its estimate."""
    from jev_benchmarking.sequential.backends import choice_answer, noul_answer

    hit = state["player_total"] < 12
    return _only_asked(questions, {ACTION: choice_answer({"hit": float(hit), "stand": float(not hit)}), SUCCESS: noul_answer(0.43)})


def random_policy(state: dict, questions: dict) -> dict:
    from jev_benchmarking.sequential.backends import choice_answer, noul_answer

    return _only_asked(questions, {ACTION: choice_answer({"hit": 0.5, "stand": 0.5}), SUCCESS: noul_answer(0.5)})


SCRIPTED = {"optimal": optimal_policy, "never_bust": never_bust_policy, "random": random_policy}
