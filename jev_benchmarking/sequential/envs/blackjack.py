"""Blackjack on RLCard's implementation, with exact action values computed alongside.

The game is RLCard's `blackjack` environment (Zha et al., 2019) with its default configuration, the setup
on which earlier LLM agents (Agent-Pro, Zhang et al., 2024) and RL agents (DQN, DMC) report win rates:
one player against a dealer, a fresh shuffled 52-card deck for every hand, hit or stand only, the dealer
draws until 17 or more and stands on every 17, no doubling, splitting, insurance or blackjack bonus. The
dealer's first card is hidden and the second is shown. RLCard deals, scores and judges every hand; this
module only renders the state as text for the model and computes the reference values.

Because every hand starts from a full deck, the value of every decision can be computed exactly given the
cards the player can see (own cards and the dealer's face-up card; the hidden card and every later card
are a uniform draw from the rest of the deck): the expected return of hitting and standing, the optimal
action, and the probability of winning the hand from here when playing optimally. These are logged with
every step (`oracle`) and never sent to the model.

Each episode is one hand, seeded by its index, so every model is dealt the same starting hand; after that
the cards depend on the actions taken, as in RLCard.
"""

from functools import lru_cache

import rlcard

from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS, Decision, EpisodeSpec, SequentialEnv, StepResult, _only_asked
from jev_benchmarking.tasks.base import choice, noul

RANK_NAME = {"A": "A", "T": "10"}  # RLCard writes ten as "T"
VALUE = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "J": 10, "Q": 10, "K": 10}
FULL_DECK = (4, 4, 4, 4, 4, 4, 4, 4, 4, 16)  # cards of value 1 (ace) .. 10 in one 52-card deck

RULES = (
    "Blackjack against a dealer, one 52-card deck shuffled for every hand. Number cards count their number, "
    "J, Q and K count 10, an ace counts 11 unless that would bring the total over 21, in which case it counts "
    "1. The dealer has one card face up and one hidden. The player may hit (take one more card) or stand "
    "(stop). A player total over 21 loses immediately. After the player stands, the dealer reveals the "
    "hidden card and draws until the dealer total is 17 or more, standing on every 17. A dealer total over "
    "21 loses; otherwise the higher total wins and equal totals tie. There is no doubling, splitting, "
    "insurance or blackjack bonus."
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


def remaining_deck(seen: list[str]) -> tuple[int, ...]:
    deck = list(FULL_DECK)
    for c in seen:
        deck[VALUE[c] - 1] -= 1
    return tuple(deck)


def _draws(deck: tuple[int, ...]):
    """(card value, probability, deck after the draw) for one draw from `deck`."""
    n = sum(deck)
    for i, k in enumerate(deck):
        if k:
            yield i + 1, k / n, deck[:i] + (k - 1,) + deck[i + 1:]


# ----------------------------------------------------------------------------------------------------
# Exact values (one deck, composition-dependent).


@lru_cache(maxsize=None)
def dealer_final(hard: int, ace: bool, deck: tuple[int, ...]) -> tuple[tuple[int, float], ...]:
    """Distribution of the dealer's final total from a dealer hand (hard, ace) drawing from `deck`;
    22 stands for a bust."""
    if hard > 21:
        return ((22, 1.0),)
    total = best_total(hard, ace)
    if total >= 17:
        return ((total, 1.0),)
    dist: dict[int, float] = {}
    for v, p, rest in _draws(deck):
        for final, q in dealer_final(hard + v, ace or v == 1, rest):
            dist[final] = dist.get(final, 0.0) + p * q
    return tuple(sorted(dist.items()))


@lru_cache(maxsize=None)
def stand_value(total: int, upcard: int, deck: tuple[int, ...]) -> tuple[float, float]:
    """(expected return, P(win)) of standing on `total`; the dealer's hidden card and draws come from `deck`."""
    if total > 21:
        return -1.0, 0.0
    win = lose = 0.0
    for final, p in dealer_final(upcard, upcard == 1, deck):
        if final == 22 or final < total:
            win += p
        elif final > total:
            lose += p
    return win - lose, win


@lru_cache(maxsize=None)
def values(hard: int, ace: bool, upcard: int, deck: tuple[int, ...]) -> dict:
    """Exact values of a player state under optimal play from here (optimal = highest expected return).
    `deck` is the deck without the player's cards and the dealer's face-up card."""
    if hard > 21:
        return {"q_hit": -1.0, "q_stand": -1.0, "v": -1.0, "p_win": 0.0, "p_win_hit": 0.0, "p_win_stand": 0.0, "best": "stand"}
    q_stand, w_stand = stand_value(best_total(hard, ace), upcard, deck)
    q_hit = w_hit = 0.0
    for v, p, rest in _draws(deck):
        nxt = values(hard + v, ace or v == 1, upcard, rest)
        q_hit += p * nxt["v"]
        w_hit += p * nxt["p_win"]
    best = "hit" if q_hit > q_stand else "stand"
    return {
        "q_hit": q_hit, "q_stand": q_stand, "v": max(q_hit, q_stand),
        "p_win": w_hit if best == "hit" else w_stand,
        "p_win_hit": w_hit, "p_win_stand": w_stand, "best": best,
    }


def state_values(player_cards: list[str], upcard: str) -> dict:
    hard, ace = hand(player_cards)
    return values(hard, ace, VALUE[upcard], remaining_deck(player_cards + [upcard]))


# ----------------------------------------------------------------------------------------------------


def _name(card) -> str:
    return RANK_NAME.get(card.rank, card.rank)


class Blackjack(SequentialEnv):
    name = "blackjack"
    max_steps = 12  # at most 11 cards fit under 22 with one deck
    description = "RLCard Blackjack (hit or stand); exact action values and win probabilities by dynamic programming."
    SEED_OFFSET = {"eval": 0, "dev": 10_000_000}
    DEFAULT_N = {"eval": 5000, "dev": 500}

    def __init__(self, ask_success: bool = False):
        self.ask_success = ask_success
        self.env = rlcard.make("blackjack", config={"game_num_players": 1, "game_num_decks": 1})

    def episodes(self, split: str = "eval", limit: int | None = None) -> list[EpisodeSpec]:
        n = limit if limit is not None else self.DEFAULT_N[split]
        base = self.SEED_OFFSET[split]
        return [EpisodeSpec(f"{split}-{i:06d}", "", base + i) for i in range(n)]

    def reset(self, spec: EpisodeSpec) -> None:
        self.env.seed(spec.payload)
        self.env.reset()
        self.result: StepResult | None = None

    @property
    def _game(self):
        return self.env.game

    @property
    def player(self) -> list[str]:
        return [_name(c) for c in self._game.players[0].hand]

    @property
    def dealer(self) -> list[str]:
        return [_name(c) for c in self._game.dealer.hand]

    @property
    def upcard(self) -> str:
        return self.dealer[1]  # RLCard hides the dealer's first card and shows the second

    def state(self) -> dict:
        hard, ace = hand(self.player)
        return {
            "rules": RULES,
            "player_cards": self.player,
            "player_total": best_total(hard, ace),
            "player_total_is_soft": ace and hard + 10 <= 21,
            "dealer_face_up_card": self.upcard,
        }

    def decision(self) -> Decision | None:
        if self.env.is_over():
            return None
        hard, ace = hand(self.player)
        oracle = dict(state_values(self.player, self.upcard))
        oracle.update(player_total=best_total(hard, ace), soft=ace and hard + 10 <= 21, upcard=self.upcard)
        questions = {ACTION: ACTION_Q, SUCCESS: SUCCESS_Q} if self.ask_success else {ACTION: ACTION_Q}
        return Decision(self.state(), questions, {"hit": "hit", "stand": "stand"}, oracle)

    def step(self, action: str) -> StepResult:
        self.env.step(action, raw_action=True)
        p = best_total(*hand(self.player))
        if not self.env.is_over():
            return StepResult(False, feedback=f"Player draws {self.player[-1]}: {p}.")
        reward = float(self.env.get_payoffs()[0])
        d = best_total(*hand(self.dealer))
        drew = f"Player draws {self.player[-1]}: {p}. " if action == "hit" else ""
        fb = f"{drew}Bust." if p > 21 else f"{drew}Dealer has {' '.join(self.dealer)} = {d}; player {p}."
        self.result = StepResult(True, reward > 0, reward, fb)
        return self.result

    def outcome(self) -> StepResult:
        return self.result or StepResult(False)


# ----------------------------------------------------------------------------------------------------
# Scripted reference policies (FunctionBackend).


def _oracle_of(state: dict) -> dict:
    return state_values(state["player_cards"], state["dealer_face_up_card"])


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
