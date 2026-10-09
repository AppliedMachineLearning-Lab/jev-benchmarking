"""Sequential track: exact Blackjack values, episode determinism, cache replay, metrics. No network."""

import json

import httpx2
import numpy as np
import pytest

from jev_benchmarking.cache import ResponseCache
from jev_benchmarking.sequential.backends import BudgetExceeded, FunctionBackend, JevBackend
from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS
from jev_benchmarking.sequential.envs.blackjack import SCRIPTED, Blackjack, dealer_final, values
from jev_benchmarking.sequential.metrics import repeat_metrics, summarize
from jev_benchmarking.sequential.runner import load_episodes, run, run_episode


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


# ---- exact values ----------------------------------------------------------------------------------


def test_dealer_distribution_sums_to_one():
    for up in range(1, 11):
        assert sum(p for _, p in dealer_final(up, up == 1)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "hard,ace,up,best",
    [  # textbook basic strategy without doubling or splitting, dealer stands on soft 17
        (12, False, 2, "hit"), (12, False, 4, "stand"), (12, False, 7, "hit"),
        (13, False, 2, "stand"), (16, False, 10, "hit"), (16, False, 6, "stand"),
        (17, False, 10, "stand"), (11, False, 10, "hit"),
        (7, True, 2, "hit"),   # A+6 = soft 17 vs 2 -> hit
        (8, True, 9, "hit"),   # soft 18 vs 9 -> hit
        (8, True, 7, "stand"),  # soft 18 vs 7 -> stand
        (9, True, 10, "stand"),  # soft 19 -> stand
    ],
)
def test_basic_strategy(hard, ace, up, best):
    assert values(hard, ace, up)["best"] == best


def test_exact_values_match_simulation():
    env, backend = Blackjack(), FunctionBackend("optimal", SCRIPTED["optimal"])
    eps = [run_episode(env, s, backend) for s in env.episodes("dev", 4000)]
    first = [(e["steps"][0]["oracle"]["p_win"], e["success"]) for e in eps if e["steps"]]
    p, y = map(np.array, zip(*first))
    # mean predicted win probability vs observed win rate, within ~3 standard errors
    assert abs(p.mean() - y.mean()) < 3 * np.sqrt(p.mean() * (1 - p.mean()) / len(p))


def test_episodes_are_deterministic():
    env, backend = Blackjack(), FunctionBackend("random", SCRIPTED["random"])
    spec = env.episodes("eval", 50)[17]
    a = run_episode(env, spec, backend, mode="sample")
    b = run_episode(env, spec, backend, mode="sample")
    assert [s["action"] for s in a["steps"]] == [s["action"] for s in b["steps"]] and a["reward"] == b["reward"]


# ---- Jev backend against a fake API -----------------------------------------------------------------


def fake_api(calls: list):
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        calls.append(body)
        hit = body["state"]["player_total"] < 15
        answers = {
            ACTION: {"type": "choice", "choice": "hit" if hit else "stand",
                     "probabilities": {"hit": 0.8 if hit else 0.2, "stand": 0.2 if hit else 0.8}, "confidence": 0.6},
            SUCCESS: {"type": "noul", "noul": 0.4},
        }
        return httpx2.Response(200, json={"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 600, "output_tokens": 0}})

    return httpx2.MockTransport(handler)


def test_jev_episodes_replay_from_cache(tmp_path):
    calls = []
    cache = ResponseCache(tmp_path / "r.db", model="jev-1.13.0")
    env = Blackjack()
    out = tmp_path / "eps.jsonl"
    run(env, JevBackend(cache, max_cost_usd=1.0, transport=fake_api(calls)), limit=30, out=out)
    n_first = len(calls)
    assert n_first > 0 and len(load_episodes(out)) == 30
    out.unlink()  # forget the episodes, keep the cache: a replay must not call the API again
    run(env, JevBackend(cache, max_cost_usd=1.0, transport=fake_api(calls)), limit=30, out=out)
    assert len(calls) == n_first
    assert len(load_episodes(out)) == 30


def test_budget_cap_stops_run(tmp_path):
    calls = []
    cache = ResponseCache(tmp_path / "r.db", model="jev-1.13.0")
    backend = JevBackend(cache, max_cost_usd=0.0001, transport=fake_api(calls))  # ~2,400 tokens
    out = tmp_path / "eps.jsonl"
    run(Blackjack(), backend, limit=200, out=out)
    assert backend.stats.cost_usd <= 0.0001 and len(load_episodes(out)) < 200


def test_sharding_partitions_episodes(tmp_path):
    env, backend = Blackjack(), FunctionBackend("optimal", SCRIPTED["optimal"])
    for shard in range(3):
        run(env, backend, limit=60, out=tmp_path / f"m.greedy.eval.shard{shard:02d}of03.jsonl", shard=shard, num_shards=3)
    eps = load_episodes(tmp_path / "m.greedy.eval.jsonl")
    assert sorted(e["uid"] for e in eps) == [s.uid for s in env.episodes("eval", 60)]


# ---- metrics ---------------------------------------------------------------------------------------


def test_metrics_reference_policy_is_exact():
    env, backend = Blackjack(ask_success=True), FunctionBackend("optimal", SCRIPTED["optimal"])
    s = summarize([run_episode(env, sp, backend) for sp in env.episodes("eval", 500)])
    assert s["optimal_action_rate"] == 1.0
    assert s["mean_regret_per_decision"] == pytest.approx(0.0)
    assert s["success_estimate_mae_vs_exact"] == pytest.approx(0.0)


def test_metrics_weak_policy_has_regret():
    env, backend = Blackjack(), FunctionBackend("never_bust", SCRIPTED["never_bust"])
    s = summarize([run_episode(env, sp, backend) for sp in env.episodes("eval", 500)])
    assert s["optimal_action_rate"] < 0.9 and s["mean_regret_per_decision"] > 0


def test_repeat_metrics_catch_repeats_and_oscillations():
    step = lambda a, f: {"action": a, "feedback": f}  # noqa: E731
    looping = {"steps": [step("go to desk 1", "desk"), step("examine desk 1", "same"), step("examine desk 1", "same"),
                         step("examine desk 1", "same"), step("examine desk 1", "same"), step("go to desk 2", "clock")]}
    oscillating = {"steps": [step("open microwave 1", "open"), step("close microwave 1", "closed"),
                             step("open microwave 1", "open"), step("close microwave 1", "closed"),
                             step("open microwave 1", "open")]}
    revisit = {"steps": [step("go to fridge 1", "closed"), step("go to sink 1", "sink"), step("go to table 1", "t"),
                         step("go to fridge 1", "closed")]}
    m = repeat_metrics([looping, oscillating, revisit])
    assert m["repeat_step_rate"] == pytest.approx((3 + 3 + 0) / 15)
    assert m["episodes_with_loop"] == pytest.approx(2 / 3)

def test_step_brier_matches_single_step_choice_brier():
    """The step-level Brier must equal metrics.choice_metrics' multiclass Brier with the expert as gold."""
    from jev_benchmarking.metrics import choice_metrics

    probs = [{"A": 0.9, "B": 0.05, "C": 0.05}, {"A": 0.6, "B": 0.3, "C": 0.1}, {"A": 0.2, "B": 0.7, "C": 0.1}]
    experts = ["A", "B", "B"]
    steps = [{"t": i, "action": max(p, key=p.get), "probs": p, "p_success": None, "oracle": {"expert": e},
              "input_tokens": 0, "cached": False, "latency_s": 0.0, "feedback": str(i)} for i, (p, e) in enumerate(zip(probs, experts))]
    ep = {"env": "x", "model": "m", "mode": "greedy", "subset": "", "success": True, "reward": 1.0, "n_steps": 3, "steps": steps}
    seq = summarize([ep])["details"]["action_vs_expert_calibration"]
    single = choice_metrics(experts, [{"choice": max(p, key=p.get), "probabilities": p, "confidence": 0.0} for p in probs])
    assert seq["brier"] == pytest.approx(single["brier"])
    assert seq["ece"] == pytest.approx(single["ece"])


def test_success_question_is_off_by_default():
    env = Blackjack()
    env.reset(env.episodes("eval", 5)[1])
    d = env.decision()
    assert d is None or set(d.questions) == {ACTION}
    env = Blackjack(ask_success=True)
    env.reset(env.episodes("eval", 5)[1])
    d = env.decision()
    assert d is None or set(d.questions) == {ACTION, SUCCESS}
