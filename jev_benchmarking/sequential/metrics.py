"""Metrics over episode logs (one dict per episode, as written by `runner.run`)."""

import math
from collections import defaultdict

import numpy as np

from jev_benchmarking.config import USD_PER_INPUT_TOKEN
from jev_benchmarking.metrics import _safe, ece

N_BOOTSTRAP = 1000


def _bootstrap(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (N_BOOTSTRAP, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _calibration(p: np.ndarray, y: np.ndarray) -> dict:
    """Calibration of probabilities `p` against binary outcomes `y`."""
    if len(p) == 0:
        return {}
    return {
        "n": int(len(p)),
        "mean_p": float(p.mean()),
        "rate": float(y.mean()),
        "ece": ece(p, y),
        "brier": float(np.mean((p - y) ** 2)),
    }


CORE = [
    "env", "model", "n_episodes",
    "success_rate", "success_ci95", "mean_steps", "cost_per_episode_usd",
    "repeat_step_rate",
    "step_accuracy_vs_expert", "step_ece", "step_brier",
    "success_estimate_ece", "success_estimate_brier",
]


def summarize(episodes: list[dict]) -> dict:
    """The headline metrics at the top level (see CORE); everything else under "details"."""
    full = _summarize_all(episodes)
    if not episodes:
        return full
    act = full.get("action_vs_expert_calibration", {})
    est = full.get("success_calibration_first_step", {})
    core = {
        "env": full["env"],
        "model": full["model"],
        "n_episodes": full["n_episodes"],
        "success_rate": full["success_rate"],
        "success_ci95": full["success_ci95"],
        "mean_steps": full["mean_steps"],
        "cost_per_episode_usd": full["cost_usd"] / full["n_episodes"],
        "repeat_step_rate": full["repeat_step_rate"],
        "step_accuracy_vs_expert": full.get("expert_agreement"),
        "step_ece": act.get("ece"),
        "step_brier": act.get("brier"),
        "success_estimate_ece": est.get("ece"),
        "success_estimate_brier": est.get("brier"),
    }
    if "exact" in full:  # Blackjack: reward and exact-value metrics replace the expert comparison
        core["mean_reward"] = full["mean_reward"]
        core["optimal_action_rate"] = full["exact"]["optimal_action_rate"]
        core["mean_regret_per_decision"] = full["exact"]["mean_regret_per_decision"]
        core["success_estimate_mae_vs_exact"] = full["exact"].get("success_vs_exact_p_win", {}).get("mae")
    core = {k: v for k, v in core.items() if v is not None}  # e.g. no success estimates unless asked
    core["details"] = full
    return core


def _summarize_all(episodes: list[dict]) -> dict:
    if not episodes:
        return {"n_episodes": 0}
    success = np.array([e["success"] for e in episodes], dtype=float)
    reward = np.array([e["reward"] for e in episodes], dtype=float)
    steps = [s for e in episodes for s in e["steps"]]
    tokens = sum(s["input_tokens"] for s in steps)
    model = episodes[0]["model"]
    out = {
        "env": episodes[0]["env"],
        "model": model,
        "mode": episodes[0]["mode"],
        "n_episodes": len(episodes),
        "success_rate": float(success.mean()),
        "success_ci95": _bootstrap(success),
        "mean_reward": float(reward.mean()),
        "reward_ci95": _bootstrap(reward),
        "mean_steps": float(np.mean([e["n_steps"] for e in episodes])),
        "truncated_rate": float(np.mean([e.get("truncated", False) for e in episodes])),
        "n_requests": len(steps),
        "input_tokens": int(tokens),
        "tokens_per_request": tokens / len(steps) if steps else math.nan,
        "cost_usd": tokens * USD_PER_INPUT_TOKEN if not model.startswith(("hf:", "scripted:")) else 0.0,
        "mean_latency_s": float(np.mean([s["latency_s"] for s in steps if not s["cached"]] or [math.nan])),
    }

    by_subset = defaultdict(list)
    for e in episodes:
        by_subset[e["subset"]].append(e["success"])
    if len(by_subset) > 1:
        out["success_by_subset"] = {k: {"n": len(v), "success_rate": float(np.mean(v))} for k, v in sorted(by_subset.items())}

    # Calibration of the success question against the episode outcome, at the first decision only:
    # one prediction per episode, so the predictions are independent.
    first = [(e["steps"][0]["p_success"], e["success"]) for e in episodes if e["steps"] and e["steps"][0]["p_success"] is not None]
    if first:
        p, y = map(np.array, zip(*first))
        out["success_calibration_first_step"] = _calibration(p.astype(float), y.astype(float))

    out.update(repeat_metrics(episodes))

    top = np.array([max(s["probs"].values()) for s in steps]) if steps else np.array([])
    out["mean_top_action_prob"] = float(top.mean()) if len(top) else math.nan

    # ALFWorld: agreement with the handcoded expert, and calibration of the top action probability
    # against "the chosen command is the expert's".
    expert = [(max(s["probs"].values()), s["action"] == s["oracle"]["expert"]) for s in steps if s["oracle"].get("expert")]
    if expert:
        p, y = map(np.array, zip(*expert))
        out["expert_agreement"] = float(y.mean())
        cal = _calibration(p.astype(float), y.astype(float))
        # Same definitions as the single-step Choice metrics (metrics.choice_metrics): ECE on the top
        # probability, and the multiclass Brier score over the full distribution, with the expert's
        # command as the gold option.
        labelled = [s for s in steps if s["oracle"].get("expert")]
        cal["brier"] = float(np.mean([
            sum((q - (a == s["oracle"]["expert"])) ** 2 for a, q in s["probs"].items()) for s in labelled
        ]))
        out["action_vs_expert_calibration"] = cal

    # Blackjack: exact values.
    exact = [s for s in steps if "q_hit" in s["oracle"]]
    if exact:
        q = lambda s, a: s["oracle"]["q_" + a]  # noqa: E731
        regret = np.array([max(q(s, "hit"), q(s, "stand")) - q(s, s["action"]) for s in exact])
        optimal = np.array([s["action"] == s["oracle"]["best"] for s in exact], dtype=float)
        p_opt = np.array([s["probs"][s["oracle"]["best"]] for s in exact])
        out["exact"] = {
            "optimal_action_rate": float(optimal.mean()),
            "mean_regret_per_decision": float(regret.mean()),
            "mean_prob_on_optimal_action": float(p_opt.mean()),
        }
        with_p = [s for s in exact if s["p_success"] is not None]
        if with_p:
            p = np.array([s["p_success"] for s in with_p], dtype=float)
            star = np.array([s["oracle"]["p_win"] for s in with_p], dtype=float)
            out["exact"]["success_vs_exact_p_win"] = {
                "mae": float(np.abs(p - star).mean()),
                "bias": float((p - star).mean()),
                "pearson": float(np.corrcoef(p, star)[0, 1]) if p.std() > 0 and star.std() > 0 else math.nan,
                "ece_vs_exact": _ece_vs_target(p, star),
            }
    return out


def _ece_vs_target(p: np.ndarray, target: np.ndarray, n_bins: int = 15) -> float:
    """Like ECE, but against known probabilities instead of 0/1 outcomes (no outcome noise)."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).sum() * abs(p[bins == b].mean() - target[bins == b].mean()) for b in np.unique(bins)) / len(p))


def repeat_metrics(episodes: list[dict], max_period: int = 2) -> dict:
    """Loops: a step whose (command, observation) equals that of one of the previous `max_period` steps
    added nothing new. Period 1 catches "examine desk 1" over and over; period 2 catches oscillations such
    as "open microwave 1 / close microwave 1". Returning to a place much later is not counted."""
    n_steps = repeats = 0
    loop_episodes = 0
    for e in episodes:
        pairs = [(s["action"], s.get("feedback", "")) for s in e["steps"]]
        rep = [any(t - k >= 0 and pairs[t] == pairs[t - k] for k in range(1, max_period + 1)) for t in range(len(pairs))]
        n_steps += len(pairs)
        repeats += sum(rep)
        run = best = 0
        for r in rep:
            run = run + 1 if r else 0
            best = max(best, run)
        loop_episodes += best >= 3
    return {
        "repeat_step_rate": repeats / n_steps if n_steps else math.nan,
        "episodes_with_loop": loop_episodes / len(episodes) if episodes else math.nan,  # >= 3 repeats in a row
    }
