"""Diagnostic probe: does Jev's FrozenLake failure depend on how the request is worded?

Asks for the best move at a fixed set of positions under several prompt variants (see `VARIANTS` in
`envs/frozenlake.py`): the same positions for every variant, one request per position, no episodes, so
early mistakes do not compound and variants are compared on identical inputs.

Positions: for each of the first `--n-maps` evaluation maps (alternating 4x4 and 8x8, the same maps as the
episodes), `--per-map` frozen tiles drawn with a fixed seed among those from which the goal can be reached
and on which not every move is optimal, with the full step budget left.

Examples:
  python scripts/probe_frozenlake.py --backend scripted:oracle                 # sanity check, free
  python scripts/probe_frozenlake.py --backend scripted:no-slip                # plans as if the ice were not slippery
  python scripts/probe_frozenlake.py --backend jev --max-cost 0.2               # all four variants
  python scripts/probe_frozenlake.py --backend jev --variants base,outcomes
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_benchmarking.cache import ResponseCache  # noqa: E402
from jev_benchmarking.config import MODEL, RESULTS_DIR, model_slug  # noqa: E402
from jev_benchmarking.metrics import ece  # noqa: E402
from jev_benchmarking.sequential.backends import FunctionBackend, JevBackend, OpenModelBackend, choice_answer  # noqa: E402
from jev_benchmarking.sequential.envs.base import ACTION  # noqa: E402
from jev_benchmarking.sequential.envs.frozenlake import (  # noqa: E402
    MOVES, STEP_LIMIT, VARIANTS, FrozenLake, build_request, deterministic_values, state_values,
)
from gymnasium.envs.toy_text.frozen_lake import generate_random_map  # noqa: E402

AWAY = {"left", "up"}  # the goal is always at the bottom right


def positions(n_maps: int, per_map: int, seed: int = 0) -> list[dict]:
    out = []
    for spec in FrozenLake().episodes("eval", n_maps):
        size, map_seed = spec.payload["size"], spec.payload["seed"]
        desc = tuple(generate_random_map(size=size, p=0.8, seed=map_seed))
        limit = STEP_LIMIT[size]
        cand = []
        for r in range(size):
            for c in range(size):
                if desc[r][c] in "HG":
                    continue
                v = state_values(desc, r, c, limit, limit)
                if v["p_win"] > 0 and len(v["optimal"]) < 4:
                    cand.append((r, c))
        rng = random.Random(f"{seed}/{spec.uid}")
        for r, c in rng.sample(cand, min(per_map, len(cand))):
            out.append({"uid": f"{spec.uid}/r{r}c{c}", "size": f"{size}x{size}", "desc": desc, "row": r, "col": c, "limit": limit})
    return out


def scripted(name: str):
    """oracle: the reference move of each variant. no-slip: the shortest safe path, i.e. planning as if the
    ice were not slippery (what a model that ignores slipping would do)."""
    def fn(state, questions):
        desc, r, c = tuple(state["map"]), state["agent_row"], state["agent_column"]
        if name == "no-slip" or "not slippery" in state["rules"]:
            best = deterministic_values(desc, r, c)["best"]
        else:
            best = state_values(desc, r, c, state["steps_left"], STEP_LIMIT[len(desc)])["best"]
        return {ACTION: choice_answer({m: float(m == best) for m in MOVES})}
    return fn


def make_backend(spec: str, max_cost: float, device: str):
    if spec == "jev":
        return JevBackend(ResponseCache(model=MODEL), max_cost_usd=max_cost)
    if spec.startswith("hf:"):
        return OpenModelBackend(spec[3:], ResponseCache(model=spec), device=device)
    if spec.startswith("scripted:"):
        return FunctionBackend(spec.split(":", 1)[1], scripted(spec.split(":", 1)[1]))
    raise SystemExit(f"unknown backend {spec!r}")


def summarize(rows: list[dict]) -> dict:
    ok = np.array([r["choice"] in r["optimal"] for r in rows], dtype=float)
    top = np.array([max(r["probs"].values()) for r in rows])
    away = np.array([set(r["optimal"]) <= AWAY for r in rows])
    p_opt = np.array([sum(q for m, q in r["probs"].items() if m in r["optimal"]) for r in rows])
    brier = np.mean([(sum(q for m, q in r["probs"].items() if m in r["optimal"]) - 1) ** 2
                     + sum(q ** 2 for m, q in r["probs"].items() if m not in r["optimal"]) for r in rows])
    out = {
        "n": len(rows),
        "step_accuracy": float(ok.mean()),
        "accuracy_optimal_toward_goal": float(ok[~away].mean()) if (~away).any() else None,
        "accuracy_optimal_away_from_goal": float(ok[away].mean()) if away.any() else None,
        "share_optimal_away": float(away.mean()),
        "mean_prob_on_optimal": float(p_opt.mean()),
        "mean_top_prob": float(top.mean()),
        "top_prob_when_right": float(top[ok == 1].mean()) if ok.any() else None,
        "top_prob_when_wrong": float(top[ok == 0].mean()) if (ok == 0).any() else None,
        "ece": ece(top, ok),
        "brier": float(brier),
        "choices": {m: sum(r["choice"] == m for r in rows) for m in MOVES},
    }
    if "q" in rows[0]:
        out["mean_regret"] = float(np.mean([max(r["q"].values()) - r["q"][r["choice"]] for r in rows]))
        out["choice_can_slip_into_hole"] = float(np.mean([r["risky"][r["choice"]] for r in rows]))
    else:
        out["choice_steps_onto_hole"] = float(np.mean([r["q_steps"][r["choice"]] == -1000.0 for r in rows]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="jev")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-maps", type=int, default=100)
    ap.add_argument("--per-map", type=int, default=5)
    ap.add_argument("--max-cost", type=float, default=0.2, help="USD cap for this run (Jev only)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pos = positions(args.n_maps, args.per_map)
    backend = make_backend(args.backend, args.max_cost, args.device)
    slug = (model_slug({"jev": MODEL}.get(args.backend, args.backend)) or "jev").replace(":", "_")
    out_dir = RESULTS_DIR / "sequential" / "frozenlake" / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for variant in args.variants.split(","):
        rows = []
        for p in pos:
            state, questions, oracle = build_request(p["desc"], p["row"], p["col"], 0, p["limit"], variant)
            answers, meta = backend.ask(state, questions)
            a = answers[ACTION]
            row = {"uid": p["uid"], "size": p["size"], "variant": variant, "choice": a["choice"],
                   "probs": a["probabilities"], "optimal": oracle["optimal"], "cached": meta["cached"],
                   "input_tokens": meta["input_tokens"]}
            if "q" in oracle:
                row["q"] = oracle["q"]
                row["risky"] = {m: any(p["desc"][rr][cc] == "H" for rr, cc in _landings(p, m)) for m in MOVES}
            else:
                row["q_steps"] = oracle["q_steps"]
            rows.append(row)
        (out_dir / f"{slug}.{variant}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary[variant] = summarize(rows)
        logging.info("%s: %d positions, accuracy %.3f", variant, len(rows), summary[variant]["step_accuracy"])
    path = out_dir / f"{slug}.summary.json"
    path.write_text(json.dumps(summary, indent=1))
    keys = ["step_accuracy", "accuracy_optimal_toward_goal", "accuracy_optimal_away_from_goal", "mean_prob_on_optimal",
            "top_prob_when_right", "top_prob_when_wrong", "ece", "brier", "mean_regret", "choice_can_slip_into_hole",
            "choice_steps_onto_hole"]
    print(f"{'variant':14s}" + "".join(f"{k[:14]:>15s}" for k in keys))
    for v, s in summary.items():
        print(f"{v:14s}" + "".join(f"{s[k]:15.3f}" if isinstance(s.get(k), float) else f"{'-':>15s}" for k in keys))
    print(f"positions: {len(pos)}  cost so far: ${backend.stats.cost_usd:.4f}  results: {out_dir}")


def _landings(p: dict, move: str):
    from jev_benchmarking.sequential.envs.frozenlake import PERPENDICULAR, landing

    return [landing(p["desc"], p["row"], p["col"], d) for d in (move, *PERPENDICULAR[move])]


if __name__ == "__main__":
    main()
