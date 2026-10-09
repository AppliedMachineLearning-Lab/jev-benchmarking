"""Play sequential benchmarks and summarize them.

Examples:
  python scripts/run_sequential.py blackjack --backend scripted:optimal --limit 2000      # reference, free
  python scripts/run_sequential.py blackjack --dry-run --limit 5000                       # token/cost estimate
  python scripts/run_sequential.py blackjack --backend jev --limit 5000 --max-cost 1.0
  python scripts/run_sequential.py frozenlake --backend scripted:optimal --limit 500
  python scripts/run_sequential.py alfworld --backend scripted:expert                     # expert reference
  python scripts/run_sequential.py alfworld --backend jev --split eval --max-cost 2.0
  python scripts/run_sequential.py alfworld --backend hf:Qwen/Qwen3.8-27B --device auto
  python scripts/run_sequential.py alfworld --backend jev --shard 3 --num-shards 8        # one of 8 parallel processes
  python scripts/run_sequential.py blackjack --backend jev --summarize                    # metrics only (all shards)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_benchmarking.cache import ResponseCache  # noqa: E402
from jev_benchmarking.config import MODEL, USD_PER_INPUT_TOKEN  # noqa: E402
from jev_benchmarking.sequential.backends import FunctionBackend, JevBackend, OpenModelBackend, estimate_tokens  # noqa: E402
from jev_benchmarking.sequential.metrics import summarize  # noqa: E402
from jev_benchmarking.sequential.runner import episodes_path, load_episodes, run  # noqa: E402


def make_env(name: str):
    if name == "blackjack":
        from jev_benchmarking.sequential.envs import blackjack

        return blackjack.Blackjack(), blackjack.SCRIPTED
    if name == "frozenlake":
        from jev_benchmarking.sequential.envs import frozenlake

        return frozenlake.FrozenLake(), frozenlake.SCRIPTED
    if name == "alfworld":
        from jev_benchmarking.sequential.envs import alfworld

        return alfworld.ALFWorld(), {**alfworld.SCRIPTED, "expert": None}
    raise SystemExit(f"unknown env {name!r}")


def make_backend(spec: str, env_name: str, scripted: dict, args):
    if spec == "jev":
        return JevBackend(ResponseCache(model=MODEL), max_cost_usd=args.max_cost)
    if spec.startswith("hf:"):
        model_id = spec[3:]
        return OpenModelBackend(model_id, ResponseCache(model=spec), device=args.device)
    if spec.startswith("scripted:"):
        name = spec.split(":", 1)[1]
        if env_name == "alfworld" and name == "expert":
            from jev_benchmarking.sequential.envs.alfworld import ExpertBackend

            return ExpertBackend.make()
        if name not in scripted:
            raise SystemExit(f"{env_name} has no scripted policy {name!r}; choose from {sorted(scripted)}")
        return FunctionBackend(name, scripted[name])
    raise SystemExit(f"unknown backend {spec!r}")


class _CountingBackend(FunctionBackend):
    """Dry run: plays with a scripted policy and records the estimated size of every request."""

    def __init__(self, inner: FunctionBackend):
        super().__init__(inner.model.split(":", 1)[1], inner.fn)
        self.est = []
        if hasattr(inner, "observe"):
            self.observe = inner.observe

    def _answer(self, state, questions):
        self.est.append(estimate_tokens(state, questions))
        return super()._answer(state, questions)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env", choices=["blackjack", "frozenlake", "alfworld"])
    ap.add_argument("--backend", default="jev", help="jev | hf:<model id> | scripted:<policy>")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--ask-success", action="store_true", help="also ask the optional success question at every step")
    ap.add_argument("--max-cost", type=float, default=0.5, help="USD cap for this run (Jev only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", type=int, default=0, help="play only this shard (run shards as parallel processes)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="estimate requests, tokens and Jev cost with a scripted policy")
    ap.add_argument("--summarize", action="store_true", help="only compute metrics from existing episode logs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    env, scripted = make_env(args.env)
    env.ask_success = args.ask_success
    if args.dry_run:
        policy = "expert" if args.env == "alfworld" else "optimal"
        backend = _CountingBackend(make_backend(f"scripted:{policy}", args.env, scripted, args))
        out = run(env, backend, split=args.split, limit=args.limit, mode=args.mode, out=Path("/dev/null"))
        n_ep = len(env.episodes(args.split, args.limit))
        tok = sum(backend.est)
        print(json.dumps({
            "env": args.env, "split": args.split, "episodes": n_ep, "policy_for_estimate": policy,
            "requests": len(backend.est), "requests_per_episode": len(backend.est) / max(n_ep, 1),
            "est_input_tokens": tok, "est_tokens_per_request": tok / max(len(backend.est), 1),
            "est_jev_cost_usd": round(tok * USD_PER_INPUT_TOKEN, 4),
            "note": "a weaker policy takes more steps; ALFWorld episodes cap at 50 steps (worst case ~50x episodes requests)",
        }, indent=1))
        return

    backend_model = {"jev": MODEL}.get(args.backend, args.backend)
    path = episodes_path(args.env, backend_model, args.mode, args.split, ask_success=args.ask_success)
    if not args.summarize:
        backend = make_backend(args.backend, args.env, scripted, args)
        run(env, backend, split=args.split, limit=args.limit, mode=args.mode, shard=args.shard, num_shards=args.num_shards)
    episodes = load_episodes(path)
    summary = summarize(episodes)
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, indent=1))  # full set is in the file
    print(f"episodes: {path}\nsummary:  {summary_path}")


if __name__ == "__main__":
    main()
