"""Replay a logged episode step by step and show what the model saw and did.

The episode log stores the actions, option probabilities, reference actions and feedback, not the requests.
This script rebuilds every request by replaying the logged actions in a fresh environment, prints it, and
checks that the replay matches the log (same options, the logged action among them, same feedback). A
mismatch means the environment is not deterministic or the state is not updated as logged.

Examples:
  python scripts/inspect_episode.py frozenlake results/sequential/frozenlake/jev.greedy.eval.jsonl
  python scripts/inspect_episode.py textworld results/sequential/textworld/jev.greedy.eval.jsonl --episode 4
  python scripts/inspect_episode.py babyai_text results/sequential/babyai_text/jev.greedy.eval.jsonl --uid eval-goto-seed000000 --full
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_sequential import make_env  # noqa: E402


def show(value, width: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1)
    return text if width <= 0 or len(text) <= width else text[:width] + f" ... [{len(text) - width} more chars]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env", choices=["blackjack", "frozenlake", "alfworld", "textworld", "babyai_text"])
    ap.add_argument("file", type=Path, help="episode log (.jsonl)")
    ap.add_argument("--episode", type=int, default=0, help="index of the episode in the file")
    ap.add_argument("--uid", help="episode id (overrides --episode)")
    ap.add_argument("--steps", type=int, default=None, help="show only the first N steps")
    ap.add_argument("--full", action="store_true", help="print the full state at every step (default: changes only)")
    ap.add_argument("--width", type=int, default=1200, help="truncate long fields (0 = no limit)")
    args = ap.parse_args()

    episodes = [json.loads(line) for line in args.file.read_text().splitlines() if line.strip()]
    ep = next(e for e in episodes if e["uid"] == args.uid) if args.uid else episodes[args.episode]
    env, _ = make_env(args.env)
    split = ep["uid"].split("-")[0] if ep["uid"].split("-")[0] in ("eval", "dev") else "eval"
    spec = next(s for s in env.episodes(split) if s.uid == ep["uid"])

    print(f"episode {ep['uid']}  model {ep['model']}  success {ep['success']}  reward {ep['reward']}  steps {ep['n_steps']}")
    env.reset(spec)
    previous, problems = {}, 0
    for step in ep["steps"][: args.steps]:
        d = env.decision()
        if d is None:
            print(f"!! step {step['t']}: the replayed episode ended before the log did")
            problems += 1
            break
        state = d.state if isinstance(d.state, dict) else {"state": d.state}
        print(f"\n===== step {step['t']} =====")
        for k, v in state.items():
            if args.full or previous.get(k) != v:
                print(f"[{k}] {show(v, args.width)}")
        previous = dict(state)
        options = list(d.actions.values())
        top = sorted(step["probs"].items(), key=lambda kv: -kv[1])[:4]
        ref = step["oracle"].get("expert") or (step["oracle"].get("optimal") or [step["oracle"].get("best")])[0]
        mark = "OK " if step["action"] == ref or step["action"] in step["oracle"].get("optimal", []) else "-- "
        print(f"options ({len(options)}): {show(options, args.width)}")
        print(f"model: {', '.join(f'{a} {p:.2f}' for a, p in top)}")
        print(f"{mark}chose: {step['action']}   reference: {ref}")
        if step["action"] not in options:
            print("!! the logged action is not among the replayed options")
            problems += 1
        r = env.step(step["action"])
        print(f"feedback: {show(r.feedback, args.width)}")
        if r.feedback[:500] != step["feedback"][:500]:
            print(f"!! feedback differs from the log: {show(step['feedback'], 300)}")
            problems += 1
    env.close()
    print(f"\nreplay {'matches the log' if problems == 0 else f'found {problems} mismatch(es)'}")


if __name__ == "__main__":
    main()
