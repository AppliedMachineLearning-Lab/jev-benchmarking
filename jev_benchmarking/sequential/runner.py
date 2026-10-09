"""Play episodes with a backend and write one JSON line per finished episode (resumable)."""

import hashlib
import json
import logging
import random
import time
from pathlib import Path

from jev_benchmarking.config import RESULTS_DIR, model_slug
from jev_benchmarking.sequential.backends import Backend, BudgetExceeded
from jev_benchmarking.sequential.envs.base import ACTION, SUCCESS, EpisodeSpec, SequentialEnv

log = logging.getLogger(__name__)


def episodes_path(env_name: str, model: str, mode: str, split: str, shard: int = 0, num_shards: int = 1,
                  ask_success: bool = False) -> Path:
    """One file per (env, model, mode, split, with/without success question); requests differ between them."""
    slug = (model_slug(model) or "jev").replace(":", "_")
    variant = ".with-success" if ask_success else ""
    suffix = f".shard{shard:02d}of{num_shards:02d}" if num_shards > 1 else ""
    return RESULTS_DIR / "sequential" / env_name / f"{slug}.{mode}.{split}{variant}{suffix}.jsonl"


def _shard_files(path: Path) -> list[Path]:
    """All episode files of one (env, model, mode, split), sharded or not."""
    stem = path.name.split(".shard")[0].removesuffix(".jsonl")
    return sorted(path.parent.glob(f"{stem}.jsonl")) + sorted(path.parent.glob(f"{stem}.shard*.jsonl"))


def load_episodes(path: Path) -> list[dict]:
    """Episodes from all shards, one per uid (a uid played twice keeps its first record)."""
    seen, out = set(), []
    for p in _shard_files(path):
        for line in p.read_text().splitlines():
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ep["uid"] not in seen:
                seen.add(ep["uid"])
                out.append(ep)
    return out


def finished_uids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            out.add(json.loads(line)["uid"])
        except (json.JSONDecodeError, KeyError):
            pass  # a line cut off by a killed run is redone
    return out


def _rng(uid: str, model: str, t: int) -> random.Random:
    return random.Random(hashlib.sha256(f"{uid}/{model}/{t}".encode()).hexdigest())


def pick(answer: dict, mode: str, rng: random.Random) -> str:
    """greedy: the API's argmax; sample: draw from the returned distribution (seeded per step, so a rerun
    replays the same episode from the cache)."""
    if mode == "greedy":
        return answer["choice"]
    keys, probs = zip(*answer["probabilities"].items())
    return rng.choices(keys, weights=probs, k=1)[0]


def run_episode(env: SequentialEnv, spec: EpisodeSpec, backend: Backend, mode: str = "greedy") -> dict:
    env.reset(spec)
    steps = []
    t0 = time.monotonic()
    result = None
    for t in range(env.max_steps):
        d = env.decision()
        if d is None:
            break
        if hasattr(backend, "observe"):  # scripted experts may read the privileged oracle
            backend.observe(d)
        answers, meta = backend.ask(d.state, d.questions)
        act = answers[ACTION]
        key = pick(act, mode, _rng(spec.uid, backend.model, t))
        if key not in d.actions:
            raise ValueError(f"{spec.uid} step {t}: answer key {key!r} is not a legal action")
        step = {
            "t": t,
            "action": d.actions[key],
            "key": key,
            "probs": {d.actions[k]: p for k, p in act["probabilities"].items()},
            "p_success": answers[SUCCESS]["noul"] if SUCCESS in answers else None,
            "n_actions": len(d.actions),
            "oracle": d.oracle,
            "request": meta["key"],
            "cached": meta["cached"],
            "input_tokens": meta["input_tokens"],
            "latency_s": round(meta["latency_s"], 4),
        }
        result = env.step(d.actions[key])
        step["feedback"] = result.feedback[:500]
        steps.append(step)
        if result.done:
            break
    final = env.outcome()
    return {
        "uid": spec.uid,
        "subset": spec.subset,
        "env": env.name,
        "model": backend.model,
        "mode": mode,
        "success": bool(final.success),
        "reward": final.reward,
        "n_steps": len(steps),
        "truncated": not final.success and len(steps) >= env.max_steps,
        "wall_s": round(time.monotonic() - t0, 3),
        "steps": steps,
    }


def shard_of(uid: str, num_shards: int) -> int:
    return int(hashlib.sha256(uid.encode()).hexdigest(), 16) % num_shards


def run(env: SequentialEnv, backend: Backend, *, split: str = "eval", limit: int | None = None,
        mode: str = "greedy", out: Path | None = None, shard: int = 0, num_shards: int = 1) -> Path:
    """Play all unfinished episodes of this shard. Run shards as separate processes to play episodes in
    parallel (each episode is sequential, episodes are independent); `load_episodes` merges them."""
    out = out or episodes_path(env.name, backend.model, mode, split, shard, num_shards, env.ask_success)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set().union(*(finished_uids(p) for p in _shard_files(out))) if out.suffix == ".jsonl" else set()
    specs = [s for s in env.episodes(split, limit) if s.uid not in done and shard_of(s.uid, num_shards) == shard]
    log.info("%s / %s: %d episodes to play (%d already done)", env.name, backend.model, len(specs), len(done))
    with out.open("a") as f:
        for i, spec in enumerate(specs, 1):
            try:
                ep = run_episode(env, spec, backend, mode)
            except BudgetExceeded as e:
                log.warning("stopping: %s", e)
                break
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(specs):
                log.info("%s: %d/%d episodes, $%.4f spent, %d requests (%d cached)", env.name, i, len(specs),
                         backend.stats.cost_usd, backend.stats.requests, backend.stats.cached)
    env.close()
    return out
