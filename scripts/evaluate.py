"""Compute metrics from cached responses (no API calls) and write results/<split>/.

Examples:
    python scripts/evaluate.py all --split dev --limit 20
    python scripts/evaluate.py belebele mmlu
"""

import argparse
import json
import math

from jev_benchmarking.cache import ResponseCache
from jev_benchmarking.config import RESULTS_DIR
from jev_benchmarking.evaluate import evaluate
from jev_benchmarking.tasks import get_tasks


def _fmt(x: float) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", nargs="*", default=["all"])
    p.add_argument("--split", choices=["eval", "dev"], default="eval")
    p.add_argument("--limit", type=int, help="use the same value as for run.py to score that sample")
    p.add_argument("--subsets", nargs="+")
    p.add_argument("--no-ci", action="store_true", help="skip bootstrap confidence intervals")
    args = p.parse_args()

    cache = ResponseCache()
    out_dir = RESULTS_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in get_tasks(args.tasks):
        if task.splits.get(args.split) is None:
            continue
        try:
            examples = task.examples(args.split, limit=args.limit, subsets=args.subsets)
        except Exception as e:
            print(f"- {task.name}: could not load ({type(e).__name__})")
            continue
        res = evaluate(task, examples, cache, with_ci=not args.no_ci)
        if task.skipped_configs:
            res["skipped_configs"] = task.skipped_configs
            print(f"  !! {task.name}: {len(task.skipped_configs)} config(s) skipped: {', '.join(task.skipped_configs)}")
        if not res["n_answered"]:
            continue
        (out_dir / f"{task.name}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
        prim = res["primary"]
        ci = prim.get("ci95")
        heads = res["heads"]
        # Calibration column: ECE of the first choice/binary head, if any.
        ece = next((m["ece"] for m in heads.values() if "ece" in m), math.nan)
        rows.append(
            f"| {task.name} | {res['n_answered']}/{res['n_examples']} | {prim['metric']}"
            f"{'' if prim['head'] in ('answer', '*') else ' (' + prim['head'] + ')'} | {_fmt(prim.get('value'))} | "
            f"{'–' if not ci else f'{_fmt(ci[0])}–{_fmt(ci[1])}'} | {_fmt(ece)} |"
        )
        print(rows[-1])

    table = "| task | answered/total | metric | value | 95% CI | ECE |\n|---|---|---|---|---|---|\n" + "\n".join(rows)
    (out_dir / "summary.md").write_text(table + "\n")
    print(f"\nWrote {out_dir}/")


if __name__ == "__main__":
    main()
