"""Score benchmark tasks with an open-weight model (exact option probabilities) -> JSONL.

Runs on a GPU node; resumable and shardable. Examples:
    python scripts/run_open_model.py all --split dev --limit 20 --out runs/gemma --device cuda
    python scripts/run_open_model.py all --shard 3 --num-shards 8 --out runs/gemma          # one array task
    python scripts/run_open_model.py boolq --split dev --limit 2 --show                      # print prompts, no model weights
Then locally: python scripts/import_responses.py runs/gemma/*.jsonl
"""

import argparse
import logging
from pathlib import Path

from jev_benchmarking.openmodel.prompts import render
from jev_benchmarking.tasks import get_tasks

DEFAULT_MODEL = "google/gemma-4-E4B-it"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks", nargs="+", help="task names, 'all' or 'probes'")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--split", choices=["eval", "dev"], default="eval")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, default=Path("runs") / "open_model")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-batch-tokens", type=int, default=32_768)
    p.add_argument("--chunk-examples", type=int, default=512)
    p.add_argument("--show", action="store_true", help="print the rendered prompts of the first example per task and exit")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    tasks = get_tasks(args.tasks)

    if args.show:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model)
        single = lambda c: len(tok.encode(c, add_special_tokens=False)) == 1  # noqa: E731
        for task in tasks:
            if task.splits.get(args.split) is None:
                continue
            e = task.examples(args.split, limit=args.limit or 1)[0]
            for qid, q in list(e.questions.items())[:2]:
                rq = render(e.state, q, single)
                n = len(tok.apply_chat_template([{"role": "user", "content": rq.text}], tokenize=True, add_generation_prompt=True)["input_ids"])
                print(f"===== {task.name} / {qid} ({n} prompt tokens, codes {rq.codes[:8]}{'...' if len(rq.codes) > 8 else ''}) gold={e.gold.get(qid)}\n{rq.text}\n")
        return

    from jev_benchmarking.openmodel.runner import run
    from jev_benchmarking.openmodel.scorer import HFScorer

    scorer = HFScorer(args.model, device=args.device, dtype=args.dtype, max_batch_tokens=args.max_batch_tokens)
    n = run(scorer, args.model, tasks, args.split, args.limit, args.out, args.shard, args.num_shards, args.chunk_examples)
    print(f"shard {args.shard}/{args.num_shards}: wrote {n} responses to {args.out}")


if __name__ == "__main__":
    main()
