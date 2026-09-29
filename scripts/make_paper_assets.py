"""Build all paper tables (paper/tables/*.tex) and figures (paper/figures/*.pdf) from cached responses
and results/eval/. Never calls the Jev API.

    HF_HUB_OFFLINE=1 python scripts/make_paper_assets.py            # uses cached per-example records
    python scripts/make_paper_assets.py --refresh-records            # rebuild records from the cache DB
"""

import argparse
import datetime
import json
import warnings

from jev_benchmarking import thresholds as thr
from jev_benchmarking.config import RESULTS_DIR, ROOT
from jev_benchmarking.report import figures, tables
from jev_benchmarking.report.meta import QUANTITATIVE_SUBJECTS
from jev_benchmarking.report.models import PAPER_MODELS, available_models
from jev_benchmarking.report.records import load_records
from jev_benchmarking.tasks import ALL_TASKS, TASKS

PAPER = ROOT / "paper"
THRESHOLD_TASKS = ("go_emotions", "unfair_tos", "agb_de", "toxic_chat")
THRESHOLD_DEV_LIMIT = 1000  # size of the dev sample thresholds are tuned on (see run.py --limit)
# The full eval run started here; everything earlier in the ledger is the dev-split pilot.
FULL_RUN_START = datetime.datetime(2026, 9, 26, 20, 40).timestamp()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refresh-records", action="store_true")
    args = p.parse_args()
    warnings.filterwarnings("ignore", module="plotnine")

    (PAPER / "tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    results = tables.load_results("eval")
    missing = set(TASKS) - set(results)
    if missing:
        raise SystemExit(f"results/eval is missing {sorted(missing)}; run scripts/evaluate.py first")
    records = {t: load_records(t, "eval", refresh=args.refresh_records) for t in TASKS}

    out = {
        "benchmark_suite": tables.benchmark_suite(results),
        "main_results": tables.main_results(results),
        "score_heads": tables.score_heads(results),
        "calibration": tables.calibration(records),
        "cost": tables.cost(FULL_RUN_START),
    }
    out.update(threshold_table(records, args.refresh_records))
    out.update(probe_table(records, args.refresh_records))
    for name, tex in out.items():
        (PAPER / "tables" / f"{name}.tex").write_text(tex)
        print(f"tables/{name}.tex")

    plots = {
        "reliability": (figures.reliability(records), 6.5, 2.9),
        "selective": (figures.selective(records), 6.5, 3.0),
        "multilingual": (figures.multilingual(records), 4.2, 4.6),
        "subjects": (figures.subjects(records), 6.0, 3.0),
    }
    # Model comparison (only models with complete results; Jev plus the open models run so far).
    compared = available_models("eval")
    if len(compared) >= 2:
        (PAPER / "tables" / "comparison.tex").write_text(tables.comparison(compared))
        print(f"tables/comparison.tex ({', '.join(compared)})")
        recs = {m: records if m == "Jev" else {t: load_records(t, "eval", refresh=args.refresh_records, model=PAPER_MODELS[m]) for t in TASKS} for m in compared}
        plots["reliability_models"] = (figures.reliability_models(recs), 6.5, 3.3)
        plots["selective_models"] = (figures.selective_models(recs), 6.5, 2.6)
    for name, (plot, w, h) in plots.items():
        plot.save(PAPER / "figures" / f"{name}.pdf", width=w, height=h, verbose=False)
        print(f"figures/{name}.pdf")


def _complete(task: str, split: str, df, limit: int | None = None) -> bool:
    expected = len({e.uid for e in ALL_TASKS[task].examples(split, limit=limit)})
    got = df["uid"].nunique() if len(df) else 0
    if got < expected:
        print(f"  skip: {task}/{split} has {got}/{expected} answered examples")
    return got >= expected


def threshold_table(records, refresh: bool) -> dict[str, str]:
    summary = {}
    for t in THRESHOLD_TASKS:
        dev = load_records(t, "dev", limit=THRESHOLD_DEV_LIMIT, refresh=refresh)
        if not _complete(t, "dev", dev, THRESHOLD_DEV_LIMIT):
            return {}
        th = thr.tune(dev)
        b = thr.apply(records[t], th)
        ml = TASKS[t].multilabel
        summary[t] = {
            "thresholds": th,
            "dev_examples": int(dev["uid"].nunique()),
            "fixed": thr.f1_scores(b, "pred_fixed", ml),
            "tuned": thr.f1_scores(b, "pred_tuned", ml),
        }
    (RESULTS_DIR / "eval" / "thresholds.json").write_text(json.dumps(summary, indent=1) + "\n")
    return {"thresholds": tables.thresholds(summary)}


def probe_table(records, refresh: bool) -> dict[str, str]:
    probe = {p: load_records(p, "eval", refresh=refresh) for p in ("probe_mmlu_shuffled", "probe_mmlu_choices_only", "probe_ceval_choices_only")}
    if not all(_complete(p, "eval", df) for p, df in probe.items()):
        return {}

    def original(task):
        d = records[task]
        return d[(d["head"] == "answer") & d["subset"].isin(QUANTITATIVE_SUBJECTS[task])]["correct"].tolist()

    rows = [
        {"benchmark": "MMLU", "condition": "original", "correct": original("mmlu")},
        {"benchmark": "MMLU", "condition": "options rotated", "correct": probe["probe_mmlu_shuffled"]["correct"].tolist()},
        {"benchmark": "MMLU", "condition": "question withheld", "correct": probe["probe_mmlu_choices_only"]["correct"].tolist()},
        {"benchmark": "C-Eval", "condition": "original", "correct": original("ceval")},
        {"benchmark": "C-Eval", "condition": "question withheld", "correct": probe["probe_ceval_choices_only"]["correct"].tolist()},
    ]
    summary = [{"benchmark": r["benchmark"], "condition": r["condition"], "n": len(r["correct"]), "accuracy": sum(r["correct"]) / len(r["correct"])} for r in rows]
    (RESULTS_DIR / "eval" / "probes.json").write_text(json.dumps(summary, indent=1) + "\n")
    return {"probes": tables.probes(rows)}


if __name__ == "__main__":
    main()
