"""Build all paper tables (paper/tables/*.tex) and figures (paper/figures/*.pdf) from cached responses
and results/eval/. Never calls the Jev API.

    HF_HUB_OFFLINE=1 python scripts/make_paper_assets.py            # uses cached per-example records
    python scripts/make_paper_assets.py --refresh-records            # rebuild records from the cache DB
"""

import argparse
import datetime
import warnings

from jev_benchmarking.config import ROOT
from jev_benchmarking.report import figures, tables
from jev_benchmarking.report.records import load_records
from jev_benchmarking.tasks import TASKS

PAPER = ROOT / "paper"
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
    for name, tex in out.items():
        (PAPER / "tables" / f"{name}.tex").write_text(tex)
        print(f"tables/{name}.tex")

    plots = {
        "reliability": (figures.reliability(records), 6.5, 2.9),
        "selective": (figures.selective(records), 6.5, 3.0),
        "multilingual": (figures.multilingual(records), 4.2, 4.6),
        "subjects": (figures.subjects(records), 6.0, 3.0),
    }
    for name, (plot, w, h) in plots.items():
        plot.save(PAPER / "figures" / f"{name}.pdf", width=w, height=h, verbose=False)
        print(f"figures/{name}.pdf")


if __name__ == "__main__":
    main()
