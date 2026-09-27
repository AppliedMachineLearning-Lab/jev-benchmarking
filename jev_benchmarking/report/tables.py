"""LaTeX tables in the style of paper/tables/*.tex: bare `tabular`, booktabs, `\\rowcolor{gray!10}` on
alternating rows, `\\multicolumn` group headers. Each function returns the tabular as a string."""

import json
import math
from collections import defaultdict

import pandas as pd

from jev_benchmarking.cache import ResponseCache
from jev_benchmarking.config import RESULTS_DIR, USD_PER_INPUT_TOKEN
from jev_benchmarking.metrics import aurc, ece, selective_accuracy
from jev_benchmarking.report.meta import CATEGORIES, META, METRIC_LABELS, SPLIT_LABELS
from jev_benchmarking.tasks import TASKS

SHADE = "\\rowcolor{gray!10}"


def load_results(split: str = "eval") -> dict[str, dict]:
    return {t: json.loads((RESULTS_DIR / split / f"{t}.json").read_text()) for t in TASKS if (RESULTS_DIR / split / f"{t}.json").is_file()}


def _f(x: float | None, digits: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _tabular(colspec: str, header: str, body: list[str]) -> str:
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "    \\toprule", f"    {header} \\\\", "    \\midrule"]
    lines += [f"    {row}" for row in body]
    lines += ["    \\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def _grouped(ncols: int, rows_by_category: dict[str, list[str]]) -> list[str]:
    body = []
    for i, (cat, rows) in enumerate(rows_by_category.items()):
        if i:
            body.append("\\midrule")
        body.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{cat}}}}} \\\\")
        body += [f"{SHADE if j % 2 else ''}{r} \\\\" for j, r in enumerate(rows)]
    return body


def _dataset(task: str, cite: bool = True) -> str:
    m = META[task]
    return f"{m.name}~\\cite{{{m.cite}}}" if cite else m.name


def benchmark_suite(results: dict[str, dict]) -> str:
    rows = defaultdict(list)
    for cat, tasks in CATEGORIES.items():
        for t in tasks:
            task = TASKS[t]
            split = "test R1--R3" if t == "anli" else SPLIT_LABELS.get(task.splits["eval"], task.splits["eval"])
            res = results.get(t, {})
            langs = str(len(res.get("subsets", {}))) if t in ("belebele", "sib200", "afrixnli") else ""
            hf_path = task.hf_path.replace("_", "\\_")
            rows[cat].append(
                f"{_dataset(t)} & {{\\scriptsize\\texttt{{{hf_path}}}}} & {split} & "
                f"{_int(res.get('n_examples', 0))} & {langs} & {META[t].primitive}"
            )
    return _tabular("l l l r r l", "Dataset & Hugging Face source & Split & $n$ & Lang. & Primitive", _grouped(6, rows))


def main_results(results: dict[str, dict]) -> str:
    rows = defaultdict(list)
    for cat, tasks in CATEGORIES.items():
        for t in tasks:
            r = results[t]
            p = r["primary"]
            metric = METRIC_LABELS.get(p["metric"], p["metric"])
            if p["head"] not in ("answer", "*"):
                metric += f" ({p['head']})"
            ci = p.get("ci95")
            heads = r["heads"]
            ece_val = next((m["ece"] for m in heads.values() if "ece" in m), None)
            rows[cat].append(
                f"{META[t].name} & {_int(r['n_answered'])} & {metric} & {_f(p['value'])} & "
                f"{'--' if not ci else f'[{_f(ci[0])}, {_f(ci[1])}]'} & {_f(ece_val)}"
            )
    return _tabular("l r l c c c", "Dataset & $n$ & Metric & Score$\\uparrow$ & 95\\% CI & ECE$\\downarrow$", _grouped(6, rows))


def score_heads(results: dict[str, dict]) -> str:
    """Per-dimension correlations for everything answered with the Score primitive."""
    names = {"answer": ""}
    body, shade = [], False
    for t in ("stsb", "sst5", "toxigen", "summeval", "helpsteer2"):
        heads = {h: m for h, m in results[t]["heads"].items() if "spearman" in m}
        first = True
        for h, m in heads.items():
            dim = names.get(h, h.capitalize())
            group = _f(m.get("group_spearman")) if "group_spearman" in m else "--"
            ds = META[t].name if first else ""
            body.append(f"{SHADE if shade else ''}{ds} & {dim or '--'} & {_f(m['spearman'])} & {_f(m['kendall'])} & {group} & {_f(m['mae'], 2)} \\\\")
            first = False
        shade = not shade
    return _tabular(
        "l l c c c c",
        "Dataset & Dimension & $\\rho\\uparrow$ & $\\tau\\uparrow$ & $\\rho_{\\text{summary}}\\uparrow$ & MAE$\\downarrow$",
        body,
    )


def calibration(records: dict[str, pd.DataFrame]) -> str:
    """Calibration and selective prediction for every task whose primary head is a Choice."""
    rows = defaultdict(list)
    for cat, tasks in CATEGORIES.items():
        for t in tasks:
            df = records.get(t)
            if df is None or TASKS[t].heads.get("answer") != "choice":
                continue
            d = df[df["head"] == "answer"]
            conf, correct, top = d["confidence"].to_numpy(), d["correct"].to_numpy(), d["prob"].to_numpy()
            rows[cat].append(
                f"{META[t].name} & {_f(correct.mean())} & {_f(ece(top, correct))} & {_f(selective_accuracy(conf, correct, 0.8))} & "
                f"{_f(selective_accuracy(conf, correct, 0.5))} & {_f(aurc(conf, correct))}"
            )
    rows = {c: r for c, r in rows.items() if r}
    return _tabular(
        "l c c c c c",
        "Dataset & Acc.$\\uparrow$ & ECE$\\downarrow$ & Acc.@80\\%$\\uparrow$ & Acc.@50\\%$\\uparrow$ & AURC$\\downarrow$",
        _grouped(6, rows),
    )


def cost(since: float, cache: ResponseCache | None = None) -> str:
    """Requests/tokens/cost/latency of the eval run (responses created at/after `since`, i.e. without the
    dev-split pilot). Latency is client-side wall time per request at 32 concurrent requests."""
    cache = cache or ResponseCache()
    usage = {task: (n, tok, lat) for task, n, tok, _, lat in cache.usage_by_task(since)}
    body, shade = [], False
    tot_n = tot_tok = 0
    lat_weighted = 0.0
    for cat, tasks in CATEGORIES.items():
        n = sum(usage.get(t, (0, 0, 0))[0] for t in tasks)
        tok = sum(usage.get(t, (0, 0, 0))[1] for t in tasks)
        lat = sum(usage.get(t, (0, 0, 0))[0] * usage.get(t, (0, 0, 0))[2] for t in tasks) / max(1, n)
        tot_n, tot_tok, lat_weighted = tot_n + n, tot_tok + tok, lat_weighted + lat * n
        body.append(f"{SHADE if shade else ''}{cat} & {_int(n)} & {tok / 1e6:.1f} & {tok / n:.0f} & {tok * USD_PER_INPUT_TOKEN:.2f} & {lat:.2f} \\\\")
        shade = not shade
    body.append("\\midrule")
    body.append(
        f"Total & {_int(tot_n)} & {tot_tok / 1e6:.1f} & {tot_tok / tot_n:.0f} & {tot_tok * USD_PER_INPUT_TOKEN:.2f} & {lat_weighted / tot_n:.2f} \\\\"
    )
    return _tabular(
        "l r r r r r",
        "Category & Requests & Input tok. (M) & Tok./req. & Cost (USD) & Latency (s)",
        body,
    )


def subset_summary(results: dict[str, dict], task: str, n_extremes: int = 5) -> tuple[list, list]:
    subs = sorted(results[task]["subsets"].items(), key=lambda kv: kv[1]["primary"])
    return subs[:n_extremes], subs[-n_extremes:]
