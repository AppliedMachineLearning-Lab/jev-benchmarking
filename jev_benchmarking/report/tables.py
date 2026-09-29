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


def score_heads(results_by_model: dict[str, dict[str, dict]]) -> str:
    """Spearman correlation per Score dimension and model (summary-level for SummEval), plus Jev's MAE."""
    labels = list(results_by_model)
    first = results_by_model[labels[0]]
    body, shade = [], False
    for t in ("stsb", "sst5", "toxigen", "summeval", "helpsteer2"):
        heads = [h for h, m in first[t]["heads"].items() if "spearman" in m]
        for k, h in enumerate(heads):
            metric = "group_spearman" if t == "summeval" else "spearman"
            vals = [results_by_model[m][t]["heads"][h][metric] for m in labels]
            best = max(vals)
            cells = " & ".join(f"\\textbf{{{_f(v)}}}" if v == best else _f(v) for v in vals)
            dim = "--" if h == "answer" else h.capitalize()
            ds = META[t].name if k == 0 else ""
            body.append(f"{SHADE if shade else ''}{ds} & {dim} & {cells} & {_f(first[t]['heads'][h]['mae'], 2)} \\\\")
        shade = not shade
    header = (f"& & \\multicolumn{{{len(labels)}}}{{c}}{{Spearman $\\rho\\uparrow$}} & \\\\\n"
              f"    \\cmidrule(lr){{3-{2 + len(labels)}}}\n    Dataset & Dimension & {' & '.join(labels)} & MAE$\\downarrow$ ({labels[0]})")
    return _tabular("l l " + "c " * len(labels) + "c", header, body)


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


def cost(since: float, until: float, cache: ResponseCache | None = None) -> str:
    """Requests/tokens/cost/latency of the eval run: responses created in [since, until), which excludes the
    dev-split pilot before it and the threshold samples/probes after it (those reuse task names, so a window
    open at the end would count them). Latency is client-side wall time per request at 32 concurrent requests."""
    cache = cache or ResponseCache()
    usage = {task: (n, tok, lat) for task, n, tok, _, lat in cache.usage_by_task(since, until)}
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


def _boot_ci(x, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


THRESHOLD_ROWS = [
    ("go_emotions", "macro_f1", "Macro-F1"),
    ("go_emotions", "micro_f1", "Micro-F1"),
    ("unfair_tos", "micro_f1", "Micro-F1"),
    ("unfair_tos", "macro_f1", "Macro-F1"),
    ("agb_de", "f1/answer", "F1"),
    ("toxic_chat", "f1/toxic", "F1 (toxicity)"),
    ("toxic_chat", "f1/jailbreak", "F1 (jailbreaking)"),
]


def thresholds(summary: dict[str, dict[str, dict]]) -> str:
    """Fixed 0.5 vs. per-question thresholds tuned on a training/validation sample, per model."""
    labels = list(summary)
    body, prev, shade = [], None, True
    for t, key, label in THRESHOLD_ROWS:
        if t != prev:
            shade = not shade
        cells = []
        for m in labels:
            fixed, tuned = summary[m][t]["fixed"][key], summary[m][t]["tuned"][key]
            cells += [_f(fixed), _f(tuned)]
        name = META[t].name if t != prev else ""
        body.append(f"{SHADE if shade else ''}{name} & {label} & {' & '.join(cells)} \\\\")
        prev = t
    groups = " & ".join(f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in labels)
    rules = " ".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(labels)))
    header = f"& & {groups} \\\\\n    {rules}\n    Dataset & Metric & " + " & ".join(["0.5", "tuned"] * len(labels))
    return _tabular("l l " + "c " * (2 * len(labels)), header, body)


def probes(rows: list[dict], labels: list[str]) -> str:
    """Calculation-heavy vs. other subjects and the two memorization probes; one accuracy column per model.
    rows: {"benchmark", "condition", "n", "acc": {label: accuracy}}."""
    body, prev, shade = [], None, True
    for r in rows:
        if r["benchmark"] != prev:
            shade = not shade
        name = r["benchmark"] if r["benchmark"] != prev else ""
        cells = " & ".join(_f(r["acc"][m]) for m in labels)
        body.append(f"{SHADE if shade else ''}{name} & {r['condition']} & {_int(r['n'])} & {cells} \\\\")
        prev = r["benchmark"]
    return _tabular("l l r " + "c " * len(labels), "Benchmark & Subjects / condition & $n$ & " + " & ".join(labels), body)


def comparison(results_by_model: dict[str, dict[str, dict]]) -> str:
    """Primary metric (best in bold) and ECE of the first Choice/Noul question, per model."""
    labels = list(results_by_model)
    rows = defaultdict(list)
    for cat, tasks in CATEGORIES.items():
        for t in tasks:
            prim = {m: results_by_model[m][t]["primary"] for m in labels}
            metric = METRIC_LABELS.get(prim[labels[0]]["metric"], prim[labels[0]]["metric"])
            if prim[labels[0]]["head"] not in ("answer", "*"):
                metric += f" ({prim[labels[0]]['head']})"
            best = max(p["value"] for p in prim.values())
            scores = [f"\\textbf{{{_f(prim[m]['value'])}}}" if prim[m]["value"] == best else _f(prim[m]["value"]) for m in labels]
            eces = [_f(next((h["ece"] for h in results_by_model[m][t]["heads"].values() if "ece" in h), None)) for m in labels]
            rows[cat].append(" & ".join([META[t].name, metric, *scores, *eces]))
    n = len(labels)
    header = (
        f"& & \\multicolumn{{{n}}}{{c}}{{Score$\\uparrow$}} & \\multicolumn{{{n}}}{{c}}{{ECE$\\downarrow$}} \\\\\n"
        f"    \\cmidrule(lr){{3-{2 + n}}} \\cmidrule(lr){{{3 + n}-{2 + 2 * n}}}\n"
        f"    Dataset & Metric & {' & '.join(labels)} & {' & '.join(labels)}"
    )
    return _tabular("l l " + "c " * (2 * n), header, _grouped(2 + 2 * n, rows))
