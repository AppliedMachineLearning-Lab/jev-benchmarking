"""Figures (plotnine + Paul Tol palettes via tol-colors). Each function returns a ggplot."""

import numpy as np
import pandas as pd
import tol_colors as tc
from plotnine import (
    aes,
    annotate,
    coord_equal,
    element_blank,
    element_text,
    facet_wrap,
    geom_abline,
    geom_boxplot,
    geom_jitter,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    guide_legend,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_size_area,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)

from jev_benchmarking.report.meta import META, QUANTITATIVE_SUBJECTS

BRIGHT = list(tc.bright)  # blue, red, green, yellow, cyan, purple, grey
MUTED = list(tc.muted)
N_BINS = 15


def _theme(base_size: float = 9):
    return theme_bw(base_size=base_size) + theme(
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(weight="bold"),
        legend_title=element_text(size=base_size),
    )


def _bins(conf: np.ndarray, hit: np.ndarray) -> pd.DataFrame:
    b = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    df = pd.DataFrame({"bin": b, "conf": conf, "hit": hit})
    return df.groupby("bin").agg(conf=("conf", "mean"), acc=("hit", "mean"), n=("hit", "size")).reset_index()


def reliability(records: dict[str, pd.DataFrame]) -> ggplot:
    """Pooled reliability diagrams. Choice: top-label probability vs. accuracy. Noul: P(yes) vs. observed
    frequency of yes, split into single-question tasks and multi-label fan-out (one Noul per label), since
    the latter (dominated by GoEmotions' 28 rare labels) behaves very differently."""
    from jev_benchmarking.tasks import TASKS

    groups = {"Choice": [], "Noul, single question": [], "Noul, multi-label fan-out": []}
    for t, d in records.items():
        groups["Choice"].append(d[d["kind"] == "choice"])
        target = "Noul, multi-label fan-out" if TASKS[t].multilabel else "Noul, single question"
        groups[target].append(d[d["kind"] == "binary"])
    parts = []
    for label, frames in groups.items():
        df = pd.concat(frames)
        hit = df["correct"] if label == "Choice" else df["gold_num"]
        b = _bins(df["prob"].to_numpy(), hit.to_numpy())
        b["panel"] = f"{label}\n({df['task'].nunique()} tasks, n={len(df):,})"
        parts.append(b)
    df = pd.concat(parts)
    df["panel"] = pd.Categorical(df["panel"], categories=[p["panel"].iloc[0] for p in parts])
    return (
        ggplot(df, aes("conf", "acc"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color=BRIGHT[6])
        + geom_line(color=BRIGHT[0])
        + geom_point(aes(size="n"), color=BRIGHT[0], fill="white", stroke=0.8)
        + facet_wrap("~panel")
        + scale_size_area(max_size=5, guide=None)
        + scale_x_continuous(limits=(0, 1), breaks=[0, 0.5, 1])
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1])
        + coord_equal()
        + labs(x="Predicted probability", y="Observed accuracy / frequency")
        + _theme()
    )


SELECTIVE_TASKS = ("mmlu", "anli", "afrixnli", "banking77", "clinc150", "emotion", "bigbench", "belebele")


def selective(records: dict[str, pd.DataFrame], tasks=SELECTIVE_TASKS) -> ggplot:
    """Accuracy on the most confident fraction of examples, as a function of coverage."""
    rows = []
    for t in tasks:
        d = records[t]
        d = d[d["head"] == "answer"]
        order = np.argsort(-d["confidence"].to_numpy(), kind="stable")
        correct = d["correct"].to_numpy()[order]
        n = len(correct)
        cov = np.arange(1, n + 1) / n
        acc = np.cumsum(correct) / np.arange(1, n + 1)
        keep = np.unique(np.linspace(max(0, int(0.05 * n) - 1), n - 1, 60).astype(int))  # skip the noisy top 5%
        rows.append(pd.DataFrame({"task": META[t].name, "coverage": cov[keep], "accuracy": acc[keep]}))
    df = pd.concat(rows)
    order = df.groupby("task")["accuracy"].last().sort_values(ascending=False).index.tolist()
    df["task"] = pd.Categorical(df["task"], categories=order)
    return (
        ggplot(df, aes("coverage", "accuracy", color="task"))
        + geom_line(size=0.9)
        + scale_color_manual(values=MUTED[: len(tasks)])
        + scale_x_continuous(limits=(0.05, 1), breaks=[0.25, 0.5, 0.75, 1], labels=lambda v: [f"{x:.0%}" for x in v])
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(x="Coverage (most confident fraction answered)", y="Accuracy", color="")
        + _theme()
        + theme(legend_position="right", legend_title=element_blank())
    )


def _script(code: str) -> str:
    script = code.split("_")[-1]
    return "Latin" if script == "Latn" else "Other script"


def multilingual(records: dict[str, pd.DataFrame]) -> ggplot:
    """Per-language accuracy on Belebele (reading comprehension) vs. SIB-200 (topic), both FLORES-based."""
    acc = {}
    for t in ("belebele", "sib200"):
        d = records[t]
        acc[t] = d[d["head"] == "answer"].groupby("subset")["correct"].mean()
    df = pd.DataFrame(acc).dropna().reset_index(names="lang")
    df["script"] = df["lang"].map(_script)
    rho = df[["belebele", "sib200"]].corr(method="spearman").iloc[0, 1]
    labelled = df[df["lang"].isin(["eng_Latn", "yor_Latn", "fuv_Latn", "luo_Latn", "kac_Latn", "bam_Latn"])]
    return (
        ggplot(df, aes("sib200", "belebele", color="script"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color=BRIGHT[6])
        + geom_point(alpha=0.8, size=1.6)
        + geom_text(aes(label="lang"), data=labelled, size=6, nudge_x=0.012, ha="left", show_legend=False, color="black")
        + annotate("text", x=0.32, y=0.97, label=f"Spearman ρ = {rho:.2f}, {len(df)} languages", size=7, ha="left")
        + scale_color_manual(values=[BRIGHT[0], BRIGHT[1]], guide=guide_legend(override_aes={"size": 2.5}))
        + scale_x_continuous(limits=(0.3, 1), labels=lambda v: [f"{x:.0%}" for x in v])
        + scale_y_continuous(limits=(0.3, 1), labels=lambda v: [f"{x:.0%}" for x in v])
        + coord_equal()
        + labs(x="SIB-200 accuracy (topic classification)", y="Belebele accuracy (reading comprehension)", color="")
        + _theme()
        + theme(legend_position="bottom", legend_title=element_blank())
    )


def subjects(records: dict[str, pd.DataFrame]) -> ggplot:
    """Per-subject accuracy on MMLU and C-Eval, split into calculation-heavy vs. other subjects."""
    rows = []
    for t in ("mmlu", "ceval"):
        d = records[t]
        per = d[d["head"] == "answer"].groupby("subset")["correct"].mean().rename_axis("subject").reset_index()
        per["benchmark"] = META[t].name
        per["type"] = np.where(per["subject"].isin(QUANTITATIVE_SUBJECTS[t]), "Calculation-heavy", "Other")
        rows.append(per)
    df = pd.concat(rows)
    df["type"] = pd.Categorical(df["type"], categories=["Other", "Calculation-heavy"])
    return (
        ggplot(df, aes("type", "correct", fill="type"))
        + geom_boxplot(width=0.5, outlier_shape="", alpha=0.35, show_legend=False)
        + geom_jitter(aes(color="type"), width=0.12, height=0, size=1.4, alpha=0.9, show_legend=False)
        + facet_wrap("~benchmark")
        + scale_fill_manual(values=[BRIGHT[0], BRIGHT[1]])
        + scale_color_manual(values=[BRIGHT[0], BRIGHT[1]])
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(x="", y="Per-subject accuracy")
        + _theme()
    )


def _reliability_panels(records: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from jev_benchmarking.tasks import TASKS

    groups = {"Choice": [], "Noul, single question": [], "Noul, multi-label fan-out": []}
    for t, d in records.items():
        groups["Choice"].append(d[d["kind"] == "choice"])
        groups["Noul, multi-label fan-out" if TASKS[t].multilabel else "Noul, single question"].append(d[d["kind"] == "binary"])
    parts = []
    for label, frames in groups.items():
        df = pd.concat(frames)
        hit = df["correct"] if label == "Choice" else df["gold_num"]
        b = _bins(df["prob"].to_numpy(), hit.to_numpy())
        b["panel"] = label
        parts.append(b)
    return pd.concat(parts)


def reliability_models(records_by_model: dict[str, dict[str, pd.DataFrame]]) -> ggplot:
    """Pooled reliability diagrams per model (same panels as `reliability`), one colour per model."""
    df = pd.concat([_reliability_panels(recs).assign(model=m) for m, recs in records_by_model.items()])
    df["panel"] = pd.Categorical(df["panel"], categories=["Choice", "Noul, single question", "Noul, multi-label fan-out"])
    df["model"] = pd.Categorical(df["model"], categories=list(records_by_model))
    return (
        ggplot(df, aes("conf", "acc", color="model"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color=BRIGHT[6])
        + geom_line()
        + geom_point(aes(size="n"), fill="white", stroke=0.7)
        + facet_wrap("~panel")
        + scale_color_manual(values=BRIGHT[: len(records_by_model)])
        + scale_size_area(max_size=4, guide=None)
        + scale_x_continuous(limits=(0, 1), breaks=[0, 0.5, 1])
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1])
        + coord_equal()
        + labs(x="Predicted probability", y="Observed accuracy / frequency", color="")
        + _theme()
        + theme(legend_position="bottom", legend_title=element_blank())
    )


COMPARE_SELECTIVE_TASKS = ("banking77", "anli", "mmlu", "belebele")


def selective_models(records_by_model: dict[str, dict[str, pd.DataFrame]], tasks=COMPARE_SELECTIVE_TASKS) -> ggplot:
    """Accuracy vs. coverage (ranked by each model's own confidence), one panel per dataset."""
    rows = []
    for m, recs in records_by_model.items():
        for t in tasks:
            d = recs[t]
            d = d[d["head"] == "answer"]
            correct = d["correct"].to_numpy()[np.argsort(-d["confidence"].to_numpy(), kind="stable")]
            n = len(correct)
            keep = np.unique(np.linspace(max(0, int(0.05 * n) - 1), n - 1, 60).astype(int))
            acc = np.cumsum(correct) / np.arange(1, n + 1)
            rows.append(pd.DataFrame({"model": m, "task": META[t].name, "coverage": (keep + 1) / n, "accuracy": acc[keep]}))
    df = pd.concat(rows)
    df["model"] = pd.Categorical(df["model"], categories=list(records_by_model))
    df["task"] = pd.Categorical(df["task"], categories=[META[t].name for t in tasks])
    return (
        ggplot(df, aes("coverage", "accuracy", color="model"))
        + geom_line(size=0.9)
        + facet_wrap("~task", nrow=1)
        + scale_color_manual(values=BRIGHT[: len(records_by_model)])
        + scale_x_continuous(breaks=[0.25, 0.5, 0.75, 1], labels=lambda v: [f"{x:.0%}" for x in v])
        + scale_y_continuous(labels=lambda v: [f"{x:.0%}" for x in v])
        + labs(x="Coverage (most confident fraction answered)", y="Accuracy", color="")
        + _theme()
        + theme(legend_position="bottom", legend_title=element_blank())
    )
