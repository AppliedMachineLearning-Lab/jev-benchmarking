"""Ordinal / rubric tasks exercising the Score primitive (gold values are mapped to level indices)."""

import math

import numpy as np

from jev_benchmarking.tasks.base import Primary, Task, ex, score


def _mean_over_heads(heads: dict[str, dict], metric: str) -> float:
    vals = [m[metric] for m in heads.values() if not math.isnan(m.get(metric, math.nan))]
    return float(np.mean(vals)) if vals else math.nan


class STSB(Task):
    name = "stsb"
    hf_path = "sentence-transformers/stsb"
    heads = {"answer": "score"}
    primary = Primary("answer", "spearman")
    description = "Semantic textual similarity on a 0-5 scale."
    # SemEval STS annotation guidelines.
    levels = [
        "The two sentences are completely dissimilar.",
        "The two sentences are not equivalent, but are on the same topic.",
        "The two sentences are not equivalent, but share some details.",
        "The two sentences are roughly equivalent, but some important information differs or is missing.",
        "The two sentences are mostly equivalent, but some unimportant details differ.",
        "The two sentences are completely equivalent, as they mean the same thing.",
    ]

    def build(self, row, config):
        q = score("How similar in meaning are `sentence_1` and `sentence_2`?", self.levels)
        # This copy stores the score normalized to 0-1; the original scale is 0-5.
        return ex({"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]}, {"answer": q}, {"answer": row["score"] * 5})


class SST5(Task):
    name = "sst5"
    hf_path = "SetFit/sst5"
    heads = {"answer": "score"}
    primary = Primary("answer", "argmax_accuracy")
    description = "Fine-grained sentence sentiment, 5 ordered levels."
    levels = ["Very negative.", "Negative.", "Neutral.", "Positive.", "Very positive."]

    def build(self, row, config):
        q = score("What is the sentiment of `sentence`?", self.levels)
        return ex({"sentence": row["text"]}, {"answer": q}, {"answer": row["label"]})


class SummEval(Task):
    name = "summeval"
    hf_path = "mteb/summeval"
    splits = {"eval": "test", "dev": None}
    primary = Primary("*", "mean_group_spearman")
    description = "Summary quality judging (LLM-as-judge): 4 expert-rated dimensions, 1-5, one Score each."
    # Dimension definitions follow G-Eval (Liu et al., 2023).
    dimensions = {
        "coherence": "Coherence: the summary is well-structured and well-organized, building from sentence to sentence into a coherent body of information about the topic.",
        "consistency": "Consistency: the summary is factually aligned with `source_document` and contains only statements entailed by it, with no hallucinated facts.",
        "fluency": "Fluency: the individual sentences of the summary are well-written, with correct grammar, spelling, punctuation, word choice and sentence structure.",
        "relevance": "Relevance: the summary includes only the important information from `source_document` and no redundancies or excess information.",
    }  # fmt: skip
    heads = {d: "score" for d in dimensions}
    levels = ["Very poor.", "Poor.", "Fair.", "Good.", "Excellent."]

    def build(self, row, config):
        out = []
        for j, summary in enumerate(row["machine_summaries"]):
            qs = {
                d: score({"criterion": definition, "question": "How well does `summary` meet `criterion`?"}, self.levels)
                for d, definition in self.dimensions.items()
            }
            gold = {d: row[d][j] - 1 for d in self.dimensions}  # mean expert rating, 1-5 -> 0-4
            out.append(ex({"source_document": row["text"], "summary": summary}, qs, gold, group=row["id"]))
        return out

    def extra_metrics(self, exs, answers, heads):
        return {
            "mean_group_spearman": _mean_over_heads(heads, "group_spearman"),
            "mean_group_kendall": _mean_over_heads(heads, "group_kendall"),
            "mean_spearman": _mean_over_heads(heads, "spearman"),
        }


class HelpSteer2(Task):
    name = "helpsteer2"
    hf_path = "nvidia/HelpSteer2"
    splits = {"eval": "validation", "dev": "train"}
    primary = Primary("*", "mean_spearman")
    description = "Response-quality judging: 5 human-rated attributes, 0-4, one Score each."
    # Attribute definitions from the HelpSteer2 paper (Wang et al., 2024).
    attributes = {
        "helpfulness": ("Overall helpfulness of `response` to `prompt`.", ["Not helpful at all.", "Slightly helpful.", "Moderately helpful.", "Very helpful.", "Extremely helpful."]),
        "correctness": ("Inclusion of all pertinent facts in `response`, without errors.", ["Mostly incorrect or missing key facts.", "Several errors or omissions.", "Some errors or omissions.", "Minor errors or omissions.", "Fully correct and complete."]),
        "coherence": ("Consistency and clarity of expression in `response`.", ["Incoherent.", "Often unclear or inconsistent.", "Somewhat clear.", "Mostly clear and consistent.", "Perfectly clear and consistent."]),
        "complexity": ("Intellectual depth required to write `response`.", ["Basic: anyone who speaks the language could write it.", "Simple: needs only everyday knowledge.", "Intermediate: needs some education in the topic.", "Advanced: needs expertise in the field.", "Expert: needs deep domain expertise."]),
        "verbosity": ("Amount of detail in `response`, relative to what `prompt` asks for.", ["Very succinct.", "Succinct.", "Moderate length.", "Verbose.", "Very verbose."]),
    }  # fmt: skip
    heads = {a: "score" for a in attributes}

    def build(self, row, config):
        qs = {a: score(instruction, levels) for a, (instruction, levels) in self.attributes.items()}
        return ex({"prompt": row["prompt"], "response": row["response"]}, qs, {a: row[a] for a in self.attributes})

    def extra_metrics(self, exs, answers, heads):
        return {"mean_spearman": _mean_over_heads(heads, "spearman")}
