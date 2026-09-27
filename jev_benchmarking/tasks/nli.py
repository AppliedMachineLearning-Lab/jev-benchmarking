"""Natural language inference, paraphrase and grounding (claim-vs-document) tasks."""

from collections import defaultdict

import numpy as np
from datasets import load_dataset

from jev_benchmarking.tasks.base import Primary, Task, choice, ex, hub_config_names, noul

NLI_LABELS = ("entailment", "neutral", "contradiction")  # XNLI/MNLI/ANLI label order
NLI_CRITERIA = {
    "entailment": "`hypothesis` is definitely true given `premise`.",
    "neutral": "`hypothesis` may or may not be true; `premise` does not settle it.",
    "contradiction": "`hypothesis` is definitely false given `premise`.",
}


def nli_example(premise: str, hypothesis: str, label: int):
    q = choice("What is the relationship between `premise` and `hypothesis`?", NLI_CRITERIA)
    return ex({"premise": premise, "hypothesis": hypothesis}, {"answer": q}, {"answer": NLI_LABELS[label]})


class ANLI(Task):
    name = "anli"
    hf_path = "facebook/anli"
    configs = ("r1", "r2", "r3")  # rounds; each round is its own split in the HF dataset
    splits = {"eval": "test", "dev": "dev"}
    description = "Adversarial NLI, rounds 1-3, 3 classes."

    def load_split(self, config, split):
        return load_dataset(self.hf_path, "plain_text", split=f"{split}_{config}")

    def build(self, row, config):
        return nli_example(row["premise"], row["hypothesis"], row["label"])


class AfriXNLI(Task):
    name = "afrixnli"
    hf_path = "masakhane/afrixnli"
    description = "NLI in 16 African languages + English/French (IrokoBench), 3 classes."

    def list_configs(self):
        return hub_config_names(self.hf_path)

    def build(self, row, config):
        return nli_example(row["premise"], row["hypothesis"], row["label"])


class PAWS(Task):
    name = "paws"
    hf_path = "google-research-datasets/paws"
    configs = ("labeled_final",)
    heads = {"answer": "binary"}
    description = "Adversarial paraphrase identification (high word overlap)."

    def build(self, row, config):
        q = noul(
            "Do `sentence_1` and `sentence_2` mean the same thing?",
            true="They are paraphrases: the same meaning, possibly with different wording or word order.",
            false="Their meanings differ, even if they share most of their words.",
        )
        return ex({"sentence_1": row["sentence1"], "sentence_2": row["sentence2"]}, {"answer": q}, {"answer": row["label"] == 1})


class LLMAggreFact(Task):
    name = "llm_aggrefact"
    hf_path = "lytang/LLM-AggreFact"  # gated (auto-approve): accept the terms on HF with the logged-in account
    splits = {"eval": "test", "dev": "dev"}
    heads = {"answer": "binary"}
    primary = Primary("*", "mean_balanced_accuracy")
    description = "Grounding / hallucination detection: is the claim supported by the document? 11 sources."

    def build(self, row, config):
        q = noul(
            "Is every part of `claim` supported by `document`?",
            true="Everything the claim states is supported by the document.",
            false="Some part of the claim is unsupported by, or contradicts, the document.",
        )
        state = {"document": row["doc"], "claim": row["claim"]}
        return ex(state, {"answer": q}, {"answer": row["label"] == 1}, subset=row["dataset"])

    def extra_metrics(self, exs, answers, heads):
        # Leaderboard metric: balanced accuracy per source dataset, then the unweighted mean.
        by_source = defaultdict(list)
        for e, a in zip(exs, answers):
            by_source[e.subset].append((e.gold["answer"], a["answer"]["noul"] >= 0.5))
        baccs = []
        for pairs in by_source.values():
            y = np.array([g for g, _ in pairs])
            p = np.array([q for _, q in pairs])
            if y.all() or not y.any():
                continue
            baccs.append(((p[y]).mean() + (~p[~y]).mean()) / 2)
        return {"mean_balanced_accuracy": float(np.mean(baccs)) if baccs else float("nan")}
