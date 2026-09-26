"""Multiple-choice QA, commonsense and reading comprehension. Options become Choice criteria (A, B, ...)."""

import json
import re
from pathlib import Path

from datasets import get_dataset_config_names

from jev_benchmarking.tasks.base import Primary, Task, choice, ex, mc_criteria, noul, option_keys

MC_INSTRUCTION = "Which option correctly answers `question`?"


class Belebele(Task):
    name = "belebele"
    hf_path = "facebook/belebele"
    splits = {"eval": "test", "dev": None}
    description = "Multilingual reading comprehension (FLORES passages), 122 language variants, 4 options."

    def list_configs(self):
        return get_dataset_config_names(self.hf_path)

    def build(self, row, config):
        options = [row[f"mc_answer{i}"].strip() for i in range(1, 5)]
        q = choice("Which option correctly answers `question` according to `passage`?", mc_criteria(options))
        gold = option_keys(4)[int(row["correct_answer_num"]) - 1]
        return ex({"passage": row["flores_passage"], "question": row["question"]}, {"answer": q}, {"answer": gold})


# Multiple-choice BIG-bench tasks, minus those tagged mathematics/arithmetic/algebra/proof/code/numerical
# response/non-language/tokenization/visual reasoning/game play/programmatic in BIG-bench's keyword index
# (see docs/datasets.md), minus date/number-heavy tasks Jev's docs list as weak spots.
BIGBENCH_EXCLUDE_MANUAL = {"date_understanding", "penguins_in_a_table", "temporal_sequences"}
BIGBENCH_TASKS = tuple(
    t for t in json.loads((Path(__file__).parent / "bigbench_mc_tasks.json").read_text()) if t not in BIGBENCH_EXCLUDE_MANUAL
)


class BigBench(Task):
    name = "bigbench"
    hf_path = "tasksource/bigbench"
    configs = BIGBENCH_TASKS
    splits = {"eval": "validation", "dev": "train"}
    max_per_config = 500
    description = f"BIG-bench multiple-choice tasks ({len(BIGBENCH_TASKS)} tasks, ≤500 rows each)."

    def build(self, row, config):
        targets = row["multiple_choice_targets"]
        scores = row["multiple_choice_scores"]
        if not targets or len(targets) > 255 or len(set(targets)) != len(targets) or not any(scores):
            return None
        keys = option_keys(len(targets))
        q = choice("Which option is the correct answer to `prompt`?", dict(zip(keys, targets)))
        gold = frozenset(k for k, s in zip(keys, scores) if s)
        return ex({"prompt": row["inputs"].strip()}, {"answer": q}, {"answer": gold})


class MMLU(Task):
    name = "mmlu"
    hf_path = "tasksource/mmlu"
    description = "Knowledge QA over 57 subjects, 4 options."

    def list_configs(self):
        return get_dataset_config_names(self.hf_path)

    def build(self, row, config):
        q = choice(MC_INSTRUCTION, mc_criteria(row["choices"]))
        state = {"subject": config.replace("_", " "), "question": row["question"]}
        return ex(state, {"answer": q}, {"answer": option_keys(4)[row["answer"]]})


class CEval(Task):
    name = "ceval"
    hf_path = "ceval/ceval-exam"
    splits = {"eval": "test", "dev": "val"}
    description = "Chinese exam questions over 52 subjects, 4 options."

    def list_configs(self):
        return get_dataset_config_names(self.hf_path)

    def build(self, row, config):
        if row["answer"] not in ("A", "B", "C", "D"):
            return None
        q = choice(MC_INSTRUCTION, {k: row[k] for k in "ABCD"})
        return ex({"subject": config.replace("_", " "), "question": row["question"]}, {"answer": q}, {"answer": row["answer"]})


def _hellaswag_clean(text: str) -> str:
    # Same cleanup as lm-evaluation-harness: WikiHow rows carry "[title]"-style markup.
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return re.sub(r"\.\.+", ".", text.replace("  ", " ")).strip()


class HellaSwag(Task):
    name = "hellaswag"
    hf_path = "Rowan/hellaswag"
    splits = {"eval": "validation", "dev": "train"}  # test labels are hidden
    description = "Commonsense sentence completion, 4 options."

    def build(self, row, config):
        context = _hellaswag_clean(f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}")
        q = choice("Which ending is the most plausible continuation of `context`?", mc_criteria([_hellaswag_clean(e) for e in row["endings"]]))
        return ex({"context": context}, {"answer": q}, {"answer": option_keys(4)[int(row["label"])]})


class WinoGrande(Task):
    name = "winogrande"
    hf_path = "allenai/winogrande"
    configs = ("winogrande_xl",)
    splits = {"eval": "validation", "dev": "train"}  # test labels are hidden
    description = "Commonsense pronoun/blank resolution, 2 options."

    def build(self, row, config):
        if row["answer"] not in ("1", "2"):
            return None
        q = choice("Which option correctly fills the blank `_` in `sentence`?", mc_criteria([row["option1"], row["option2"]]))
        return ex({"sentence": row["sentence"]}, {"answer": q}, {"answer": option_keys(2)[int(row["answer"]) - 1]})


class _LabeledChoices(Task):
    """Datasets whose options come with their own labels (A-E or 1-4) and an `answerKey`."""

    def build(self, row, config):
        criteria = dict(zip(row["choices"]["label"], row["choices"]["text"]))
        if row["answerKey"] not in criteria:
            return None
        return ex({"question": row["question"]}, {"answer": choice(MC_INSTRUCTION, criteria)}, {"answer": row["answerKey"]})


class ARC(_LabeledChoices):
    name = "arc"
    hf_path = "allenai/ai2_arc"
    configs = ("ARC-Challenge", "ARC-Easy")
    description = "Grade-school science QA (Challenge + Easy), 3-5 options."


class CommonsenseQA(_LabeledChoices):
    name = "commonsense_qa"
    hf_path = "tau/commonsense_qa"
    splits = {"eval": "validation", "dev": "train"}  # test labels are hidden
    description = "Commonsense QA, 5 options."


class AlphaNLI(Task):
    name = "art"
    hf_path = "allenai/art"
    configs = ("anli",)
    splits = {"eval": "validation", "dev": "train"}  # test labels were never released
    description = "Abductive NLI (αNLI): pick the more plausible explanation, 2 options."

    def build(self, row, config):
        if row["label"] not in (1, 2):
            return None
        q = choice(
            "Which hypothesis best explains what happened between `observation_1` and `observation_2`?",
            mc_criteria([row["hypothesis_1"], row["hypothesis_2"]]),
        )
        state = {"observation_1": row["observation_1"], "observation_2": row["observation_2"]}
        return ex(state, {"answer": q}, {"answer": option_keys(2)[row["label"] - 1]})


class BoolQ(Task):
    name = "boolq"
    hf_path = "google/boolq"
    splits = {"eval": "validation", "dev": "train"}
    heads = {"answer": "binary"}
    primary = Primary("answer", "accuracy")
    description = "Yes/no questions over Wikipedia passages."

    def build(self, row, config):
        question = row["question"].strip()
        question = question[0].upper() + question[1:] + ("" if question.endswith("?") else "?")
        q = noul("According to `passage`, is the answer to `question` yes?")
        return ex({"passage": row["passage"], "question": question}, {"answer": q}, {"answer": bool(row["answer"])})


class PubMedQA(Task):
    name = "pubmedqa"
    hf_path = "qiaojin/PubMedQA"
    configs = ("pqa_labeled",)
    splits = {"eval": "train", "dev": None}  # the 1,000 expert-labelled questions
    description = "Biomedical yes/no/maybe questions answered from a PubMed abstract."
    criteria = {
        "yes": "The abstract's results support answering yes.",
        "no": "The abstract's results support answering no.",
        "maybe": "The abstract's results are mixed or inconclusive.",
    }

    def build(self, row, config):
        abstract = "\n".join(row["context"]["contexts"])
        q = choice("Based on `abstract`, what is the answer to `question`?", self.criteria)
        return ex({"question": row["question"], "abstract": abstract}, {"answer": q}, {"answer": row["final_decision"]})
