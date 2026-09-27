"""Contamination probes on the calculation-heavy MMLU / C-Eval subjects. Not part of the benchmark suite.

- choices-only: the question is withheld; only the subject and the four options are shown. Accuracy far
  above chance means the answer can be inferred from the options alone, through memorization or
  answer-option artifacts. C-Eval serves as the control for artifacts.
- shuffled: the original question with the options rotated, so every option moves to a different letter.
  A drop versus the original order would indicate memorized answer positions.
"""

import hashlib

from jev_benchmarking.tasks.base import Task, choice, ex, hub_config_names, mc_criteria, option_keys

# Subjects whose items mostly require calculation (Jev's documented numeric weak spot).
QUANTITATIVE_SUBJECTS = {
    "mmlu": {
        "abstract_algebra", "college_mathematics", "elementary_mathematics", "high_school_mathematics",
        "high_school_statistics", "college_physics", "high_school_physics", "college_chemistry",
        "high_school_chemistry", "econometrics", "formal_logic", "conceptual_physics", "machine_learning",
    },
    "ceval": {
        "advanced_mathematics", "discrete_mathematics", "probability_and_statistics", "college_physics",
        "college_chemistry", "high_school_mathematics", "high_school_physics", "high_school_chemistry",
        "middle_school_mathematics", "middle_school_physics", "middle_school_chemistry", "accountant",
        "tax_accountant", "electrical_engineer", "metrology_engineer",
    },
}  # fmt: skip

CHOICES_ONLY_INSTRUCTION = (
    "An exam question in `subject` has exactly one correct answer among the options, but the question "
    "itself is not shown. Which option is most likely the correct answer?"
)


def _rotation(uid: str) -> int:
    """1-3: rotate by at least one so that every option changes its letter."""
    return 1 + int(hashlib.sha256(uid.encode()).hexdigest(), 16) % 3


class _QuantMMLU(Task):
    hf_path = "tasksource/mmlu"

    def list_configs(self):
        return [c for c in hub_config_names(self.hf_path) if c in QUANTITATIVE_SUBJECTS["mmlu"]]


class MMLUChoicesOnly(_QuantMMLU):
    name = "probe_mmlu_choices_only"
    description = "MMLU calculation-heavy subjects, question withheld."

    def build(self, row, config):
        q = choice(CHOICES_ONLY_INSTRUCTION, mc_criteria(row["choices"]))
        return ex({"subject": config.replace("_", " ")}, {"answer": q}, {"answer": option_keys(4)[row["answer"]]})


class MMLUShuffled(_QuantMMLU):
    name = "probe_mmlu_shuffled"
    description = "MMLU calculation-heavy subjects, options rotated so each changes its letter."

    def build(self, row, config):
        # The row index is not available here, so derive the rotation from the content itself.
        k = _rotation(row["question"] + "".join(row["choices"]))
        options = row["choices"][k:] + row["choices"][:k]
        gold = option_keys(4)[(row["answer"] - k) % 4]
        q = choice("Which option correctly answers `question`?", mc_criteria(options))
        return ex({"subject": config.replace("_", " "), "question": row["question"]}, {"answer": q}, {"answer": gold})


class CEvalChoicesOnly(Task):
    name = "probe_ceval_choices_only"
    hf_path = "ceval/ceval-exam"
    splits = {"eval": "test", "dev": "val"}
    description = "C-Eval calculation-heavy subjects, question withheld (control for option artifacts)."

    def list_configs(self):
        return [c for c in hub_config_names(self.hf_path) if c in QUANTITATIVE_SUBJECTS["ceval"]]

    def build(self, row, config):
        if row["answer"] not in ("A", "B", "C", "D"):
            return None
        q = choice(CHOICES_ONLY_INSTRUCTION, {k: row[k] for k in "ABCD"})
        return ex({"subject": config.replace("_", " ")}, {"answer": q}, {"answer": row["answer"]})


PROBES = {cls.name: cls() for cls in (MMLUChoicesOnly, MMLUShuffled, CEvalChoicesOnly)}
