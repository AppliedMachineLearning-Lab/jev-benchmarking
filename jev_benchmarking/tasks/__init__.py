"""Task registry. `TASKS` maps task name -> Task instance, in the order used for reports."""

from jev_benchmarking.tasks import classification, multiple_choice, nli, safety, scoring
from jev_benchmarking.tasks.base import Example, Task
from jev_benchmarking.tasks.probes import PROBES

_CLASSES = [
    # classification
    classification.AGNews, classification.IMDB, classification.RottenTomatoes, classification.SST2,
    classification.Emotion, classification.FinancialPhraseBank, classification.Banking77, classification.CLINC150,
    classification.SIB200, classification.LanguageID, classification.SMSSpam, classification.GoEmotions,
    # NLI, paraphrase, grounding
    nli.ANLI, nli.AfriXNLI, nli.PAWS, nli.LLMAggreFact,
    # multiple choice, commonsense, reading comprehension
    multiple_choice.BoolQ, multiple_choice.Belebele, multiple_choice.PubMedQA, multiple_choice.MMLU,
    multiple_choice.CEval, multiple_choice.BigBench, multiple_choice.HellaSwag, multiple_choice.WinoGrande,
    multiple_choice.ARC, multiple_choice.CommonsenseQA, multiple_choice.AlphaNLI,
    # guardrails and legal
    safety.ToxiGen, safety.OpenAIModeration, safety.ToxicChat, safety.PromptInjections, safety.AGBDE,
    safety.UnfairToS,
    # Score primitive
    scoring.STSB, scoring.SST5, scoring.SummEval, scoring.HelpSteer2,
]  # fmt: skip

TASKS: dict[str, Task] = {cls.name: cls() for cls in _CLASSES}
assert len(TASKS) == len(_CLASSES), "duplicate task name"


# Everything addressable by name. "all" means the benchmark suite only, never the probes.
ALL_TASKS: dict[str, Task] = {**TASKS, **PROBES}


def get_tasks(names: list[str] | None) -> list[Task]:
    if not names or names == ["all"]:
        return list(TASKS.values())
    if names == ["probes"]:
        return list(PROBES.values())
    unknown = [n for n in names if n not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"Unknown task(s): {', '.join(unknown)}. Known: {', '.join(ALL_TASKS)}")
    return [ALL_TASKS[n] for n in names]


__all__ = ["ALL_TASKS", "PROBES", "TASKS", "Example", "Task", "get_tasks"]
