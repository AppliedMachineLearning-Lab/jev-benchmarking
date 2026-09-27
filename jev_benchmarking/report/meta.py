"""Display metadata for the paper: names, citation keys, categories (order = order in tables)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetMeta:
    name: str  # display name
    cite: str  # bib key(s), comma-separated
    primitive: str  # as used in our requests


CATEGORIES: dict[str, dict[str, DatasetMeta]] = {
    "Text classification": {
        "ag_news": DatasetMeta("AG News", "zhang2015agnews", "Choice (4)"),
        "imdb": DatasetMeta("IMDB", "maas2011imdb", "Choice (2)"),
        "rotten_tomatoes": DatasetMeta("Rotten Tomatoes", "pang2005rt", "Choice (2)"),
        "sst2": DatasetMeta("SST-2", "socher2013sst,wang2019glue", "Choice (2)"),
        "emotion": DatasetMeta("Emotion", "saravia2018emotion", "Choice (6)"),
        "financial_phrasebank": DatasetMeta("Financial PhraseBank", "malo2014fpb", "Choice (3)"),
        "sms_spam": DatasetMeta("SMS Spam", "almeida2011sms", "Noul"),
        "language_id": DatasetMeta("Language ID", "papluca2021langid", "Choice (20)"),
        "go_emotions": DatasetMeta("GoEmotions", "demszky2020goemotions", "Noul $\\times$28"),
    },
    "Intent and topic routing": {
        "banking77": DatasetMeta("Banking77", "casanueva2020banking77", "Choice (77)"),
        "clinc150": DatasetMeta("CLINC150", "larson2019clinc", "Choice (151)"),
        "sib200": DatasetMeta("SIB-200", "adelani2024sib200", "Choice (7)"),
    },
    "NLI, paraphrase and grounding": {
        "anli": DatasetMeta("ANLI", "nie2020anli", "Choice (3)"),
        "afrixnli": DatasetMeta("AfriXNLI", "adelani2025irokobench", "Choice (3)"),
        "paws": DatasetMeta("PAWS", "zhang2019paws", "Noul"),
        "llm_aggrefact": DatasetMeta("LLM-AggreFact", "tang2024minicheck", "Noul"),
    },
    "Reading comprehension and knowledge": {
        "boolq": DatasetMeta("BoolQ", "clark2019boolq", "Noul"),
        "belebele": DatasetMeta("Belebele", "bandarkar2024belebele", "Choice (4)"),
        "pubmedqa": DatasetMeta("PubMedQA", "jin2019pubmedqa", "Choice (3)"),
        "mmlu": DatasetMeta("MMLU", "hendrycks2021mmlu", "Choice (4)"),
        "ceval": DatasetMeta("C-Eval", "huang2023ceval", "Choice (4)"),
    },
    "Commonsense and reasoning": {
        "bigbench": DatasetMeta("BIG-bench (MC)", "srivastava2023bigbench", "Choice (2--118)"),
        "hellaswag": DatasetMeta("HellaSwag", "zellers2019hellaswag", "Choice (4)"),
        "winogrande": DatasetMeta("WinoGrande", "sakaguchi2020winogrande", "Choice (2)"),
        "arc": DatasetMeta("ARC (E+C)", "clark2018arc", "Choice (3--5)"),
        "commonsense_qa": DatasetMeta("CommonsenseQA", "talmor2019csqa", "Choice (5)"),
        "art": DatasetMeta("$\\alpha$NLI (ART)", "bhagavatula2020art", "Choice (2)"),
    },
    "Safety, moderation and legal": {
        "toxigen": DatasetMeta("ToxiGen", "hartvigsen2022toxigen", "Noul + Score"),
        "openai_moderation": DatasetMeta("OpenAI Moderation", "markov2023moderation", "Noul $\\times$8"),
        "toxic_chat": DatasetMeta("ToxicChat", "lin2023toxicchat", "Noul $\\times$2"),
        "prompt_injections": DatasetMeta("Prompt Injections", "deepset2023promptinjections", "Noul"),
        "agb_de": DatasetMeta("AGB-DE", "braun2024agbde", "Noul"),
        "unfair_tos": DatasetMeta("UNFAIR-ToS", "lippi2019claudette,chalkidis2022lexglue", "Noul $\\times$8"),
    },
    "Rubric scoring": {
        "stsb": DatasetMeta("STS-B", "cer2017stsb", "Score (6)"),
        "sst5": DatasetMeta("SST-5", "socher2013sst", "Score (5)"),
        "summeval": DatasetMeta("SummEval", "fabbri2021summeval", "Score (5) $\\times$4"),
        "helpsteer2": DatasetMeta("HelpSteer2", "wang2024helpsteer2", "Score (5) $\\times$5"),
    },
}

META = {task: meta for group in CATEGORIES.values() for task, meta in group.items()}
CATEGORY_OF = {task: cat for cat, group in CATEGORIES.items() for task in group}

METRIC_LABELS = {
    "accuracy": "Acc.",
    "f1": "F1",
    "macro_f1": "Macro-F1",
    "micro_f1": "Micro-F1",
    "mean_auprc": "Mean AUPRC",
    "mean_balanced_accuracy": "Bal. Acc.",
    "spearman": "Spearman $\\rho$",
    "argmax_accuracy": "Acc.",
    "mean_group_spearman": "Mean $\\rho$ (summary-level)",
    "mean_spearman": "Mean $\\rho$",
}

# Split names as reported (the HF split we score on).
SPLIT_LABELS = {"test": "test", "validation": "val.", "val": "val.", "train": "all", "dev": "dev"}

# MMLU / C-Eval subjects whose items mostly require calculation (Jev's documented numeric weak spot).
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
