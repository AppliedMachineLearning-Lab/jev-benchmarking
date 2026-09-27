"""Single-label text classification (Choice) and multi-label emotion (Noul fan-out)."""

from functools import cached_property

from datasets import load_dataset

from jev_benchmarking.tasks.base import Primary, Task, choice, ex, hub_config_names, noul

SENTIMENT_2 = {
    "negative": "The text expresses a negative opinion.",
    "positive": "The text expresses a positive opinion.",
}


class _Sentiment2(Task):
    text_field = "text"
    labels = ("negative", "positive")
    primary = Primary("answer", "accuracy")

    def build(self, row, config):
        q = choice("What is the overall sentiment of `review` toward the movie?", SENTIMENT_2)
        return ex({"review": row[self.text_field].strip()}, {"answer": q}, {"answer": self.labels[row["label"]]})


class IMDB(_Sentiment2):
    name = "imdb"
    hf_path = "stanfordnlp/imdb"
    configs = ("plain_text",)
    splits = {"eval": "test", "dev": "train"}
    description = "Binary movie-review sentiment (long reviews)."


class RottenTomatoes(_Sentiment2):
    name = "rotten_tomatoes"
    hf_path = "cornell-movie-review-data/rotten_tomatoes"
    description = "Binary movie-review sentiment (snippets)."


class SST2(_Sentiment2):
    name = "sst2"
    hf_path = "stanfordnlp/sst2"
    text_field = "sentence"
    splits = {"eval": "validation", "dev": "train"}  # test labels are hidden (-1)
    description = "Binary sentence sentiment (GLUE SST-2)."


class AGNews(Task):
    name = "ag_news"
    hf_path = "fancyzhx/ag_news"
    splits = {"eval": "test", "dev": "train"}
    labels = ("World", "Sports", "Business", "Sci/Tech")
    criteria = {
        "World": "International news, politics, conflicts and world affairs.",
        "Sports": "Sports events, teams and athletes.",
        "Business": "Companies, markets, the economy and finance.",
        "Sci/Tech": "Science, technology, computing, the internet and space.",
    }
    description = "News topic classification, 4 classes."

    def build(self, row, config):
        q = choice("What is the topic of the news `article`?", self.criteria)
        return ex({"article": row["text"]}, {"answer": q}, {"answer": self.labels[row["label"]]})


class Emotion(Task):
    name = "emotion"
    hf_path = "dair-ai/emotion"
    configs = ("split",)
    labels = ("sadness", "joy", "love", "anger", "fear", "surprise")
    description = "Emotion of English tweets, 6 classes."

    def build(self, row, config):
        q = choice("Which emotion does the author of `text` express most strongly?", {l: None for l in self.labels})
        return ex({"text": row["text"]}, {"answer": q}, {"answer": self.labels[row["label"]]})


class Banking77(Task):
    name = "banking77"
    hf_path = "mteb/banking77"
    splits = {"eval": "test", "dev": "train"}
    description = "Banking customer-query intent, 77 classes."

    @cached_property
    def intents(self) -> list[str]:
        return sorted(set(load_dataset(self.hf_path, split="test")["label_text"]))

    def build(self, row, config):
        q = choice("Which intent best describes the bank customer's `query`?", {i: None for i in self.intents})
        return ex({"query": row["text"]}, {"answer": q}, {"answer": row["label_text"]})


class CLINC150(Task):
    name = "clinc150"
    hf_path = "clinc/clinc_oos"
    configs = ("plus",)
    description = "Virtual-assistant intent, 150 intents + out-of-scope."

    @cached_property
    def intents(self) -> list[str]:
        return load_dataset(self.hf_path, "plus", split="test").features["intent"].names

    def build(self, row, config):
        criteria = {i: None for i in self.intents if i != "oos"}
        criteria["oos"] = "Out of scope: the request matches none of the other intents."
        q = choice("Which intent does the user's `utterance` to a virtual assistant express?", criteria)
        return ex({"utterance": row["text"]}, {"answer": q}, {"answer": self.intents[row["intent"]]})

    def extra_metrics(self, exs, answers, heads):
        gold = [e.gold["answer"] for e in exs]
        pred = [a["answer"]["choice"] for a in answers]
        ins = [(g, p) for g, p in zip(gold, pred) if g != "oos"]
        oos_gold = sum(g == "oos" for g in gold)
        oos_pred = sum(p == "oos" for p in pred)
        oos_hit = sum(g == p == "oos" for g, p in zip(gold, pred))
        return {
            "in_scope_accuracy": sum(g == p for g, p in ins) / max(1, len(ins)),
            "oos_recall": oos_hit / max(1, oos_gold),
            "oos_precision": oos_hit / max(1, oos_pred),
        }


class SIB200(Task):
    name = "sib200"
    hf_path = "Davlan/sib200"
    labels = ("science/technology", "travel", "politics", "sports", "health", "entertainment", "geography")
    description = "Topic classification in 205 language varieties, 7 classes."

    def list_configs(self):
        return hub_config_names(self.hf_path)

    def build(self, row, config):
        q = choice("What is the topic of `text`?", {l: None for l in self.labels})
        return ex({"text": row["text"]}, {"answer": q}, {"answer": row["category"]})


class SMSSpam(Task):
    name = "sms_spam"
    hf_path = "ucirvine/sms_spam"
    configs = ("plain_text",)
    splits = {"eval": "train", "dev": None}  # the only split
    heads = {"answer": "binary"}
    primary = Primary("answer", "f1")
    description = "SMS spam detection."

    def build(self, row, config):
        q = noul("Is the SMS `message` spam, such as unsolicited advertising, prize scams or promotional bulk messaging?")
        return ex({"message": row["sms"].strip()}, {"answer": q}, {"answer": row["label"] == 1})


class LanguageID(Task):
    name = "language_id"
    hf_path = "papluca/language-identification"
    languages = {
        "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
        "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
        "pt": "Portuguese", "ru": "Russian", "sw": "Swahili", "th": "Thai", "tr": "Turkish", "ur": "Urdu",
        "vi": "Vietnamese", "zh": "Chinese",
    }  # fmt: skip
    description = "Language identification, 20 languages."

    def build(self, row, config):
        q = choice("In which language is `text` written?", self.languages)
        return ex({"text": row["text"]}, {"answer": q}, {"answer": row["labels"]})


class FinancialPhraseBank(Task):
    name = "financial_phrasebank"
    hf_path = "atrost/financial_phrasebank"
    labels = ("negative", "neutral", "positive")
    criteria = {
        "negative": "The news is likely to have a negative effect on the company's stock price.",
        "neutral": "The news is unlikely to affect the company's stock price.",
        "positive": "The news is likely to have a positive effect on the company's stock price.",
    }
    description = "Financial news sentiment from an investor's view, 3 classes."

    def build(self, row, config):
        q = choice("From an investor's point of view, what is the sentiment of the financial news `sentence`?", self.criteria)
        return ex({"sentence": row["sentence"]}, {"answer": q}, {"answer": self.labels[row["label"]]})


class GoEmotions(Task):
    name = "go_emotions"
    hf_path = "google-research-datasets/go_emotions"
    configs = ("simplified",)
    multilabel = True
    primary = Primary("*", "macro_f1")
    description = "Multi-label emotion of Reddit comments; one Noul per each of 28 labels in a single request."
    labels = (
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
        "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
        "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
    )  # fmt: skip
    heads = {l: "binary" for l in labels}

    def build(self, row, config):
        qs = {
            l: noul(f"Does the author of `comment` express {l}?")
            for l in self.labels
            if l != "neutral"
        }
        qs["neutral"] = noul("Is `comment` emotionally neutral, expressing no particular emotion?")
        gold = {l: i in row["labels"] for i, l in enumerate(self.labels)}
        return ex({"comment": row["text"]}, qs, gold)
