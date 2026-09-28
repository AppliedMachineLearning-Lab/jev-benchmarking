import math
import os

import pytest

from jev_benchmarking.openmodel.prompts import choice_codes, confidence, render, to_answer
from jev_benchmarking.tasks.base import choice, noul, score


def test_choice_render_named_and_anonymous_keys():
    named = render({"review": "great"}, choice("Sentiment of `review`?", {"negative": None, "positive": "Upbeat."}))
    assert named.codes == ["A", "B"] and named.keys == ["negative", "positive"]
    assert "A: negative\nB: positive (Upbeat.)" in named.text
    mc = render({"question": "q"}, choice("Which?", {"1": "first", "2": "second"}))
    assert "A: first\nB: second" in mc.text and mc.keys == ["1", "2"]


def test_two_letter_codes_skip_multi_token():
    codes = choice_codes(30, is_single_token=lambda c: c != "AC")
    assert codes[:3] == ["AA", "AB", "AD"] and len(codes) == 30


def test_noul_and_score_render():
    n = render("text", noul("Is it spam?", true="Unsolicited ads.", false="Normal message."))
    assert n.codes == ["Yes", "No"] and "Yes means: Unsolicited ads." in n.text
    s = render("text", score({"criterion": "x", "question": "How well?"}, ["bad", "ok", "good"]))
    assert s.codes == ["0", "1", "2"] and '"criterion": "x"' in s.text


def test_to_answer_formats_match_jev():
    q = choice("Which?", {"a": None, "b": None, "c": None})
    rq = render("s", q)
    a = to_answer(rq, [0.2, 0.7, 0.1], q)
    assert a["choice"] == "b" and a["confidence"] == pytest.approx((3 * 0.7 - 1) / 2)
    sq = score("How?", ["x", "y", "z"])
    s = to_answer(render("s", sq), [0.0, 0.5, 0.5], sq)
    assert s["score"] == pytest.approx(1.5) and s["probabilities"] == {"0": 0.0, "1": 0.5, "2": 0.5}
    assert confidence([0.25] * 4) == 0.0 and confidence([1, 0, 0, 0]) == 1.0


@pytest.mark.skipif(os.environ.get("JEV_TEST_OPENMODEL") != "1", reason="downloads a tiny model; set JEV_TEST_OPENMODEL=1")
@pytest.mark.parametrize(
    "model_id",
    [
        "trl-internal-testing/tiny-Gemma4ForConditionalGeneration",
        # Hybrid linear-attention (recurrent) + full-attention layers, like Qwen3.8-27B: left padding must not leak
        # into the recurrent state.
        "trl-internal-testing/tiny-Qwen3_5ForConditionalGeneration",
    ],
)
def test_batched_scoring_equals_single_prompt_scoring(model_id):
    from jev_benchmarking.openmodel.scorer import HFScorer

    scorer = HFScorer(model_id, device="cpu", dtype="float32", max_batch_tokens=10_000)
    texts = ["Short question?", "A much longer question " * 20, "Medium length question about something " * 3]
    prompts = [scorer.prompt_ids(t) for t in texts]
    codes = [["A", "B", "C"], ["Yes", "No"], ["0", "1", "2", "3"]]
    batched = scorer.score(prompts, codes)
    single = [scorer.score([p], [c])[0] for p, c in zip(prompts, codes)]
    for b, s in zip(batched, single):
        assert b.logprobs == pytest.approx(s.logprobs, abs=1e-4)
        assert all(lp <= 0 and math.isfinite(lp) for lp in b.logprobs)


def test_each_model_gets_its_own_database(tmp_path):
    import json
    import subprocess
    import sys

    from jev_benchmarking.config import MODEL, cache_db_path

    assert cache_db_path(MODEL).name == "responses.db"
    assert cache_db_path("hf:google/gemma-4-E4B-it").name == "responses.google__gemma-4-E4B-it.db"

    jsonl = tmp_path / "run.jsonl"
    row = {"key": "k1", "task": "boolq", "uid": "0", "latency_s": 0.1,
           "response": {"model": "hf:org/tiny", "answers": {"answer": {"type": "noul", "noul": 0.7}},
                        "usage": {"input_tokens": 50, "output_tokens": 0}}}
    jsonl.write_text(json.dumps(row) + "\n")
    env = {**os.environ, "JEV_CACHE_DB": str(tmp_path / "responses.db")}
    subprocess.run([sys.executable, "scripts/import_responses.py", str(jsonl)], check=True, env=env, capture_output=True)
    assert (tmp_path / "responses.org__tiny.db").is_file()
    assert not (tmp_path / "responses.db").exists()  # the Jev database is never created or touched

    row["response"]["model"] = "jev-1.13.0"
    jsonl.write_text(json.dumps(row) + "\n")
    r = subprocess.run([sys.executable, "scripts/import_responses.py", str(jsonl)], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "refusing" in r.stderr
