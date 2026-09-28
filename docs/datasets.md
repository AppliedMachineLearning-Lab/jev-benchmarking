# Dataset audit & benchmark list

Status: **final** (audit 2026-09-24, run completed 2026-09-27). Source list: `benchmarks.txt`. All
sizes/splits verified against the Hugging Face Hub + dataset-viewer APIs on 2026-09-24. All 20 original
datasets (with the fixes below) and all 18 proposed additions were run: 37 tasks in total, since SWAG
was replaced by HellaSwag. Results: `results/eval/`; paper: `paper/` (not in git).

## What Jev constrains (from docs.typesafe.ai, jev-1.13)

- Input: text / JSON `state` + typed questions. Output: **Choice** (≤255 options, probs + confidence),
  **Score** (2–10 ordered levels, probs + confidence), **Noul** (P(yes)). No generation.
- Context: 32k tokens for state + longest question, 64k total per request.
- Documented weak spots ("jaggedness"): math/counting/numeric precision, date arithmetic, multi-hop
  indirection, long irrelevant context, adversarial content, generation. English best; other
  languages (esp. CJK) weaker.
- Price: **$0.042 / 1M input tokens, output free**. Rate limit 250k tok/s, **1,200 req/min**
  (limits "adjusting dynamically").

Consequence: cost is negligible (whole suite estimated ≈ $10–30, actual $9.15); the binding constraint is
requests/minute.
One request per example (state = example), all label questions fanned out inside that request.

## Audit of the original 20

| # | Dataset | Eval split (n) | Primitive | Verdict | Notes / fixes |
|---|---|---|---|---|---|
| 1 | facebook/belebele | test, 122 langs × 900 = 109,800 | Choice (4) | ✅ keep | Passage→state. Biggest item (~1.5 h at rate limit, still ~$3). Could subset langs, but not needed for cost. `correct_answer_num` is 1-indexed. |
| 2 | tasksource/bigbench | validation, 167 tasks, 201k rows | Choice | ⚠️ subset | Many tasks are generative or arithmetic (e.g. `arithmetic` 15k rows) = Jev's documented weak spots / not expressible. Keep only tasks with `multiple_choice_targets`, drop math/counting/code tasks, cap ~500/task. **Final:** 93 tasks, 13,228 rows (keyword-based exclusion + 3 date/number tasks; list in `jev_benchmarking/tasks/bigbench_mc_tasks.json`). |
| 3 | google/boolq | validation (3,270) | Noul | ✅ keep | Test has no public labels; validation is standard. |
| 4 | stanfordnlp/imdb | test (25,000) | Choice/Noul | ✅ keep | Long reviews, cheap anyway. Near ceiling for most models. |
| 5 | facebook/anli | test r1–r3 (3,200) | Choice (3) | ✅ keep | Adversarial NLI; expect low absolute numbers (useful honest datapoint). CC-BY-NC. |
| 6 | ceval/ceval-exam | test, 52 subjects (12,342) | Choice (4) | ⚠️ keep, report by category | HF copy has test answers populated (verify against official protocol, which historically hid them). Chinese + many computation-heavy STEM items → hits two documented weak spots. Report STEM vs. humanities separately. CC-BY-NC-SA. |
| 7 | allenai/swag | ~~test~~ validation (20,006) | Choice (4) | 🔁 replace | **Test labels are all `-1`.** SWAG is saturated and superseded by HellaSwag (adversarially filtered, reported in nearly every LLM paper). → use Rowan/hellaswag validation. |
| 8 | fancyzhx/ag_news | test (7,600) | Choice (4) | ✅ keep | |
| 9 | tasksource/mmlu | test, 57 subjects (14,042) | Choice (4) | ✅ keep, report by category | Math/physics subjects hit numeric weakness; report per-category. |
| 10 | cornell-movie-review-data/rotten_tomatoes | test (1,066) | Choice/Noul | ✅ keep | |
| 11 | stanfordnlp/sst2 | ~~test~~ validation (872) | Choice/Noul | ⚠️ fix split | **Test labels are all `-1`.** Use validation (standard practice). |
| 12 | dair-ai/emotion | test (2,000), config `split` | Choice (6) | ✅ keep | URL in list had `/viewer/split/test` suffix. Labels are hashtag-derived (noisy). |
| 13 | mteb/banking77 | test (3,076) | Choice (77) | ✅ keep | Good fit for intent-routing story. |
| 14 | Davlan/sib200 | test, 205 langs × 204 = 41,820 | Choice (7) | ✅ keep | Topic classification, very multilingual. **Final:** config entry `nqo_Nkoo.zip` is a broken duplicate of `nqo_Nkoo` and is skipped; all 205 varieties are evaluated. |
| 15 | allenai/art (αNLI) | validation (1,532) | Choice (2) | ✅ keep | Test labels not released; validation is standard. |
| 16 | toxigen/toxigen-data | `annotated` test (940) | Noul + Score | ✅ keep | `toxicity_human` is a 1–5 float → binarize for Noul (fix threshold up front) **and** evaluate as Score (Spearman). **Final:** toxic = `toxicity_ai + toxicity_human > 5.5` (lm-evaluation-harness convention). |
| 17 | ucirvine/sms_spam | train only (5,574) | Noul | ✅ keep (easy) | Only one split; use all as eval (zero-shot, so fine). 13% spam. Near ceiling. |
| 18 | papluca/language-identification | test (10,000) | Choice (20) | ✅ kept | Valid, but trivial for fastText (~99.5%) and not really a "judgment". Kept as a cheap sanity check (Jev: 99.6%). |
| 19 | d4br4/agb-de | test (755) | Noul | ✅ keep | German T&C clause voidness; **37/755 positive** → report F1(pos)/AUPRC, not accuracy. |
| 20 | mteb/AfriXNLI | test | Choice (3) | 🔁 replace | mteb copy is **binarized** (neutral dropped, 400/lang). Use masakhane/afrixnli (3-class, 18 langs × 600 = 10,800, Apache-2.0) to match IrokoBench published numbers. |

**Gap:** 19/20 are single-label Choice tasks. Nothing exercises **Score**, multi-question **fan-out**,
or Jev's marketed use cases (guardrails, routing with out-of-scope, RAG grounding, LLM-as-judge).

## Proposed additions (18)

Chosen to (a) cover Score and fan-out, (b) match Jev's advertised use cases, (c) have published
baselines we can cite for free, (d) stay small. All parquet/CSV/JSON (no loading scripts), none gated
unless noted.

| Dataset | Eval split (n) | Primitive | Why |
|---|---|---|---|
| Rowan/hellaswag | validation (10,042) | Choice (4) | Replaces SWAG; ubiquitous LLM baseline. |
| allenai/winogrande (`winogrande_xl`) | validation (1,267) | Choice (2) | Commonsense coreference; test labels hidden. |
| allenai/ai2_arc (Easy + Challenge) | test (2,376 + 1,172) | Choice (4) | Standard science QA, huge baseline literature. |
| tau/commonsense_qa | validation (1,221) | Choice (5) | Test labels hidden. |
| sentence-transformers/stsb | test (1,379) | **Score** (0–5) | Canonical ordinal similarity; Spearman. |
| SetFit/sst5 | test (2,210) | **Score** (5) | Fine-grained sentiment as an ordinal scale. |
| mteb/summeval | test (100 docs × 16 summaries = 1,600) | **Score** ×4 (fan-out) | LLM-as-judge benchmark (coherence/consistency/fluency/relevance); G-Eval etc. baselines. |
| nvidia/HelpSteer2 | validation (1,038) | **Score** ×5 (fan-out) | Response-quality judging (helpfulness, correctness, …, 0–4). |
| google-research-datasets/go_emotions (`simplified`) | test (5,427) | **Noul ×28 (fan-out)** | Multi-label; showcases many questions per request. |
| clinc/clinc_oos (`plus`) | test (5,500, incl. 1,000 OOS) | Choice (150 + oos) | Intent routing **with out-of-scope** → tests confidence-gated routing. |
| mmathys/openai-moderation-api-evaluation | all (1,680) | Noul ×8 (fan-out) | Guardrails; multi-category moderation, widely reported. |
| lmsys/toxic-chat (`toxicchat0124`) | test (5,083) | Noul | Real user prompts; Llama Guard & co. report on it. CC-BY-NC. |
| deepset/prompt-injections | test (116) | Noul | Prompt-injection detection (en/de). Tiny → wide CIs; cheap. |
| lytang/LLM-AggreFact | test (~29k across 11 sources) | Noul | Grounding / hallucination check (matches citation-check & RAG cookbooks). **Public leaderboard = free LLM baselines.** Gated (auto-accept, needs HF token). |
| google-research-datasets/paws (`labeled_final`) | test (8,000) | Noul | Adversarial paraphrase detection. |
| qiaojin/PubMedQA (`pqa_labeled`) | all (1,000) | Choice (yes/no/maybe) | Biomedical, abstract as state. |
| atrost/financial_phrasebank | test (970) | Choice (3) | Finance sentiment (parquet mirror; original repo is script-based). |
| coastalcph/lex_glue (`unfair_tos`) | test (1,607) | Noul ×8 (fan-out) | Unfair ToS clauses, pairs with AGB-DE (en vs. de). |

Optional, **not run**: nyu-mll/multi_nli (validation_matched 9,815), truthfulqa/truthful_qa MC1
(817), coastalcph/lex_glue `ledgar` (100-way, 10k), mteb/amazon_massive_intent (multilingual intent),
facebook/xnli, nguha/legalbench (subset).

Considered and rejected: ybisk/piqa, allenai/social_i_qa, allenai/scifact, takala/financial_phrasebank
(loading scripts only, no longer supported by `datasets` ≥ 4); walledai/XSTest, allenai/wildguardmix
(gated, overlapping with the moderation sets above); HaluEval (known artifacts; AggreFact covers it
better); lukaemon/bbh (chain-of-thought / System-Two tasks; could serve as a "limits" appendix).

## Budget: estimate vs. actual

Estimate (before the run): ~365k requests at ~1–2k input tokens each → **$15–30**, ~5 h at 1,200 req/min.

Actual:

| Run | Requests | Input tokens | Cost |
|---|---|---|---|
| Dev pilot (20 per task) | 620 | 0.3M | $0.01 |
| Full eval run (all 37 tasks, full eval splits) | 346,009 | 217.9M | $9.15 |
| Threshold tuning (1,000 dev examples × 4 tasks) | 3,830 | 2.4M | $0.10 |
| Memorization probes (MMLU/C-Eval calculation-heavy subjects) | 7,570 | 3.1M | $0.13 |
| **Total** | 358,029 | 223.7M | **$9.40** |

The eval run took 5 h 15 min at 1,100 req/min (client-side cap); 630 input tokens per request on average,
0.36 s mean latency. One request (a BIG-bench item) exceeded the context limit (HTTP 400
`max_tokens_exceeded`); all others were answered.

## Additional analyses (beyond the suite)

- **Per-question thresholds** for GoEmotions, UNFAIR-ToS, AGB-DE and ToxicChat: tuned for F1 on 1,000
  dev examples, applied unchanged to the eval split (`jev_benchmarking/thresholds.py`,
  `results/eval/thresholds.json`).
- **Memorization probes** on the calculation-heavy MMLU subjects: options rotated, question withheld,
  plus a C-Eval question-withheld control (`jev_benchmarking/tasks/probes.py`,
  `results/eval/probes.json`).

## Methodology notes for the paper (all implemented)

- Freeze one question template per dataset **before** looking at test results; iterate only on
  train/dev splits. Log the exact request JSON and the returned `model` version (`jev-1.13.0`),
  pinned rather than `jev-latest`.
- Report accuracy / macro-F1 (Spearman for Score), plus **calibration** (ECE, Brier) and
  **selective accuracy vs. coverage** using `confidence`, since calibration is Jev's main claim.
- Report latency and $ per 1k examples, which is where Jev should stand out.
- Bootstrap 95% CIs; small sets (prompt-injections, AGB-DE, WinoGrande) need them.
