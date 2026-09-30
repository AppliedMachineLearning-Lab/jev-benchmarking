# Evaluating and Benchmarking the System One Model Jev

Code for the paper **[Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/abs/2609.37647)**
(Tobias Deußer, Lorenz Sparrenberg, Rafet Sifa; arXiv:2609.37647, 2026).

- **Paper:** <https://arxiv.org/abs/2609.37647>
- **Raw responses (dataset):** <https://doi.org/10.5281/zenodo.23039006>

The repository benchmarks TypeSafe's [Jev](https://docs.typesafe.ai/introduction) (pinned to `jev-1.13.0`)
zero-shot on 37 public datasets, and compares it with two open-weight LLMs (Qwen3.8-27B, Gemma-4-E4B) scored on
identical requests via their exact option probabilities. It contains the dataset definitions and question
templates, the evaluation harness, the analysis code for every table and figure, and the per-dataset results.
The dataset audit, the selection rationale and the actual API cost are documented in
[docs/datasets.md](docs/datasets.md).

## Installation

Requires Python 3.10 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/AppliedMachineLearning-Lab/jev-benchmarking.git
cd jev-benchmarking
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,paper]"
```

or with plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,paper]"
```

Optional dependency groups: `dev` (tests), `paper` (tables and figures), `openmodel` (scoring open-weight models
with PyTorch and `transformers`; needs a GPU for real models).

Datasets are downloaded from the Hugging Face Hub on first use. `lytang/LLM-AggreFact` is gated: accept its terms
on Hugging Face and log in (`hf auth login`) before using it.

## Reproducing the results without API access

All responses from the paper are published on Zenodo, so every number can be recomputed offline:

```bash
mkdir -p cache
# download responses.db, responses.google__gemma-4-E4B-it.db and responses.Qwen__Qwen3.8-27B.db
# from https://doi.org/10.5281/zenodo.23039006 into cache/
python scripts/evaluate.py all --split eval                                     # Jev -> results/eval/
python scripts/evaluate.py all --split eval --model hf:Qwen/Qwen3.8-27B        # -> results/eval/open_models/
python scripts/evaluate.py all --split eval --model hf:google/gemma-4-E4B-it
python scripts/make_paper_assets.py                                             # tables and figures -> paper/
pytest                                                                          # offline tests
```

Responses are keyed by a hash of the exact request. The code rebuilds each request from the Hugging Face
datasets and looks up its response, so a dataset that changes upstream would no longer match its cached
responses.

## Running Jev yourself

Set your TypeSafe API key in `TYPESAFE_API_KEY`, or store it in a file and point `TYPESAFE_API_KEY_FILE` to it.

```bash
python scripts/check_api.py                     # validate the key (free: lists models)
python scripts/run.py --list                    # all tasks (and the memorization probes)
python scripts/run.py all --split dev --limit 20 --dry-run --show 1   # inspect requests + cost estimate
python scripts/run.py all --split dev --limit 20 --max-cost 0.05      # send (asks for confirmation)
python scripts/evaluate.py all --split dev --limit 20                 # metrics -> results/dev/
python scripts/usage.py                         # spend per task, estimate accuracy, rejected requests
./scripts/run_full.sh                           # full eval run, detached; re-launch to resume (log in logs/)
```

Additional analyses used in the paper:

```bash
python scripts/run.py go_emotions unfair_tos agb_de toxic_chat --split dev --limit 1000   # threshold tuning sample
python scripts/run.py probes                    # memorization probes (MMLU/C-Eval calculation-heavy subjects)
```

- **Splits:** `--split dev` uses a training/validation split for prompt work; `--split eval` is the reported
  split. Freeze the question wording before running `eval`.
- **Sampling:** `--limit N` takes a deterministic, hash-ordered sample spread evenly over configs (languages,
  subjects). Smaller samples are subsets of larger ones, so pilot answers are reused.
- **Cache:** every Jev response is stored in `cache/responses.db`, keyed by the exact request; other models
  get one database each (`cache/responses.<org>__<model>.db`). Nothing is ever paid for twice, identical
  requests are sent once, and `evaluate.py` never calls the API. The Jev database doubles as the spend ledger.
- **Spend guards:** `--max-cost` caps a single run (default $0.02), and `JEV_TOTAL_BUDGET_USD` caps the total
  spend recorded in the ledger (default $10.40; set it to your own budget).

## Scoring open-weight models

The `openmodel` backend renders every Jev question as a chat prompt whose answer options are single tokens,
and reads the exact option probabilities from one forward pass (no generation, thinking disabled). The
responses have the same format as Jev's, so all evaluation code applies unchanged.

```bash
pip install -e ".[openmodel]"
python scripts/run_open_model.py boolq --split dev --limit 2 --show          # print prompts (tokenizer only)
python scripts/run_open_model.py all --model google/gemma-4-E4B-it --out runs/gemma --device cuda
python scripts/run_open_model.py all --shard 0 --num-shards 8 --out runs/gemma   # one shard of a parallel run
python scripts/import_responses.py runs/gemma/*.jsonl                        # -> cache/responses.google__gemma-4-E4B-it.db
python scripts/evaluate.py all --model hf:google/gemma-4-E4B-it
```

Runs are resumable and shardable. `--device auto` splits a model that does not fit on one GPU over all
visible GPUs. `scripts/marvin/` contains the Slurm scripts we used on the Marvin cluster (University of Bonn):
environment setup, prefetching models and datasets for offline GPU jobs, job arrays, and syncing. Adapt paths,
partitions and e-mail addresses to your own cluster.

## Layout

- `jev_benchmarking/tasks/`: one class per dataset (request construction and gold labels); `probes.py` holds
  the memorization probes (addressable by name or `probes`, never part of `all`)
- `jev_benchmarking/runner.py`: async, rate-limited, resumable sender for the TypeSafe API with a spend cap
- `jev_benchmarking/openmodel/`: prompt rendering and exact option scoring for open-weight models
- `jev_benchmarking/evaluate.py`, `metrics.py`: accuracy, F1, correlation, calibration (ECE, Brier), selective
  accuracy, bootstrap confidence intervals
- `jev_benchmarking/thresholds.py`: per-question Noul thresholds tuned on dev data
- `jev_benchmarking/report/`: paper tables (LaTeX) and figures (plotnine, tol-colors)
- `scripts/`: entry points; `scripts/marvin/`: Slurm scripts
- `results/eval/`: per-dataset metrics for Jev; `results/eval/open_models/`: the same for the open-weight models
- `responses/`: license of the released model responses

## License

The model responses published on Zenodo are licensed under the terms in
[responses/LICENSE_RESPONSES.md](responses/LICENSE_RESPONSES.md): the Jev responses under the Jev Responses
License 1.0 (research and evaluation use; no model distillation, no training models to imitate Jev, no
developing or facilitating similar or competing products), the Gemma-4-E4B and Qwen3.8-27B responses under the
Apache License 2.0.

## Citation

```bibtex
@misc{deußer2026evaluatingbenchmarkingmodeljev,
      title={Evaluating and Benchmarking the System One Model Jev}, 
      author={Tobias Deußer and Lorenz Sparrenberg and Rafet Sifa},
      year={2026},
      eprint={2609.37647},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.37647}, 
}
```
