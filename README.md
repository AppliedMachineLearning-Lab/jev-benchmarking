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

## Sequential decisions

The sequential track lets the model choose the next action in an environment at every step: one request per
step, with the state and one Choice question over the legal actions, executing the most probable option.
A reference action is logged at every step (never sent to the model), so step accuracy, ECE and Brier score can
be reported alongside the success rate.

| Task | Implementation | Episodes | Reference action | Environment |
|---|---|---|---|---|
| `alfworld` | ALFWorld (`alfworld` package), unseen split, 50 steps | 134 | handcoded ALFWorld expert | `.venv` + `alfworld-download` |
| `blackjack` | RLCard `blackjack`, default configuration | 5,000 | exact optimum (DP) | `.venv` |
| `frozenlake` | Gymnasium `FrozenLake-v1`, slippery, random 4x4 and 8x8 maps | 500 | exact optimum (DP) | `.venv` |
| `textworld` | TextWorld's `tw-make` challenges (coin collector, treasure hunter, cooking), 3 levels x 10 seeds, 50 steps; coin collector and treasure hunter with goal-only objectives | 90 | TextWorld solver (`policy_commands`) | `.venv` |
| `babyai_text` | original BabyAI-Text packages (Carta et al., 2023), `BabyAI-MixedTrainLocal-v0`, their prompt | 250 | BabyAI bot | `.venv-babyai` |
| `babyai_text_instructions` | diagnostic only: `babyai_text` with BALROG's instruction prompt (action descriptions, two tips) added to the state | 250 | BabyAI bot | `.venv-babyai` |

Install the sequential dependencies with `pip install -e ".[sequential]"` (then `alfworld-download` for
ALFWorld). TextWorld games are generated with `tw-make` on first use and cached in `~/.cache/jev-benchmarking/`.
BabyAI-Text needs an old gym and numpy and therefore its own environment:

```bash
bash scripts/setup_babyai_text_env.sh          # creates .venv-babyai (Python 3.10) with the original packages
source .venv-babyai/bin/activate
```

The API key can also be put in `.env` or `.env.local` (see `.env.example`).

```bash
python scripts/run_sequential.py blackjack --backend scripted:optimal          # reference policy, free
python scripts/run_sequential.py textworld --backend scripted:expert           # expert / solver reference
python scripts/run_sequential.py frozenlake --backend scripted:random --mode sample   # uniform policy (greedy would always move left)
python scripts/run_sequential.py frozenlake --dry-run                          # requests and cost estimate
python scripts/run_sequential.py frozenlake --backend jev --limit 1            # one episode first
python scripts/run_sequential.py frozenlake --backend jev --max-cost 1.0       # full run
python scripts/run_sequential.py alfworld --backend jev --shard 0 --num-shards 4   # one of 4 parallel processes
python scripts/run_sequential.py frozenlake --backend hf:Qwen/Qwen3.8-27B --device auto   # open-weight model
python scripts/run_sequential.py frozenlake --backend jev --summarize          # metrics only
python scripts/probe_frozenlake.py --backend jev                               # FrozenLake prompt-variant probe
python scripts/run_sequential.py babyai_text_instructions --backend jev --max-cost 1.0   # BabyAI-Text with BALROG's instruction prompt (diagnostic)
python scripts/inspect_episode.py frozenlake results/sequential/frozenlake/jev.greedy.eval.jsonl --episode 0   # replay and check one episode
```

- **Uniform reference policies:** `scripted:random` gives every action the same probability. Run it with
  `--mode sample`; under the default `--mode greedy` the tie always goes to the first action (in FrozenLake, "left").
- **Expert labels:** the ALFWorld handcoded expert breaks ties with Python's `random`; the environment seeds it per
  episode, so the expert's commands are reproducible. The TextWorld expert follows TextWorld's `policy_commands`,
  which can choose a different (equally winning) plan on a different platform or Python version: on the archived
  games the expert needs 851 steps in total on Linux (Python 3.13) and 863 on macOS (Python 3.12). Model trajectories
  do not depend on the expert.
- **Output:** one JSON line per episode in `results/sequential/<task>/<model>.<mode>.<split>.jsonl` (every step's
  action, option probabilities, reference action and feedback) and the metrics in `<...>.summary.json`.
- **Metrics:** success rate (95% bootstrap CI), mean steps, cost per episode, repeat rate, step accuracy against
  the reference, step ECE and step Brier; for Blackjack and FrozenLake also the regret per decision.
- **Cache and spend guards:** as for the single-step track; deterministic episodes replay from the cache for free,
  and runs resume where they stopped.

## Layout

- `jev_benchmarking/tasks/`: one class per dataset (request construction and gold labels); `probes.py` holds
  the memorization probes (addressable by name or `probes`, never part of `all`)
- `jev_benchmarking/runner.py`: async, rate-limited, resumable sender for the TypeSafe API with a spend cap
- `jev_benchmarking/openmodel/`: prompt rendering and exact option scoring for open-weight models
- `jev_benchmarking/evaluate.py`, `metrics.py`: accuracy, F1, correlation, calibration (ECE, Brier), selective
  accuracy, bootstrap confidence intervals
- `jev_benchmarking/thresholds.py`: per-question Noul thresholds tuned on dev data
- `jev_benchmarking/report/`: paper tables (LaTeX) and figures (plotnine, tol-colors)
- `jev_benchmarking/sequential/`: sequential track: environments (`envs/`), episode runner, backends, metrics
- `scripts/`: entry points; `scripts/marvin/`: Slurm scripts
- `results/eval/`: per-dataset metrics for Jev; `results/eval/open_models/`: the same for the open-weight models
- `responses/`: license of the released model responses

## License

The code in this repository is licensed under the [MIT License](LICENSE).

The model responses published on Zenodo are licensed under the terms in
[responses/LICENSE_RESPONSES.md](responses/LICENSE_RESPONSES.md): the Jev responses under the Jev Responses
License 1.0 (research and evaluation use; no model distillation, no training models to imitate Jev, no
developing or facilitating similar or competing products), the Gemma-4-E4B and Qwen3.8-27B responses under the
Apache License 2.0.

## Citation

```bibtex
@misc{deußer2026evaluatingbenchmarkingmodeljev,
      title={Evaluating and Benchmarking the System One Model Jev}, 
      author={Tobias Deußer and Rajkumar Ramamurthy and Lorenz Sparrenberg and Rafet Sifa},
      year={2026},
      eprint={2609.37647},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.37647}, 
}
```
