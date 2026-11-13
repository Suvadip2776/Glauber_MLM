# Mixing Times of Glauber Dynamics on Masked Language Models

Code for the paper *Mixing Times of Glauber Dynamics on Masked Language Models*.

Glauber dynamics here means: repeatedly pick a random token position, mask it, and resample it from a
masked language model's temperature-scaled conditional. This repository runs that chain on any masked
LM and measures everything in the paper: the rectangle incompatibility test, exact and empirical mixing
times, slow-mixing basins, coupling and hitting-time phase diagrams, trap states and topic basins.

---

## 1. Setup (5 minutes)

Requires Python ≥ 3.10. A GPU is recommended for full-size runs. Everything also runs on CPU or
Apple Silicon (MPS) at smaller sizes.

```bash
git clone https://github.com/Suvadip2776/Glauber_MLM.git
cd Glauber_MLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: API keys, see Section 6
```

## 2. Check that it works (≈10 minutes, laptop is fine)

```bash
bash scripts/reproduce.sh smoke
```

This runs every experiment at a tiny size and should end with `SMOKE OK`. It downloads
`bert-base-uncased`, `gpt2`, a small sentence embedder and a small zero-shot classifier (~1.5 GB).

## 3. Reproduce a result: one command each

```bash
bash scripts/reproduce.sh list          # show everything available
bash scripts/reproduce.sh fig2          # e.g. the coupling phase diagram
```

| Command | Paper result | Hardware | Main output files (in `outputs/<command>/`) |
|---|---|---|---|
| `table1` | §4, Table 1, Figs 5–6: rectangle incompatibility test | GPU or laptop (≈20 min on an M1) | `summary.csv`, `rect_<model>.csv` |
| `exact` | §5: exact mixing times on solvable BERT chains | CPU | `<mode>/results.csv`, `<mode>/exact_small.png` |
| `table2` | §5.4, Table 2: `!` basin drift and slow-mixing certificate | GPU | `drift_<fill>_N100_summary.csv`, `escape_tau1.0.json`, `entry_tau1.0.json`, `certificates.csv` |
| `fig2` | §6.1, Fig. 2: coupling phase diagram (RoBERTa-base) | GPU | `coupling.png`, `coupling_summary.csv` |
| `fig3` | §6.1, Figs 3 and 7: embedding hitting times (BERT-base) | GPU | `hitting.png`, `hitting.csv` |
| `table3` | §6.2, Table 3, App. B.5: topic basins / politics | GPU | `occupancy_<scorer>.csv`, `politics_<scorer>.csv`, `transitions_<scorer>.csv` |
| `table4` | App. A.1, Table 4, App. C: trap states | GPU | `traps.csv`, `traps.json`, `trajectories.csv` |
| `influence` | App. A.1: influence coefficients α(τ) | GPU | `influence.csv`, `influence_summary.csv` |
| `quality` | Glauber as a sampler vs MH-PLL and autoregressive | GPU | `quality.csv`, `samples.csv` |
| `mdm` | The same experiments on a masked diffusion LM | GPU with ≥ 24 GB | `rectangle/`, `coupling/`, `traps/` |

Useful variations:

```bash
OUT=outputs/fig2_test bash scripts/reproduce.sh fig2 --taus 1 2 --ns 5 10   # extra args go to the script
MODEL=roberta-base bash scripts/reproduce.sh table3                         # topic basins on another MLM
MODEL=Dream-org/Dream-v0-Base-7B bash scripts/reproduce.sh mdm              # LLaDA is the default
PYTHON=python3.11 bash scripts/reproduce.sh table1                          # choose the interpreter
```

Every experiment is also a normal Python module with `--help`, for example
`python -m experiments.coupling run --help`. The docstring at the top of each file in `experiments/`
explains what it measures and gives the full command.

## 4. What each experiment does

| Script | What it measures |
|---|---|
| `experiments/rectangle.py` | Rectangle defect δ for random two-site swaps. Any nonzero δ certifies that no joint distribution matches the model's conditionals. GPT-2 is the autoregressive control (δ = 0 exactly). |
| `experiments/exact_small.py` | Restricts n positions to K candidate tokens each (≤ 4096 states) so the chain can be solved exactly: stationary law, worst-case mixing time, spectral and conductance bounds, Dobrushin α(τ), and the coupling/hitting diagnostics on the same chain. Also compares with the MH chain that targets the pseudo-likelihood (Goyal et al. 2022). |
| `experiments/drift_basin.py` | For a basin around a reference sequence (default: all `!`), the expected one-step drift on every level of a band, escape and entry times, and a two-start lower bound on the mixing time. |
| `experiments/coupling.py` | Two chains under an exact maximal coupling; time until they agree, over a (τ, n) grid. |
| `experiments/hitting.py` | Time until the sentence embedding moves a cosine distance R from its start, over a (τ, n) grid. |
| `experiments/traps.py` | Detects trap states (300-step mean embedding drift < 0.12 for ≥ 150 steps), computes score margins Δ̄ and Δ*, labels perfect vs approximate traps, and measures escape times by restarting chains from each trap. |
| `experiments/topics.py` | Long chains from topic-seeded sentences, scored by zero-shot classifiers (and optionally an LLM judge), with null baselines from natural text and random tokens. |
| `experiments/influence.py` | Influence coefficients c_ij(τ) and α(τ) at the seed sentence and at states the chain actually visits. |
| `experiments/sample_quality.py` | Perplexity, distinct-2 and repetition of Glauber samples vs number of steps. |
| `experiments/smoke.py` | Sanity check for a model: run it before long jobs on an unfamiliar model. |

## 5. Outputs

Each run writes to its `--out` folder:

* `args*.json`: the exact arguments used, so any result can be rerun.
* `.csv` files with per-chain or per-sample rows, plus `summary` files.
* `.png` figures from the `plot` subcommands.

Times are in single-site Glauber steps; one "sweep" is n steps. An infinite value (`inf`) means the
event did not happen within the step budget.

## 6. API keys (both optional)

Put them in `.env`. It is in `.gitignore`, so it is never committed.

* `HF_TOKEN`: every model and dataset used is public. A token only avoids download rate limits on
  shared clusters.
* `OPENROUTER_API_KEY`: used only by `python -m experiments.topics score --backend openrouter`, an
  LLM-judge robustness check for the topic analysis. Responses are cached in
  `<out>/.openrouter_cache.jsonl`. `OPENROUTER_MODEL` picks the judge model.

## 7. Running at scale

* **Laptop, reduced sizes, everything in one go:** `bash scripts/run_local.sh outputs/local`.
  Logs go to `outputs/local/logs/`.
* **SLURM cluster:** `python scripts/make_jobs.py` writes one line per (τ, n) cell to `jobs/`, then
  `sbatch --array=1-36 scripts/slurm_array.sh jobs/coupling.txt`. Each cell writes its own CSV, so
  parallel jobs never collide. Run the `plot` subcommand afterwards.
* **Hugging Face Jobs:** `scripts/hf_job.sh <stage>` runs one group of experiments inside a job
  that mounts a storage bucket at `/bucket` (code in `/bucket/code`, results in `/bucket/outputs`).
  Point the job's volume at your own bucket.

## 8. Models

Any Hugging Face masked LM works through the same adapter (`glauber/models.py`):

| Family | Example ids | Notes |
|---|---|---|
| BERT-style MLMs | `bert-base-uncased`, `bert-large-uncased`, `roberta-base`, `roberta-large`, `answerdotai/ModernBERT-base` | CPU, MPS or GPU. |
| LLaDA | `GSAI-ML/LLaDA-8B-Base` | ~16 GB in bf16. Use `transformers==4.46.2` if loading fails. |
| Dream | `Dream-org/Dream-v0-Base-7B` | ~15 GB in bf16. Needs `transformers==4.46.2`. |
| MDLM | `kuleshov-group/mdlm-owt` | CUDA only. Needs `pip install "flash-attn==2.7.4.post1" --no-build-isolation`. |

## 9. Troubleshooting

* **First run is slow:** MS MARCO passages are downloaded once and cached in `outputs/.cache/`.
  Models are cached by Hugging Face in `~/.cache/huggingface`.
* **Apple Silicon:** set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Random seeds are not reproducible on MPS.
* **Out of GPU memory:** lower `--pairs` (coupling) or `--chains` (other scripts).
* **Long runs on a laptop:** keep it plugged in with the lid open; sleep pauses the run.
* **A new model behaves oddly:** run `python -m experiments.smoke --model <id>` first. It checks
  tokenization, a fill-in-the-blank, Glauber steps and the coupling.

## 10. Repository layout

```
glauber/            the library
  models.py         MaskedLM adapter (BERT-style, LLaDA, Dream, MDLM) and CausalLM scorer
  chain.py          batched Glauber step, maximal coupling, pseudo-likelihood, MH step
  exact.py          exact small chains: kernel, stationary law, exact t_mix, spectrum, α, bottleneck
  metrics.py        confidence intervals, Clopper-Pearson, survival curves, trap detection, embedder
  topics.py         zero-shot and OpenRouter topic judges
  utils.py          seeds, devices, .env loading, MS MARCO seed texts
experiments/        one script per experiment (Section 4)
scripts/            reproduce.sh (start here), smoke test, laptop run, SLURM and HF Jobs runners
configs/            topic seed sentences
paper_code/         original scripts used for the first version of the paper
```
