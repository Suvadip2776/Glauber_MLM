# Glauber Dynamics on Masked Language Models

This notebook implements **iterative masked-token resampling** as a Glauber dynamics Markov chain over the discrete space of token sequences. At each step, a random token position is masked and resampled from the MLM's temperature-scaled conditional distribution. The code provides tools to study mixing times, phase transitions, metastable trap states, and semantic drift in masked language models (MLMs) such as BERT, RoBERTa, and ModernBERT.

---

## Requirements

```bash
pip install torch transformers sentence-transformers datasets scikit-learn matplotlib plotly tqdm
```

A GPU is strongly recommended. All sections auto-detect CUDA and fall back to CPU.

---

## Models

Models are downloaded automatically from HuggingFace on first run. Experiments in the paper use:

| Model | HuggingFace ID |
|---|---|
| BERT-base-uncased | `bert-base-uncased` |
| RoBERTa-base | `roberta-base` |
| ModernBERT-base | `answerdotai/ModernBERT-base` |

Sentence embeddings use `jinaai/jina-embeddings-v2-base-en` (768-dim) or `all-MiniLM-L6-v2` depending on the section.

---

## Notebook Sections

### 1. PCA Visualizations

Runs a Glauber chain and tracks semantic drift by encoding each generated sentence with a SentenceTransformer. Projects the full embedding trajectory down to 2D via PCA and plots it, with color running from warm (early steps) to cool (late steps). Clusters in the trajectory correspond to trap states.

**Entry point:** `run_glauber_with_embeddings(...)`

| Parameter | Default | Description |
|---|---|---|
| `seed_text` | — | Starting sentence |
| `model_name` | `"roberta-base"` | MLM to run the chain on |
| `emb_model_name` | `"all-MiniLM-L6-v2"` | Sentence embedding model for tracking drift |
| `steps` | `50` | Number of Glauber steps |
| `temperature` | `1.0` | Softmax temperature. Lower = slower mixing / sharper peaks |
| `top_k` | `None` | If set, restricts sampling to the top-k tokens at each step |
| `print_every` | `10` | Print current sentence every N steps |

---

### 2. Probability and Sentence Analysis

Runs a basic Glauber chain, printing the evolving sentence and the top-3 candidate tokens at each sampled position. Useful for inspecting step-by-step dynamics and token-level probabilities.

**Entry point:** `run_glauber(...)`

| Parameter | Default | Description |
|---|---|---|
| `seed_text` | — | Starting sentence |
| `model_name` | `"roberta-base"` | MLM to use |
| `steps` | `200` | Number of Glauber steps |
| `temperature` | `1.0` | Softmax temperature |
| `top_k` | `None` | Restrict sampling to top-k tokens |
| `print_every` | `20` | Print interval |

---

### 3. MS MARCO + Fixed "Just Kidding!" Context

Samples a short passage from the MS MARCO dataset and runs two parallel Glauber chains on a shared update schedule: one chain (`v1`) updates freely, while the other (`v2`) has the suffix `" Just kidding!"` frozen in place. This lets you directly compare how a fixed tail context shifts the chain's stationary behavior.

**Entry point:** `run_glauber_with_jk(...)`

| Parameter | Default | Description |
|---|---|---|
| `seed_text` | MS MARCO sample | Starting passage (use `get_random_msmarco_passage()`) |
| `model_name` | `"roberta-base"` | MLM to use |
| `steps` | `2000` | Number of shared update steps |
| `temperature` | `1.0` | Softmax temperature |
| `top_k` | `None` | Restrict sampling to top-k tokens |
| `print_every` | `20` | Print interval |

---

### 4. Plotly Visualizations

An enhanced version of the PCA trajectory plot with interactive Plotly output. Hovering over any point shows the full sentence, step number, and phrase count. Also includes a second subplot tracking the occurrence count of a user-specified phrase over time.

**Entry point:** `run_glauber_with_embeddings(...)` with `use_plotly=True`

Additional parameters beyond Section 1:

| Parameter | Default | Description |
|---|---|---|
| `use_plotly` | `True` | Use interactive Plotly (`True`) or static Matplotlib (`False`) |
| `count_phrase_str` | `"isaac newton"` | Phrase to track occurrence counts for over time |

Output is saved as an `.html` file for viewing in a browser.

---

### 5. Coupling Experiment

Empirically measures mixing time using an **r-flip coupling**: two chains initialized from different MS MARCO passages update the same position at each step. With probability `r = Σ min(pₖ, qₖ)` they are forced to sample the same token; otherwise they sample independently. The number of steps until the chains become identical estimates the mixing time.

Running over a grid of temperatures and sequence lengths produces a coupling phase diagram — the main empirical result of Section 6.1 of the paper.

**Key functions:** `glauber_r_flip(...)`, `run_experiments(...)`, `plot_phase_diagram(...)`

| Parameter | Default | Description |
|---|---|---|
| `temperatures` | `[0.5 … 2.0]` | Temperature grid |
| `token_lengths` | `[5 … 35]` | Sequence length grid |
| `trials` | `3` | Independent coupling trials per grid point |
| `max_steps` | `1500` | Step budget; no coupling = `NaN` (shown as black in the heatmap) |

Also includes `get_word_probabilities(...)`, a standalone utility that masks a single position in a sentence and plots the top-k predictions — useful for inspecting conditional distributions directly.

---

### 6. Repeated Token Analysis and Minority Token Contamination

Masks every position in a sentence independently and records the top-k most probable replacement tokens at each one. Produces a grouped bar chart showing rank-1 and rank-2 predictions across all positions simultaneously. Used in the paper to demonstrate that a single minority token (e.g., one "Green" in an otherwise all-"Blue" sequence) biases predictions globally across the context window.

**Configuration (set at top of section):**

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `"roberta-base"` | MLM to query |
| `SENTENCE` | `""` | Input sentence to analyze |
| `TOP_K` | `2` | Number of top predictions to show per position |

---

### 7. Trap Detection on Embedding Trajectories

Scans an embedding trajectory (output of any of the Glauber runs above) for **metastable trap states**: contiguous windows where the chain's embedding stays within distance `epsilon` of its starting point for at least `trap_duration` steps.

**Entry point:** `detect_traps(embeddings, texts, epsilon, trap_duration)`

| Parameter | Default | Description |
|---|---|---|
| `embeddings` | — | List of embedding tensors from a prior Glauber run |
| `texts` | — | Corresponding list of decoded sentence strings |
| `epsilon` | `0.5` | Maximum embedding distance to still count as "stuck" |
| `trap_duration` | `1000` | Minimum number of steps to qualify as a trap |

---

## Experimental Setup

All experiments in the paper were run on Google Colab using T4 and A100 GPUs. Seed texts are drawn from [MS MARCO v2.1](https://huggingface.co/datasets/ms_marco) unless otherwise noted. Default hyperparameters across sections: `temperature = 1.0`, `steps = 10,000`, `trials = 100`.


