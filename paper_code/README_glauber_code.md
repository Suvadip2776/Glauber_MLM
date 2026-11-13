# Glauber Dynamics on Masked Language Models — Code

Code accompanying *Mixing Times of Glauber Dynamics on Masked Language Models*.

`glauber_code.ipynb` contains six cells. The first three are self-contained
experiments that produce paper tables when run in the notebook. The last three
form a four-stage cluster pipeline for the §6.1 phase-boundary results.

| Cell | Role | Paper output |
|------|------|--------------|
| 0 | Rectangle incompatibility test across MLMs, with GPT-2 as autoregressive control | §4, Table 1 |
| 1 | Positive-drift certificate for the slow-mixing basin $B = \{x : \#\{i : x_i = \texttt{!}\} \geq 0.9N\}$ | §6.4, Table 4 |
| 2 | Empirical estimation of influence coefficients $c_{ij}(\tau)$ and $\alpha(\tau)$ | Appendix A.1 |
| 3 | Glauber chain runner — one $(\tau, n)$ cell per invocation | §6.1, phase diagram |
| 4 | Aggregation of cell-3 JSON outputs into a single CSV | (intermediate) |
| 5 | Plot median hitting time vs $\tau$ from the aggregated CSV | §6.1, Figure X |

## Cell-by-cell

### Cell 0 — Rectangle test (§4, Table 1)

Tests path-dependence of MLM conditional log-ratios. For each model, samples
300 rectangles defined by two positions $(i, j)$ and two replacement tokens;
measures

$$\delta = (s_{x \to y} + s_{y \to z}) - (s_{x \to w} + s_{w \to z})$$

where each $s_{\cdot \to \cdot}$ is a conditional log-probability ratio. A
compatible joint distribution implies $\delta = 0$ for every rectangle
(Proposition 4); $\delta \neq 0$ certifies incompatibility.

Models tested: `bert-base-uncased`, `bert-large-uncased`, `roberta-base`,
`roberta-large`, `answerdotai/ModernBERT-base`. Control: `gpt2`, where
$\delta = 0$ by telescoping construction (each edge is a joint-log-prob
difference).

**Outputs:** `rectangle_multi_model_results.pkl`, console table matching paper
Table 1.

**Expected runtime:** ~30 min on a single A100/V100 (six models loaded
sequentially).

### Cell 1 — Exclamation drift (§6.4, Table 4)

Verifies positive expected drift in the count of `!` tokens at the boundary of
the basin $B$. For each of 200 randomly sampled boundary states (exactly
$0.9N$ `!` tokens, positions chosen uniformly), computes the expected
one-step change in `!`-count under the Glauber kernel:

$$\delta_\tau(x) = \frac{1}{N}\!\left[\sum_{i:\, x_i \neq \texttt{!}} p_{\theta,\tau}(\texttt{!} \mid x_{-i}) - \sum_{i:\, x_i = \texttt{!}} \bigl(1 - p_{\theta,\tau}(\texttt{!} \mid x_{-i})\bigr)\right].$$

All 200 boundary states exhibit $\delta_\tau > 0$, certifying slow mixing for
$B$ via a standard drift / submartingale argument (paper §6.4).

The cell uses `TEMPERATURE = 0.5` by default. **To produce the τ=1.0 row of
Table 4, edit the constant and re-run.**

**Outputs:** `exclamation_drift.png`, `exclamation_drift_results.pkl`, console
summary matching paper Table 4.

**Expected runtime:** ~15–20 min on a single A100/V100.

### Cell 2 — Influence estimation (Appendix A.1)

Empirically estimates the influence coefficients

$$c_{ij}(\tau) = \sup_{\substack{x, y \in \mathcal{X} \\ x_k = y_k\ \forall k \neq j}} \|p_{\theta, \tau}(\cdot \mid x_{-i}) - p_{\theta, \tau}(\cdot \mid y_{-i})\|_{\mathrm{TV}}$$

and the contraction quantity $\alpha(\tau) = \max_i \sum_{j \neq i} c_{ij}(\tau)$
on a 14-token seed sentence. The supremum is replaced by an empirical maximum
over a finite set of contexts and replacement tokens, so reported values are
**lower bounds** on the true $c_{ij}(\tau)$ and $\alpha(\tau)$. This is
sufficient for the appendix's argument: the empirical α already greatly
exceeds 1 at τ=1, ruling out the contraction sufficient condition there.

Configuration constants live in the `class Args:` block at the top of the
cell — edit and re-run to change τ range, sample counts, etc.

**Outputs:** `influence_estimation_results.png`, console output of $\alpha(\tau)$
across temperatures.

**Expected runtime:** ~15 min on a single A100/V100.

### Cell 3 — Phase-boundary chain runner (§6.1)

Runs $N_{\text{chains}}$ Glauber chains in parallel (batched on one GPU) at a
single $(\tau, n)$ configuration, embeds the sequences periodically, and
records hitting times to thresholds $d \in \{0.4, 0.7, 1.0\}$ in
Jina/MiniLM cosine-distance space. Output is a single JSON file per
invocation.

**This cell is designed for SLURM array submission, not interactive
notebook execution.** It uses `argparse` and `if __name__ == "__main__":`
expecting CLI invocation. To use it from the notebook, export to a `.py`
file:

```bash
jupyter nbconvert --to script --no-prompt glauber_code.ipynb
```

Then submit a SLURM array with one job per $(\tau, n)$ cell. A typical sweep
covers $\tau \in [0.3, 3.0]$ and $n \in [10, 100]$ — 50–60 cells. Each cell
takes 10–60 minutes on an A100 depending on $n$ and `max_steps`.

**Outputs:** one `results/tau{τ}_n{n}.json` per job invocation, containing
chain trajectories and hitting times.

### Cell 4 — Aggregation

Collates every `results/*.json` from Cell 3 into a single
`phase_boundary.csv`, prints a per-cell hit-rate summary, and flags any
cells with non-100% hit rate (the censored / interesting cells). Run once
after all Cell-3 jobs complete.

Same CLI/main pattern as Cell 3 — invoke from a shell, not the notebook
kernel.

**Outputs:** `phase_boundary.csv`, console summary table.

### Cell 5 — Plotting

Reads `phase_boundary.csv` and produces `hitting_time_vs_tau.png`: median
hitting time vs $\tau$ on log-log axes, one curve per $n$, faceted by
threshold $d \in \{0.4, 0.7, 1.0\}$.

This cell **does** run in the notebook (no CLI). Run after Cell 4 has produced
the CSV.

## Workflow summary

```
# Notebook (interactive):
Cell 0  -> Table 1
Cell 1  -> Table 4 row τ=0.5
        (edit TEMPERATURE = 1.0, re-run)  -> Table 4 row τ=1.0
Cell 2  -> Appendix A.1 numbers

# Cluster (CLI):
jupyter nbconvert --to script glauber_code.ipynb     # extract cells as .py
sbatch --array=0-N submit.sh                          # one (τ, n) per job
python aggregate.py results/                          # write CSV

# Notebook (interactive again):
Cell 5  -> hitting_time_vs_tau.png
```

## Dependencies

```
torch >= 2.0
transformers >= 4.48        # ModernBERT requires recent transformers
numpy
matplotlib
sentence-transformers       # used by Cells 1 and 3 for embedding
scipy
pandas                      # cells 4 and 5
tqdm
```

Models are downloaded from HuggingFace on first run (~1.5 GB total across the
six MLMs in Cell 0).

## Hardware

All experiments target a single GPU. Tested on NVIDIA A100 (40 GB) and V100
(16 GB).

- Cells 0, 1, 2: single GPU, 15–30 min each
- Cell 3: single GPU per cell, 10–60 min depending on $(\tau, n)$; ~50 cells
  total → 8–40 GPU-hours, ideally parallelized
- Cells 4, 5: CPU-only, seconds

## Reproducibility

Each cell sets `random`, `numpy`, and `torch` seeds (default `42`). On GPU,
exact bit-level reproducibility additionally requires
`torch.use_deterministic_algorithms(True)`, which is not set by default.
Results are stable across runs to within sampling noise of the reported
summary statistics; individual rectangle/state values may differ slightly.

## Notes

- **GPT-2 control in Cell 0** computes $\delta$ from joint log-probability
  differences, not conditional log-ratios, so $\delta = 0$ holds algebraically
  rather than empirically. This is the intended comparison: AR models possess
  a joint and trivially pass the rectangle test; MLMs do not.
- **Cell 2's $\alpha(\tau)$ estimates** are sup approximations and should be
  read as lower bounds on the true contraction quantity. The structural
  conclusion of Appendix A.1 (that the contraction sufficient condition is far
  from being met at natural temperatures) follows from the lower bound alone.
- **Cell 1's drift quantity** (Table 4) is the expected one-step change in
  `!`-count, not the score margin $\Delta_i$ from Theorem 8. The drift
  argument is a separate slow-mixing certificate that does not require the
  uniform local margin hypothesis.
