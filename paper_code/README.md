# Code used for the submitted paper

These are the original scripts and READMEs behind the submitted version of
*Mixing Times of Glauber Dynamics on Masked Language Models*:

* `political_drift_experiments.py`, the politics/topic recurrence experiment (§6.2, Table 3), with
  `README_political_drift.md`.
* `README_glauber_dynamics_notebook.md` and `README_glauber_code.md` describe the original notebooks
  (coupling phase diagram, embedding hitting times, traps, rectangle test, drift).

The rest of this repository (`glauber/`, `experiments/`, `scripts/`) is the main codebase. It
re-implements every one of these experiments behind one model adapter, corrects the coupling to a
proper maximal coupling, adds calibrated null baselines, and adds further experiments (exact solvable
chains, drift certificates, masked-diffusion LMs). See the top-level README.
