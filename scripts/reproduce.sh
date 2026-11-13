#!/usr/bin/env bash
# One command per paper result.  Usage:
#   bash scripts/reproduce.sh <name> [extra args passed to the experiment]
#   bash scripts/reproduce.sh list
# Results go to outputs/<name>/ (override with OUT=...). Use PYTHON=... to pick an interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
NAME=${1:-list}; shift || true
OUT=${OUT:-outputs/$NAME}
EMB=sentence-transformers/all-MiniLM-L6-v2

case $NAME in
  table1)      # §4, Table 1, Figs 5-6: rectangle incompatibility test (5 MLMs + GPT-2 control)
    $PY -m experiments.rectangle --n_rect 300 --out $OUT "$@" ;;
  exact)       # §5: exact mixing times on solvable BERT chains (CPU is fine)
    for M in first spread; do
      $PY -m experiments.exact_small run --model bert-base-uncased --n_slots 2 3 4 5 6 --K 4 \
        --taus 0.3 0.5 0.75 1 1.5 2 3 --n_templates 3 --slot_mode $M --embedder $EMB --out $OUT/$M "$@"
      $PY -m experiments.exact_small plot --out $OUT/$M
    done ;;
  table2)      # §5.4, Table 2: '!' basin drift, escape/entry times and the two-start certificate
    for F in model adversarial random; do
      $PY -m experiments.drift_basin drift --ref_token '!' --N 100 --taus 0.5 1 --fill $F --states 100 --out $OUT "$@"
    done
    $PY -m experiments.drift_basin escape --ref_token '!' --Ns 10 20 40 60 80 100 --tau 1 --chains 64 --steps 20000 --out $OUT
    $PY -m experiments.drift_basin entry  --ref_token '!' --Ns 10 20 40 60 80 100 --tau 1 --chains 64 --steps 20000 --out $OUT
    $PY -m experiments.drift_basin certify --out $OUT ;;
  fig2)        # §6.1, Fig. 2: coupling phase diagram on RoBERTa-base
    $PY -m experiments.coupling run --model roberta-base --out $OUT "$@"
    $PY -m experiments.coupling plot --out $OUT ;;
  fig3)        # §6.1, Figs 3 and 7: embedding hitting times on BERT-base
    $PY -m experiments.hitting run --model bert-base-uncased --taus 0.5 1 1.5 2 --ns 10 20 50 100 --out $OUT "$@"
    $PY -m experiments.hitting plot --out $OUT ;;
  table3)      # §6.2, Table 3, App. B.5: topic basins / politics with null baselines
    $PY -m experiments.topics run --model ${MODEL:-bert-base-uncased} --tau 1 --steps 10000 --record_every 200 --out $OUT "$@"
    $PY -m experiments.topics score --out $OUT --classifier facebook/bart-large-mnli
    $PY -m experiments.topics score --out $OUT --classifier MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
    $PY -m experiments.topics summarize --out $OUT --scorer facebook/bart-large-mnli
    $PY -m experiments.topics summarize --out $OUT --scorer MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33 ;;
  table4)      # App. A.1, Table 4, App. C: trap states (detector, margins, escape times)
    $PY -m experiments.traps --model bert-base-uncased --tau 1 --chains 30 --steps 10000 --restarts 32 --out $OUT "$@" ;;
  influence)   # App. A.1: influence coefficients alpha(tau)
    $PY -m experiments.influence --model bert-base-uncased --out $OUT "$@" ;;
  quality)     # Glauber as a sampler: fluency/diversity vs steps, with MH-PLL and AR references
    $PY -m experiments.sample_quality --model bert-base-uncased --mh --mh_steps 1600 --out $OUT "$@" ;;
  mdm)         # same experiments on a masked diffusion LM (GPU, ~20 GB):  MODEL=Dream-org/Dream-v0-Base-7B
    M=${MODEL:-GSAI-ML/LLaDA-8B-Base}
    $PY -m experiments.smoke --model $M
    $PY -m experiments.rectangle --models $M --ar_control none --n_rect 300 --out $OUT/rectangle
    $PY -m experiments.coupling run --model $M --taus 0.5 1 2 3 --ns 5 10 20 --max_steps 2000 --out $OUT/coupling
    $PY -m experiments.coupling plot --out $OUT/coupling
    $PY -m experiments.traps --model $M --tau 1 --chains 16 --steps 3000 --restarts 8 --out $OUT/traps ;;
  smoke)       # tiny end-to-end run of everything (~10 min on a laptop)
    bash scripts/smoke_all.sh ;;
  list|*)
    grep -E '^  [a-z0-9]+\)' "$0" | sed -E 's/^  ([a-z0-9]+)\) *# */  \1\t/' | column -t -s $'\t' ;;
esac
