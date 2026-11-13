#!/usr/bin/env bash
# Reduced-scale run of every experiment on a laptop (Apple M1, 16 GB, MPS).
# Lane A (CPU): exact solvable chains.  Lane B (MPS, sequential): everything else.
# LLaDA-8B / Dream-7B do not fit in 16 GB -> run those on a GPU node (README).
# Each step logs to $O/logs/<name>.log; a failing step is recorded and the lane continues.
set -uo pipefail
PY=${PYTHON:-python3}
cd "$(dirname "$0")/.."
O=${1:-outputs/local}
mkdir -p "$O/logs"
E=sentence-transformers/all-MiniLM-L6-v2

step() {  # step <name> <cmd...>
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a "$O/logs/status.txt"
  if "$@" > "$O/logs/$name.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK    $name" | tee -a "$O/logs/status.txt"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $name (see logs/$name.log)" | tee -a "$O/logs/status.txt"
  fi
}

laneA() {
  for M in spread first; do
    step exact_$M $PY -m experiments.exact_small run --model bert-base-uncased --device cpu \
      --n_slots 2 3 4 5 6 --K 4 --taus 0.3 0.5 0.75 1 1.5 2 3 --n_templates 3 --slot_mode $M \
      --embedder $E --out $O/exact_small/bert_$M
    step exact_${M}_plot $PY -m experiments.exact_small plot --out $O/exact_small/bert_$M
  done
}

laneB() {
  step rectangle $PY -m experiments.rectangle --n_rect 300 --out $O/rectangle
  step smoke_mdlm $PY -m experiments.smoke --model kuleshov-group/mdlm-owt
  step rectangle_mdlm $PY -m experiments.rectangle --models kuleshov-group/mdlm-owt --ar_control none \
    --n_rect 300 --out $O/rectangle_mdlm
  for F in model adversarial random; do
    step drift_$F $PY -m experiments.drift_basin drift --ref_token '!' --N 100 --taus 0.5 1 --fill $F \
      --states 40 --out $O/drift/bang
  done
  step escape $PY -m experiments.drift_basin escape --ref_token '!' --Ns 10 20 40 60 80 100 --tau 1 \
    --chains 32 --steps 10000 --out $O/drift/bang
  step entry $PY -m experiments.drift_basin entry --ref_token '!' --Ns 10 20 40 60 80 100 --tau 1 \
    --chains 32 --steps 10000 --out $O/drift/bang
  step certify $PY -m experiments.drift_basin certify --out $O/drift/bang
  step influence $PY -m experiments.influence --model bert-base-uncased --out $O/influence/bert
  step traps $PY -m experiments.traps --model bert-base-uncased --tau 1 --chains 30 --steps 10000 \
    --restarts 16 --out $O/traps/bert
  step coupling $PY -m experiments.coupling run --model roberta-base --max_steps 5000 --out $O/coupling/roberta
  step coupling_plot $PY -m experiments.coupling plot --out $O/coupling/roberta
  step coupling_basin $PY -m experiments.coupling run --model roberta-base --pair_mode basin \
    --taus 0.5 1 2 --ns 10 20 30 --max_steps 5000 --out $O/coupling/roberta_basin
  step hitting $PY -m experiments.hitting run --model bert-base-uncased --taus 0.5 1 1.5 2 --ns 10 20 50 100 \
    --chains 50 --steps 3000 --every 10 --out $O/hitting/bert
  step hitting_plot $PY -m experiments.hitting plot --out $O/hitting/bert
  step topics_run $PY -m experiments.topics run --model bert-base-uncased --tau 1 --steps 10000 \
    --record_every 250 --out $O/topics/bert
  step topics_bart $PY -m experiments.topics score --out $O/topics/bert --classifier facebook/bart-large-mnli
  step topics_deberta $PY -m experiments.topics score --out $O/topics/bert \
    --classifier MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
  step topics_llm $PY -m experiments.topics score --out $O/topics/bert --backend openrouter
  step topics_sum_bart $PY -m experiments.topics summarize --out $O/topics/bert --scorer facebook/bart-large-mnli
  step topics_sum_deberta $PY -m experiments.topics summarize --out $O/topics/bert \
    --scorer MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
  step topics_sum_llm $PY -m experiments.topics summarize --out $O/topics/bert \
    --scorer "openrouter:${OPENROUTER_MODEL:-anthropic/claude-sonnet-4.5}"
  step quality $PY -m experiments.sample_quality --model bert-base-uncased --mh --mh_steps 1600 \
    --out $O/quality/bert
}

laneA &
laneB
wait
echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$O/logs/status.txt"
