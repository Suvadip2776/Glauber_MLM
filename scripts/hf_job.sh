#!/usr/bin/env bash
# Runs one stage of the experiments inside a Hugging Face Job.
# A storage bucket (e.g. hf://buckets/<org>/<bucket>) is mounted at /bucket:
#   /bucket/code     <- this repository (synced from the laptop)
#   /bucket/outputs  <- results, written step by step so nothing is lost on timeout
# Usage (inside the job):  bash /bucket/code/scripts/hf_job.sh <stage>
set -uo pipefail
STAGE=$1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
rm -rf /work && cp -r /bucket/code /work && cd /work
O=/bucket/outputs
mkdir -p $O/logs
LOG=$O/logs/status_$STAGE.txt
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a $LOG

pip install -q --no-cache-dir "numpy<2.3" scipy pandas matplotlib tqdm "datasets==3.6.0" "sentence-transformers==3.4.1" \
  requests httpx sentencepiece accelerate einops pyarrow 2>&1 | tail -2
pip install -q --no-cache-dir "transformers==${TRANSFORMERS_VERSION:-4.48.3}" 2>&1 | tail -2

step() {  # step <name> <cmd...>   (a failing step is logged; the stage continues)
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] START $name" | tee -a $LOG
  if "$@" > "$O/logs/${STAGE}__$name.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK    $name" | tee -a $LOG
  else
    echo "[$(date +%H:%M:%S)] FAIL  $name" | tee -a $LOG
    tail -20 "$O/logs/${STAGE}__$name.log"
  fi
}
P=python
E=sentence-transformers/all-MiniLM-L6-v2

mdm_suite() {  # mdm_suite <model> <tag>
  local M=$1 T=$2
  step smoke $P -m experiments.smoke --model $M
  step rectangle $P -m experiments.rectangle --models $M --ar_control none --n_rect 300 --out $O/rectangle_$T
  step coupling $P -m experiments.coupling run --model $M --taus 0.5 1 2 3 --ns 5 10 20 --pairs 16 \
    --max_steps 2000 --out $O/coupling/$T
  step coupling_plot $P -m experiments.coupling plot --out $O/coupling/$T
  step traps $P -m experiments.traps --model $M --tau 1 --chains 16 --steps 3000 --every 10 \
    --window 150 --min_len 60 --restarts 8 --restart_steps 1500 --out $O/traps/$T
  step quality $P -m experiments.sample_quality --model $M --n 32 --chains 16 --steps 1600 \
    --n_checkpoints 8 --out $O/quality/$T
}

case $STAGE in
  bert_basin)   # drift over the whole band, escape/entry scaling, certificate, natural traps
    for F in adversarial random; do
      step drift_$F $P -m experiments.drift_basin drift --ref_token '!' --N 100 --taus 0.5 1 --fill $F \
        --states 100 --out $O/drift/bang
    done
    step escape $P -m experiments.drift_basin escape --ref_token '!' --Ns 10 20 40 60 80 100 --tau 1 \
      --chains 64 --steps 20000 --out $O/drift/bang
    step entry $P -m experiments.drift_basin entry --ref_token '!' --Ns 10 20 40 60 80 100 --tau 1 \
      --chains 64 --steps 20000 --out $O/drift/bang
    step certify $P -m experiments.drift_basin certify --out $O/drift/bang
    step influence $P -m experiments.influence --model bert-base-uncased --out $O/influence/bert
    step traps $P -m experiments.traps --model bert-base-uncased --tau 1 --chains 30 --steps 10000 \
      --restarts 32 --out $O/traps/bert
    ;;
  bert_diag)    # Fig. 2 coupling, Fig. 3 hitting times, sampler quality
    step coupling $P -m experiments.coupling run --model roberta-base --out $O/coupling/roberta
    step coupling_plot $P -m experiments.coupling plot --out $O/coupling/roberta
    step coupling_basin $P -m experiments.coupling run --model roberta-base --pair_mode basin \
      --out $O/coupling/roberta_basin
    step coupling_basin_plot $P -m experiments.coupling plot --out $O/coupling/roberta_basin
    step hitting $P -m experiments.hitting run --model bert-base-uncased --taus 0.5 1 1.5 2 \
      --ns 10 20 50 100 --chains 100 --steps 5000 --every 5 --out $O/hitting/bert
    step hitting_plot $P -m experiments.hitting plot --out $O/hitting/bert
    step quality $P -m experiments.sample_quality --model bert-base-uncased --mh --mh_steps 1600 \
      --out $O/quality/bert
    ;;
  topics)       # topic basins / politics with calibrated nulls, three judges, two MLMs
    for M in bert-base-uncased roberta-base; do
      T=${M%%-*}
      step run_$T $P -m experiments.topics run --model $M --tau 1 --steps 10000 --record_every 200 \
        --out $O/topics/$T
      step bart_$T $P -m experiments.topics score --out $O/topics/$T --classifier facebook/bart-large-mnli
      step deberta_$T $P -m experiments.topics score --out $O/topics/$T \
        --classifier MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33
      step llm_$T $P -m experiments.topics score --out $O/topics/$T --backend openrouter
      for S in facebook/bart-large-mnli MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33 \
               "openrouter:${OPENROUTER_MODEL:-anthropic/claude-sonnet-4.5}"; do
        step sum_${T}_${S##*/} $P -m experiments.topics summarize --out $O/topics/$T --scorer "$S"
      done
    done
    ;;
  llada_traps|dream_traps|mdlm_traps)   # rerun only the trap step of an MDM stage
    T=${STAGE%_traps}
    case $T in llada) M=GSAI-ML/LLaDA-8B-Base ;; dream) M=Dream-org/Dream-v0-Base-7B ;;
      mdlm) M=kuleshov-group/mdlm-owt
            pip install -q --no-cache-dir "flash-attn==2.7.4.post1" --no-build-isolation 2>&1 | tail -2 ;; esac
    step traps $P -m experiments.traps --model $M --tau 1 --chains 16 --steps 3000 --every 10 \
      --window 150 --min_len 60 --restarts 8 --restart_steps 1500 --out $O/traps/$T ;;
  bert_basin_rest)   # bert_basin without the slow adversarial drift
    step drift_random $P -m experiments.drift_basin drift --ref_token '!' --N 100 --taus 0.5 1 --fill random \
      --states 100 --out $O/drift/bang
    step escape $P -m experiments.drift_basin escape --ref_token '!' --Ns 10 20 40 60 80 100 --tau 1 \
      --chains 64 --steps 20000 --out $O/drift/bang
    step entry $P -m experiments.drift_basin entry --ref_token '!' --Ns 10 20 40 60 80 100 --tau 1 \
      --chains 64 --steps 20000 --out $O/drift/bang
    step certify $P -m experiments.drift_basin certify --out $O/drift/bang
    step influence $P -m experiments.influence --model bert-base-uncased --out $O/influence/bert ;;
  bert_embed)   # steps that need the sentence embedder: Fig. 3 hitting times and Table 4 traps
    step hitting $P -m experiments.hitting run --model bert-base-uncased --taus 0.5 1 1.5 2 \
      --ns 10 20 50 100 --chains 100 --steps 5000 --every 5 --out $O/hitting/bert
    step hitting_plot $P -m experiments.hitting plot --out $O/hitting/bert
    step traps $P -m experiments.traps --model bert-base-uncased --tau 1 --chains 30 --steps 10000 \
      --restarts 32 --out $O/traps/bert ;;
  bert_finish)  # adversarial drift (fewer states, optimised at tau=1) and influence coefficients
    step influence $P -m experiments.influence --model bert-base-uncased --out $O/influence/bert
    step drift_adversarial $P -m experiments.drift_basin drift --ref_token '!' --N 100 --taus 1 0.5 \
      --fill adversarial --states 48 --levels 0.5 0.6 0.7 0.8 0.9 0.95 --out $O/drift/bang ;;
  topics_llm)   # rerun only the LLM-judge scoring and its summaries
    J="openrouter:${OPENROUTER_MODEL:-anthropic/claude-sonnet-4.5}"
    for T in bert roberta; do
      step llm_$T $P -m experiments.topics score --out $O/topics/$T --backend openrouter
      step sum_${T}_llm $P -m experiments.topics summarize --out $O/topics/$T --scorer "$J"
    done ;;
  llada) mdm_suite GSAI-ML/LLaDA-8B-Base llada ;;
  dream) mdm_suite Dream-org/Dream-v0-Base-7B dream ;;
  mdlm)
    pip install -q --no-cache-dir "flash-attn==2.7.4.post1" --no-build-isolation 2>&1 | tail -2
    mdm_suite kuleshov-group/mdlm-owt mdlm ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "[$(date +%H:%M:%S)] STAGE DONE" | tee -a $LOG
