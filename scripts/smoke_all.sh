#!/usr/bin/env bash
# Tiny CPU/MPS run of every experiment (~10 min, downloads bert-base-uncased, gpt2,
# all-MiniLM-L6-v2 and a small zero-shot model). Verifies the pipeline, not the science.
set -euo pipefail
PY=${PYTHON:-python3}
cd "$(dirname "$0")/.."
O=${1:-outputs/smoke}
P=$O/passages.txt; mkdir -p "$O"
$PY - <<PY
import json; d=json.load(open("configs/topic_seeds.json")); a=[x for v in d.values() for x in v]
open("$P","w").write("\n".join(" ".join(a[i:i+4]) for i in range(0,len(a)-4,2)))
json.dump({k:v[:2] for k,v in list(d.items())[:2]}, open("$O/seeds_small.json","w"))
PY
E=sentence-transformers/all-MiniLM-L6-v2
C=MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33
$PY -m experiments.smoke
$PY -m experiments.exact_small run --n_templates 1 --n_slots 2 3 4 --K 4 --taus 0.5 1 2 --pairs 50 --sim_tmax 2000 --embedder $E --out $O/exact
$PY -m experiments.exact_small plot --out $O/exact
$PY -m experiments.rectangle --models bert-base-uncased --n_rect 10 --n_tokens 20 --seed_source $P --n_texts 20 --out $O/rect
$PY -m experiments.drift_basin drift --N 12 --taus 0.5 1 --levels 0.5 0.9 1 --fill model --states 4 --batch 4 --out $O/drift
$PY -m experiments.drift_basin escape --Ns 10 --chains 4 --steps 60 --out $O/drift
$PY -m experiments.drift_basin entry --Ns 10 --chains 4 --steps 60 --seed_source $P --out $O/drift
$PY -m experiments.drift_basin certify --out $O/drift
$PY -m experiments.coupling run --taus 3 --ns 5 --pairs 4 --max_steps 300 --seed_source $P --n_texts 20 --out $O/coupling
$PY -m experiments.coupling plot --out $O/coupling
$PY -m experiments.hitting run --taus 2 --ns 8 --chains 4 --steps 40 --every 5 --embedder $E --seed_source $P --out $O/hitting
$PY -m experiments.influence --taus 1 5 --n 6 --chains 2 --burn 10 --snap_every 10 --snaps 2 --seed_source $P --out $O/influence
$PY -m experiments.traps --tau 0.3 --chains 2 --steps 300 --n 8 --every 5 --window 50 --thresh 0.05 --min_len 50 --embedder $E --seed_source $P --out $O/traps
$PY -m experiments.topics run --steps 40 --record_every 10 --seed_json $O/seeds_small.json --n_null 6 --seed_source $P --out $O/topics
$PY -m experiments.topics score --out $O/topics --classifier $C
$PY -m experiments.topics summarize --out $O/topics --scorer $C --burn 10
$PY -m experiments.sample_quality --n 8 --chains 3 --steps 30 --n_checkpoints 3 --mh --mh_steps 8 --scorer gpt2 --seed_source $P --out $O/quality
echo "SMOKE OK -> $O"
