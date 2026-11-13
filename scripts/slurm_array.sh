#!/usr/bin/env bash
# Generic SLURM array launcher: one line of a job file = one task.
#   sbatch --array=1-$(wc -l < jobs/coupling.txt) scripts/slurm_array.sh jobs/coupling.txt
# Generate job files with scripts/make_jobs.py. Edit the #SBATCH lines for your cluster.
#SBATCH --job-name=glauber
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
[ -f .env ] && set -a && . ./.env && set +a
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
echo "[$(date)] $CMD"
eval "$CMD"
