"""Write one-command-per-line job files for scripts/slurm_array.sh (one (tau, n) cell per task).
   python scripts/make_jobs.py   -> jobs/coupling.txt, jobs/hitting.txt
Cells append to a shared CSV; the `run` commands skip/append safely per cell."""
from pathlib import Path

Path("jobs").mkdir(exist_ok=True)
taus = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
ns = [5, 10, 15, 20, 25, 30]
with open("jobs/coupling.txt", "w") as f:
    for t in taus:
        for n in ns:
            f.write(f"python -m experiments.coupling run --model roberta-base --taus {t} --ns {n} --pairs 16 "
                    f"--max_steps 10000 --out outputs/coupling/roberta\n")
with open("jobs/hitting.txt", "w") as f:
    for t in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0]:
        for n in [10, 20, 30, 50, 75, 100]:
            f.write(f"python -m experiments.hitting run --model bert-base-uncased --taus {t} --ns {n} --chains 100 "
                    f"--steps 5000 --every 5 --out outputs/hitting/bert\n")
print("wrote jobs/coupling.txt jobs/hitting.txt")
