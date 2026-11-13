"""Embedding-drift diagnostics (Section 6.1, Fig. 3, Fig. 7) on a (tau, n) grid.

Interpretation: hitting time of d(x_t, x_0) = 1 - cos(phi(x_t), phi(x_0))
measures *forgetting of the initial meaning*, which is necessary but not sufficient for
mixing. exact_small.py checks on solvable chains how well it tracks the exact t_mix.

  python -m experiments.hitting run --model bert-base-uncased --taus 0.3 0.5 0.7 1 1.3 1.5 2 3 \
      --ns 10 20 30 50 75 100 --chains 100 --steps 5000 --every 5 \
      --embedder sentence-transformers/all-MiniLM-L6-v2 --out outputs/hitting/bert
  python -m experiments.hitting plot --out outputs/hitting/bert
  # cluster: one (tau, n) cell per job with --taus X --ns Y (results append to one CSV)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glauber.chain import glauber_step
from glauber.metrics import Embedder, censored_median
from glauber.models import MaskedLM
from glauber.utils import load_seed_texts, out_dir, pick_device, save_args, set_seed


def run(args):
    set_seed(args.seed)
    out = out_dir(args.out); save_args(args, out / f"args_{'_'.join(map(str, args.taus))}_{'_'.join(map(str, args.ns))}.json")
    lm = MaskedLM(args.model, device=pick_device(args.device))
    emb = Embedder(args.embedder, device=pick_device(args.device))
    gen = torch.Generator(device=lm.device).manual_seed(args.seed) if lm.device != "mps" else None
    texts = load_seed_texts(args.seed_source, max(args.chains * 2, 300), seed=args.seed)
    for n in args.ns:
        pool = [c[:n] for c in (lm.content_ids(t) for t in texts) if len(c) >= n][: args.chains]
        for tau in args.taus:
            e = lm.frame(pool[0])
            ids = e.ids[None].repeat(len(pool), 1)
            ids[:, e.positions] = torch.tensor(pool, device=lm.device)
            pos_t = torch.tensor(e.positions, device=lm.device)
            e0 = emb(lm.decode_batch(ids, e.positions))
            T = np.full((len(pool), len(args.R)), np.inf)
            curve = []
            for t in range(1, args.steps + 1):
                glauber_step(lm, ids, pos_t, tau, gen)
                if t % args.every:
                    continue
                d = 1 - (emb(lm.decode_batch(ids, e.positions)) * e0).sum(1)
                curve.append((t, float(d.mean())))
                for k, R in enumerate(args.R):
                    T[(d >= R) & np.isinf(T[:, k]), k] = t
                if args.stop_when_all_hit and np.isfinite(T).all():
                    break
            rows = [{"tau": tau, "n": n, "chain": c, "R": R, "T": T[c, k], "steps": args.steps,
                     "model": args.model, "embedder": args.embedder}
                    for c in range(len(pool)) for k, R in enumerate(args.R)]
            cells = out_dir(args.out, "cells")  # one file per (tau, n): safe for parallel jobs
            pd.DataFrame(rows).to_csv(cells / f"hit_tau{tau}_n{n}.csv", index=False)
            pd.DataFrame([{"tau": tau, "n": n, "t": t, "mean_d": v} for t, v in curve]).to_csv(
                cells / f"curve_tau{tau}_n{n}.csv", index=False)
            print(f"tau={tau} n={n}: " + ", ".join(
                f"R={R}: med={censored_median(T[:, k])} hit={np.isfinite(T[:, k]).mean():.2f}"
                for k, R in enumerate(args.R)))


def plot(args):
    import matplotlib.pyplot as plt
    out = Path(args.out)
    df = pd.concat([pd.read_csv(f) for f in sorted((out / "cells").glob("hit_*.csv"))])
    df.to_csv(out / "hitting.csv", index=False)
    Rs = sorted(df["R"].unique())
    fig, ax = plt.subplots(2, len(Rs), figsize=(5 * len(Rs), 8), squeeze=False)
    for k, R in enumerate(Rs):
        d = df[df["R"] == R]
        hr = d.assign(hit=np.isfinite(d["T"])).pivot_table(index="tau", columns="n", values="hit", aggfunc="mean")
        hr = hr.sort_index(ascending=False)
        im = ax[0, k].imshow(hr.to_numpy(), aspect="auto", cmap="inferno", vmin=0, vmax=1)
        ax[0, k].set_xticks(range(len(hr.columns)), hr.columns); ax[0, k].set_yticks(range(len(hr.index)), hr.index)
        ax[0, k].set(title=f"P(hit d ≥ {R})", xlabel="n", ylabel="τ"); fig.colorbar(im, ax=ax[0, k])
        for tau, g in d.groupby("tau"):
            med = g.groupby("n")["T"].median()
            ax[1, k].plot(med.index, med.values, "o-", label=f"τ={tau}")
        nn = np.array(sorted(d["n"].unique()), float)
        ax[1, k].plot(nn, nn * np.log(nn), "k--", label="n log n")
        ax[1, k].set(xlabel="n", ylabel="median hitting time", yscale="log", title=f"R={R}")
        ax[1, k].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "hitting.png", dpi=180)
    print(f"wrote {out/'hitting.png'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="bert-base-uncased")
    r.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    r.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    r.add_argument("--ns", type=int, nargs="+", default=[10, 20, 50, 100])
    r.add_argument("--R", type=float, nargs="+", default=[0.4, 0.7, 1.0])
    r.add_argument("--chains", type=int, default=100)
    r.add_argument("--steps", type=int, default=5000)
    r.add_argument("--every", type=int, default=5)
    r.add_argument("--stop_when_all_hit", action="store_true")
    r.add_argument("--seed_source", default="msmarco")
    r.add_argument("--device", default="auto")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    p = sub.add_parser("plot"); p.add_argument("--out", required=True); p.set_defaults(func=plot)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
