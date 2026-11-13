"""Coupling phase diagram (Section 6.1, Fig. 2) with an exact maximal coupling.

What the numbers mean:
  * coupling inequality: ||P^t(x,.) - P^t(y,.)||_TV <= P(T_xy > t) for EACH pair, so
    the empirical survival of T_xy is an upper-bound estimate for the pairs tested;
    it bounds worst-case t_mix only if the tested pairs include the worst starts.
  * failure to couple within a budget is NOT a lower bound on t_mix. Lower bounds
    come from experiments/drift_basin.py (certify) and exact_small.py.
  --pair_mode random       independent MS MARCO passages (the paper's Fig. 2)
  --pair_mode basin        one chain started in the all-'!' state (or --ref_text trap)
                           and one from MS MARCO: a deliberately adversarial pair.

Coupling construction: the shared token is drawn from min(p, q)/r with probability
r = sum_k min(p_k, q_k). Drawing the non-shared tokens independently from p and q would
give the wrong marginals (X ~ min(p,q) + (1-r)p ≠ p), so here the non-shared draws use the residuals
(p - min)/(1 - r), (q - min)/(1 - r), which makes each chain an exact Glauber chain.

  python -m experiments.coupling run --model roberta-base --taus 0.5 1 1.5 2 2.5 3 \
      --ns 5 10 15 20 25 30 --pairs 16 --max_steps 10000 --out outputs/coupling/roberta
  python -m experiments.coupling plot --out outputs/coupling/roberta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glauber.chain import coupled_step
from glauber.metrics import censored_median, survival
from glauber.models import MaskedLM
from glauber.utils import load_seed_texts, out_dir, pick_device, save_args, set_seed


@torch.no_grad()
def couple(lm, X, Y, positions, tau, max_steps, gen):
    pos_t = torch.tensor(positions, device=lm.device)
    B = X.shape[0]
    T = np.full(B, np.inf)
    eq = (X[:, positions] == Y[:, positions]).all(1).cpu().numpy()
    T[eq] = 0
    for t in range(1, max_steps + 1):
        live = np.where(~np.isfinite(T))[0]
        if len(live) == 0:
            break
        li = torch.tensor(live, device=lm.device)
        Xl, Yl = X[li], Y[li]
        coupled_step(lm, Xl, Yl, pos_t, tau, gen)
        X[li], Y[li] = Xl, Yl
        met = (Xl[:, positions] == Yl[:, positions]).all(1).cpu().numpy()
        T[live[met]] = t
    return T


def run(args):
    set_seed(args.seed)
    out = out_dir(args.out); save_args(args, out / "args.json")
    lm = MaskedLM(args.model, device=pick_device(args.device))
    gen = torch.Generator(device=lm.device).manual_seed(args.seed) if lm.device != "mps" else None
    texts = load_seed_texts(args.seed_source, args.n_texts, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    cells = out_dir(args.out, "cells")  # one CSV per (tau, n): safe for parallel SLURM tasks
    for n in args.ns:
        pool = [c[:n] for c in (lm.content_ids(t) for t in texts) if len(c) >= n]
        for tau in args.taus:
            cell = cells / f"tau{tau}_n{n}_{args.pair_mode}.csv"
            if cell.exists():
                continue
            ix = rng.choice(len(pool), 2 * args.pairs, replace=len(pool) < 2 * args.pairs)
            Xc = torch.tensor([pool[i] for i in ix[: args.pairs]], device=lm.device)
            Yc = torch.tensor([pool[i] for i in ix[args.pairs:]], device=lm.device)
            if args.pair_mode == "basin":
                if args.ref_text:
                    ref = lm.content_ids(args.ref_text)[:n]
                    Xc = torch.tensor([ref] * args.pairs, device=lm.device)
                else:
                    Xc = torch.full_like(Xc, lm.token_id(args.ref_token))
            e = lm.frame(Xc[0].tolist())
            X = e.ids[None].repeat(args.pairs, 1); X[:, e.positions] = Xc
            Y = e.ids[None].repeat(args.pairs, 1); Y[:, e.positions] = Yc
            T = couple(lm, X, Y, e.positions, tau, args.max_steps, gen)
            df = pd.DataFrame({"tau": tau, "n": n, "pair": range(args.pairs), "T": T,
                               "pair_mode": args.pair_mode, "model": args.model})
            df.to_csv(cell, index=False)
            print(f"tau={tau} n={n}: median T={censored_median(T)} censored={np.isinf(T).mean():.2f}")


def plot(args):
    import matplotlib.pyplot as plt
    out = Path(args.out)
    df = pd.concat([pd.read_csv(f) for f in sorted((out / "cells").glob("*.csv"))])
    df.to_csv(out / "coupling.csv", index=False)
    df["T"] = df["T"].replace(np.inf, np.nan)
    agg = df.groupby(["tau", "n"])["T"].agg(
        median=lambda s: s.fillna(np.inf).median(), censored=lambda s: s.isna().mean()).reset_index()
    agg.to_csv(out / "coupling_summary.csv", index=False)
    piv = agg.pivot(index="tau", columns="n", values="median").sort_index(ascending=False)
    m = np.ma.masked_invalid(piv.replace(np.inf, np.nan).to_numpy())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap("inferno_r").copy(); cmap.set_bad("black")
    im = ax[0].imshow(m, aspect="auto", cmap=cmap)
    ax[0].set_xticks(range(len(piv.columns)), piv.columns); ax[0].set_yticks(range(len(piv.index)), piv.index)
    ax[0].set(xlabel="n", ylabel="τ", title="median coupling time (black = >50% censored)")
    fig.colorbar(im, ax=ax[0])
    grid = np.unique(np.logspace(0, np.log10(df["T"].max(skipna=True) or 10), 50).astype(int))
    for (tau, n), g in df.groupby(["tau", "n"]):
        if n == max(df["n"]) or n == min(df["n"]):
            ax[1].plot(grid, survival(g["T"].fillna(np.inf), grid), label=f"τ={tau}, n={n}")
    ax[1].set(xscale="log", xlabel="t", ylabel="P(T_couple > t)  (≥ ||P^t(x)-P^t(y)||_TV)",
              title="coupling survival = per-pair TV upper bound")
    ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "coupling.png", dpi=180)
    print(f"wrote {out/'coupling.png'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="roberta-base")
    r.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    r.add_argument("--ns", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    r.add_argument("--pairs", type=int, default=16)
    r.add_argument("--max_steps", type=int, default=10000)
    r.add_argument("--pair_mode", choices=["random", "basin"], default="random")
    r.add_argument("--ref_token", default="!")
    r.add_argument("--ref_text", default=None)
    r.add_argument("--seed_source", default="msmarco")
    r.add_argument("--n_texts", type=int, default=500)
    r.add_argument("--device", default="auto")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    p = sub.add_parser("plot"); p.add_argument("--out", required=True); p.set_defaults(func=plot)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
