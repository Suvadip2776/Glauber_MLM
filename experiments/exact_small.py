"""Exactly solvable MLM Glauber chains.

Validates the mixing theory and the empirical diagnostics on chains small enough to
solve exactly, measures how t_mix scales with n, and compares the Glauber stationary
law with the pseudo-likelihood (PLL) target that an MH correction (Goyal et al. 2022)
samples from.

For every (template, n, tau) it reports, EXACTLY:
  t_mix(1/4) worst-case, d(2^k) curve, t_rel and spectral lower bound,
  Dobrushin alpha(tau) and the Theorem-5 upper bound when alpha < 1,
  best sweep-cut conductance Phi* and the bottleneck lower bound 1/(4 Phi*),
  TV(mu, pi_PLL), Spearman(mu, pi_PLL), flow asymmetry (non-reversibility),
  rectangle |delta|, t_mix of the MH chain targeting exp(PLL),
and, SIMULATED on the same chain, the paper's diagnostics:
  maximal-coupling times from random pairs and from the bottleneck pair,
  embedding-distance hitting times (optional, --embedder).

Example (≈10 min on one GPU, a few min on CPU for the default grid):
  python -m experiments.exact_small run --model bert-base-uncased \
      --n_slots 2 3 4 5 6 --K 4 --taus 0.3 0.5 0.75 1 1.5 2 3 --n_templates 3 \
      --out outputs/exact_small/bert
  python -m experiments.exact_small plot --out outputs/exact_small/bert
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from glauber import exact as ex
from glauber.metrics import censored_median
from glauber.models import MaskedLM
from glauber.utils import load_seed_texts, out_dir, pick_device, save_args, set_seed

DEFAULT_TEMPLATES = [
    "The overnight train rattled through the mountains as thunder echoed across the empty valley.",
    "Doctors recommend eating vegetables and exercising regularly to stay healthy.",
    "The team scored three goals in the final minutes to win the championship game.",
]


def run(args):
    set_seed(args.seed)
    out = out_dir(args.out)
    save_args(args, out / "args.json")
    dev = pick_device(args.device)
    lm = MaskedLM(args.model, device=dev)
    templates = (DEFAULT_TEMPLATES if args.templates == "default"
                 else load_seed_texts(args.templates, args.n_templates, seed=args.seed))[: args.n_templates]
    embedder = None
    if args.embedder != "none":
        from glauber.metrics import Embedder
        embedder = Embedder(args.embedder, device=dev)

    rows = []
    rng = np.random.default_rng(args.seed)
    for ti, text in enumerate(templates):
        for n in args.n_slots:
            if args.K ** n > args.max_states:
                print(f"skip n={n}: {args.K}^{n} states > --max_states")
                continue
            cache = out / f"chain_t{ti}_n{n}_K{args.K}_{args.slot_mode}.pkl"
            if cache.exists():
                ch = pickle.loads(cache.read_bytes())
            else:
                ch = ex.build_small_chain(lm, text, n, args.K, n_tokens=args.n_tokens, slot_mode=args.slot_mode)
                cache.write_bytes(pickle.dumps(ch))
            emb = embedder(ch.texts(lm)) if embedder is not None else None
            for tau in args.taus:
                rows.append(analyse(ch, lm, tau, args, rng, emb, ti))
                pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
                r = rows[-1]
                print(f"[t{ti} n={n} tau={tau}] S={r['S']} t_mix={r['t_mix']} t_rel={r['t_rel']:.1f} "
                      f"alpha={r['alpha']:.2f} bottleneckLB={r['bottleneck_lb']:.1f} "
                      f"coupling_med={r['coupling_median']} TV(mu,PLL)={r['tv_mu_pll']:.3f}")
    print(f"wrote {out/'results.csv'}")


def analyse(ch, lm, tau, args, rng, emb, ti):
    n, S = ch.n, ch.S
    probs = ex.cond_probs(ch, tau)
    P = ex.transition_matrix(ch, probs, "glauber")
    mu, resid = ex.stationary(P)
    t_mix, curve = ex.exact_mixing(P, mu, args.eps, args.max_doublings, device=args.matrix_device)
    vals, v2, t_rel = ex.spectrum(P)
    spec_lb = max(0.0, (t_rel - 1) * math.log(1 / (2 * args.eps)))
    phi, Sset = ex.bottleneck(P, mu, v2)
    alpha, _ = ex.exact_alpha(ch, probs)
    thm5 = n / (1 - alpha) * (math.log(n) + math.log(1 / args.eps)) if alpha < 1 else math.inf

    # Glauber stationary law vs the PLL target (what MH would sample, Goyal et al.)
    lpi = ex.log_pll(ch, probs)
    pi_pll = np.exp(lpi - lpi.max()); pi_pll /= pi_pll.sum()
    P_mh = ex.transition_matrix(ch, probs, "mh_pll")
    t_mix_mh, _ = ex.exact_mixing(P_mh, pi_pll, args.eps, args.max_doublings, device=args.matrix_device)
    delta = ex.rectangle_defects(ch, probs, args.n_rect, rng)

    # ---- diagnostics simulated on the same chain
    sim = ex.SmallSim(ch, probs)
    x0 = rng.integers(S, size=args.pairs)
    y0 = rng.integers(S, size=args.pairs)
    Tc = sim.coupling_times(x0, y0, args.sim_tmax, rng)
    worst = {}
    if Sset is not None and len(Sset) < S:
        a = Sset[np.argmax(mu[Sset])]
        comp = np.setdiff1d(np.arange(S), Sset)
        b = comp[np.argmax(mu[comp])]
        Tw = sim.coupling_times(np.full(args.pairs, a), np.full(args.pairs, b), args.sim_tmax, rng)
        worst = {"coupling_worstpair_median": censored_median(Tw),
                 "coupling_worstpair_censored": float(np.isinf(Tw).mean())}
    hit = {}
    if emb is not None:
        Th = sim.hitting_times(rng.integers(S, size=args.pairs), emb, args.hit_R, args.sim_tmax, rng)
        for k, R in enumerate(args.hit_R):
            hit[f"hit_R{R}_median"] = censored_median(Th[:, k])
            hit[f"hit_R{R}_censored"] = float(np.isinf(Th[:, k]).mean())

    top = np.argsort(-mu)[:3]
    top_pll = np.argsort(-pi_pll)[:3]
    texts = ch.texts(lm, np.concatenate([top, top_pll]))
    row = {
        "template": ti, "n": n, "K": args.K, "S": S, "tau": tau,
        "t_mix": t_mix, "t_rel": t_rel, "spectral_lb": spec_lb,
        "bottleneck_phi": phi, "bottleneck_lb": (1 / (4 * phi)) if phi > 0 else math.inf,
        "alpha": alpha, "thm5_bound": thm5,
        "n_log_n": n * math.log(n) if n > 1 else 1.0,
        "coupling_median": censored_median(Tc), "coupling_p90": float(np.quantile(Tc, 0.9)),
        "coupling_censored": float(np.isinf(Tc).mean()), **worst, **hit,
        "tv_mu_pll": ex.tv(mu, pi_pll), "spearman_mu_pll": float(spearmanr(mu, pi_pll).correlation),
        "t_mix_mh_pll": t_mix_mh, "flow_asymmetry": ex.flow_asymmetry(P, mu),
        "rect_abs_delta_mean": float(delta.mean()), "rect_abs_delta_median": float(np.median(delta)),
        "mu_max": float(mu.max()), "stationary_residual": resid,
        "d_curve": json.dumps(curve),
        "top_mu_texts": json.dumps(texts[:3]), "top_pll_texts": json.dumps(texts[3:]),
    }
    return row


def plot(args):
    import matplotlib.pyplot as plt
    out = Path(args.out)
    df = pd.read_csv(out / "results.csv").replace([np.inf], np.nan)
    g = df.groupby(["n", "tau"]).median(numeric_only=True).reset_index()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for tau, d in g.groupby("tau"):
        ax[0].plot(d["n"], d["t_mix"], "o-", label=f"τ={tau}")
    nn = np.arange(2, g["n"].max() + 1)
    ax[0].plot(nn, nn * np.log(nn) * 2, "k--", label="2·n log n")
    ax[0].set(yscale="log", xlabel="n (free sites)", ylabel="exact t_mix(1/4)", title="Exact mixing time")
    ax[0].legend(fontsize=7)

    for col, lab in [("coupling_median", "coupling median"), ("bottleneck_lb", "bottleneck LB"),
                     ("spectral_lb", "spectral LB"), ("thm5_bound", "Thm 5 UB")]:
        if col in df:
            ax[1].scatter(df["t_mix"], df[col], s=10, label=lab)
    lim = [1, np.nanmax(df["t_mix"]) * 2]
    ax[1].plot(lim, lim, "k:")
    ax[1].set(xscale="log", yscale="log", xlabel="exact t_mix", ylabel="diagnostic / bound",
              title="Diagnostics vs exact")
    hit_cols = [c for c in df if c.startswith("hit_R") and c.endswith("median")]
    for c in hit_cols:
        ax[1].scatter(df["t_mix"], df[c], s=10, marker="x", label=c)
    ax[1].legend(fontsize=7)

    for n, d in g.groupby("n"):
        ax[2].plot(d["tau"], d["tv_mu_pll"], "o-", label=f"n={n}")
    ax[2].set(xlabel="τ", ylabel="TV(μ_τ, π_PLL)", title="Glauber law ≠ PLL target")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "exact_small.png", dpi=180)
    print(f"wrote {out/'exact_small.png'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="bert-base-uncased")
    r.add_argument("--templates", default="default", help="'default', 'msmarco', or a .txt/.json file")
    r.add_argument("--n_templates", type=int, default=3)
    r.add_argument("--n_tokens", type=int, default=None, help="truncate template to this many tokens")
    r.add_argument("--n_slots", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    r.add_argument("--K", type=int, default=4)
    r.add_argument("--slot_mode", choices=["spread", "first"], default="spread",
                   help="'first' = contiguous slots (strong interactions); 'spread' = evenly spaced")
    r.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    r.add_argument("--eps", type=float, default=0.25)
    r.add_argument("--max_states", type=int, default=4096)
    r.add_argument("--max_doublings", type=int, default=40)
    r.add_argument("--pairs", type=int, default=200)
    r.add_argument("--sim_tmax", type=int, default=20000)
    r.add_argument("--n_rect", type=int, default=2000)
    r.add_argument("--embedder", default="none", help="e.g. sentence-transformers/all-MiniLM-L6-v2")
    r.add_argument("--hit_R", type=float, nargs="+", default=[0.2, 0.4])
    r.add_argument("--device", default="auto")
    r.add_argument("--matrix_device", default="cpu", help="device for dense P^t (cuda is much faster)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    p = sub.add_parser("plot")
    p.add_argument("--out", required=True)
    p.set_defaults(func=plot)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
