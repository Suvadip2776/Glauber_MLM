"""Influence coefficients: worst-case (Appendix A.1) vs states the chain visits.

For a state x, site j and alternative b, y = x(j->b);
    c_ij(x) = max_b TV(p_tau(.|x_-i), p_tau(.|y_-i)),  alpha(x) = max_i sum_{j≠i} c_ij(x).
Maximising over a finite set of (x, b) gives a LOWER bound on the worst-case
alpha(tau) of Definition 2, so alpha_hat > 1 rules the Theorem-5 condition out,
but alpha_hat < 1 cannot certify it. The script also reports the
distribution of alpha(x) over states actually visited at temperature tau: the
quantitative version of the "typical influence is much smaller" conjecture in §7.

  python -m experiments.influence --model bert-base-uncased --taus 0.5 1 1.5 2 3 5 10 \
      --n 20 --chains 8 --burn 2000 --snap_every 500 --snaps 4 --out outputs/influence/bert
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from glauber.chain import run_chains
from glauber.models import MaskedLM, log_softmax_tau
from glauber.utils import load_seed_texts, out_dir, pick_device, save_args, set_seed

SEED_SENTENCE = "The overnight train rattled through the mountains as thunder echoed across the valley."


@torch.no_grad()
def influence_matrix(lm: MaskedLM, ids: torch.Tensor, positions, taus, m_top=4, m_rand=2, gen=None):
    """ids [L] -> {tau: c [n, n]} for a single state."""
    n = len(positions)
    base = lm.site_logits(ids[None], positions)[0]                  # [n, V]
    lp_base = {t: log_softmax_tau(base, t).exp() for t in taus}
    c = {t: np.zeros((n, n)) for t in taus}
    allowed = torch.nonzero(lm.allowed).squeeze(-1)
    for jj, j in enumerate(positions):
        cur = int(ids[j])
        top = [b for b in torch.topk(base[jj], m_top + 1).indices.tolist() if b != cur][:m_top]
        rnd = allowed[torch.randint(len(allowed), (m_rand,), device=lm.device, generator=gen)].tolist()
        alts = top + rnd
        Y = ids[None].repeat(len(alts), 1)
        Y[:, j] = torch.tensor(alts, device=lm.device)
        lg = lm.site_logits(Y, positions)                             # [m, n, V]
        for t in taus:
            q = log_softmax_tau(lg, t).exp()
            tvs = 0.5 * (q - lp_base[t][None]).abs().sum(-1)          # [m, n]
            c[t][:, jj] = tvs.max(0).values.cpu().numpy()
    for t in taus:
        np.fill_diagonal(c[t], 0.0)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 11.0])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--burn", type=int, default=2000)
    ap.add_argument("--snap_every", type=int, default=500)
    ap.add_argument("--snaps", type=int, default=4)
    ap.add_argument("--m_top", type=int, default=4)
    ap.add_argument("--m_rand", type=int, default=2)
    ap.add_argument("--seed_sentence", default=SEED_SENTENCE)
    ap.add_argument("--seed_source", default="msmarco")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.burn % args.snap_every == 0, "--burn must be a multiple of --snap_every"
    set_seed(args.seed)
    out = out_dir(args.out); save_args(args, out / "args.json")
    lm = MaskedLM(args.model, device=pick_device(args.device))
    gen = torch.Generator(device=lm.device).manual_seed(args.seed) if lm.device != "mps" else None
    rows = []

    def record(ids, positions, source, chain_tau, step, chain):
        c = influence_matrix(lm, ids, positions, args.taus, args.m_top, args.m_rand, gen)
        txt = lm.decode(ids, positions)
        for t, M in c.items():
            if chain_tau is not None and t != chain_tau:
                continue
            rows.append({"source": source, "chain_tau": chain_tau, "tau": t, "step": step, "chain": chain,
                         "alpha": float(M.sum(1).max()), "mean_row_sum": float(M.sum(1).mean()),
                         "max_c": float(M.max()), "text": txt})

    # (a) worst-case-style estimate on the coherent seed sentence (Appendix A.1)
    e = lm.encode(args.seed_sentence)
    record(e.ids, e.positions, "seed_sentence", None, 0, -1)
    print(pd.DataFrame(rows)[["tau", "alpha"]].to_string(index=False))

    # (b) states visited by the chain at temperature tau
    texts = load_seed_texts(args.seed_source, 200, seed=args.seed)
    pool = [c[: args.n] for c in (lm.content_ids(t) for t in texts) if len(c) >= args.n][: args.chains]
    for tau in args.taus:
        e = lm.frame(pool[0])
        ids = e.ids[None].repeat(len(pool), 1)
        ids[:, e.positions] = torch.tensor(pool, device=lm.device)
        for b in range(len(pool)):
            record(ids[b], e.positions, "msmarco_start", tau, 0, b)

        def cb(step, X):
            if step >= args.burn and (step - args.burn) % args.snap_every == 0:
                for b in range(X.shape[0]):
                    record(X[b], e.positions, "chain", tau, step, b)

        run_chains(lm, ids, e.positions, tau, args.burn + args.snap_every * (args.snaps - 1),
                   record_every=args.snap_every, callback=cb, seed=args.seed)
        d = pd.DataFrame(rows)
        d.to_csv(out / "influence.csv", index=False)
        s = d[(d.chain_tau == tau) & (d.source == "chain")]["alpha"]
        print(f"tau={tau}: chain-state alpha median={s.median():.3f} q90={s.quantile(0.9):.3f} "
              f"frac<1={(s < 1).mean():.2f}")

    d = pd.DataFrame(rows)
    summ = d.groupby(["source", "tau"])["alpha"].describe(percentiles=[0.1, 0.5, 0.9]).reset_index()
    summ.to_csv(out / "influence_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
