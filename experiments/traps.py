"""Trap catalogue (Appendix A.1, Table 4) + restart escape times.

1. Run long chains, record text every `--every` steps, embed, detect traps
   (rolling mean step-to-step cosine drift < --thresh for >= --min_len steps).
2. For each trap's representative state, compute the per-site score margins
   Delta_i = s_i(x_i) - max_{a≠x_i} s_i(a); Delta* = min_i, mean Delta.
   "perfect" = every token is the argmax (Assumption 1 with Delta* > 0).
3. (--restarts R) Restart R fresh chains from each trap state and measure the time
   until Hamming distance from it reaches --escape_frac * n. This is a direct,
   repeatable escape-time measurement (vs a single observed dwell), to correlate
   with Delta*, mean Delta and the Theorem-8 bound.
The trap texts can be fed to drift_basin.py --ref_text to build natural basins.

  python -m experiments.traps --model bert-base-uncased --tau 1 --chains 30 --steps 10000 \
      --n 20 --every 10 --restarts 32 --restart_steps 5000 --out outputs/traps/bert
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch

from glauber.chain import glauber_step, run_chains
from glauber.metrics import Embedder, detect_traps
from glauber.models import MaskedLM
from glauber.utils import load_seed_texts, out_dir, pick_device, save_args, save_json, set_seed


@torch.no_grad()
def margins(lm, ids, positions):
    lg = lm.site_logits(ids[None], positions)[0]                    # [n, V]
    cur = ids[list(positions)]
    s_cur = lg.gather(-1, cur[:, None]).squeeze(-1)
    lg2 = lg.clone()
    lg2[torch.arange(len(positions)), cur] = float("-inf")
    gap = (s_cur - lg2.max(-1).values).cpu().numpy()
    return gap


@torch.no_grad()
def restart_escape(lm, ids, positions, tau, R, steps, frac, gen):
    X = ids[None].repeat(R, 1)
    ref = ids[list(positions)]
    pos_t = torch.tensor(positions, device=lm.device)
    thr = max(1, math.ceil(frac * len(positions)))
    T = np.full(R, np.inf)
    for t in range(1, steps + 1):
        glauber_step(lm, X, pos_t, tau, gen)
        h = (X[:, positions] != ref[None]).sum(-1).cpu().numpy()
        T[(h >= thr) & np.isinf(T)] = t
        if np.isfinite(T).all():
            break
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--chains", type=int, default=30)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--window", type=int, default=300)
    ap.add_argument("--thresh", type=float, default=0.12,
                    help="paper used 0.12 on mean cosine drift; recalibrate per embedder")
    ap.add_argument("--min_len", type=int, default=150)
    ap.add_argument("--restarts", type=int, default=0)
    ap.add_argument("--restart_steps", type=int, default=5000)
    ap.add_argument("--escape_frac", type=float, default=0.3)
    ap.add_argument("--seed_source", default="msmarco")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    set_seed(args.seed)
    out = out_dir(args.out); save_args(args, out / "args.json")
    dev = pick_device(args.device)
    lm = MaskedLM(args.model, device=dev)
    emb = Embedder(args.embedder, device=dev)
    gen = torch.Generator(device=lm.device).manual_seed(args.seed) if lm.device != "mps" else None

    texts = load_seed_texts(args.seed_source, args.chains * 3, seed=args.seed)
    pool = [c[: args.n] for c in (lm.content_ids(t) for t in texts) if len(c) >= args.n][: args.chains]
    e = lm.frame(pool[0])
    ids = e.ids[None].repeat(len(pool), 1)
    ids[:, e.positions] = torch.tensor(pool, device=lm.device)
    snaps = {"step": [], "ids": []}

    def cb(step, X):
        snaps["step"].append(step); snaps["ids"].append(X.clone().cpu())

    run_chains(lm, ids, e.positions, args.tau, args.steps, record_every=args.every, callback=cb, seed=args.seed)
    steps = np.array(snaps["step"])
    allids = torch.stack(snaps["ids"])                                # [T, B, L]
    rows, traps = [], []
    for b in range(allids.shape[1]):
        tx = lm.decode_batch(allids[:, b], e.positions)
        rows += [{"chain": b, "step": int(s), "text": t} for s, t in zip(steps, tx)]
        E = emb(tx)
        for tr in detect_traps(E, steps, args.window, args.thresh, args.min_len):
            mid = (tr["start_idx"] + tr["end_idx"]) // 2
            st = allids[mid, b].to(lm.device)
            g = margins(lm, st, e.positions)
            rec = {"chain": b, **tr, "text": lm.decode(st, e.positions), "mean_gap": float(g.mean()),
                   "min_gap": float(g.min()), "all_argmax": bool((g > 0).all()),
                   "type": "perfect" if (g > 0).all() else "approx",
                   "thm8_lb_steps": float(math.exp(max(g.min(), 0) / args.tau) / (4 * lm.allowed.sum().item())),
                   "token_ids": st[e.positions].tolist()}
            if args.restarts:
                T = restart_escape(lm, st, e.positions, args.tau, args.restarts, args.restart_steps,
                                   args.escape_frac, gen)
                rec.update({"restart_median_escape": float(np.median(T)),
                            "restart_frac_escaped": float(np.isfinite(T).mean())})
            traps.append(rec)
            print(f"chain {b} trap {tr['duration']} steps [{rec['type']}] Δ*={rec['min_gap']:.2f} "
                  f"Δ̄={rec['mean_gap']:.2f}: {rec['text'][:80]}")
    pd.DataFrame(rows).to_csv(out / "trajectories.csv", index=False)
    save_json(traps, out / "traps.json")
    df = pd.DataFrame(traps)
    if len(df):
        df.drop(columns=["token_ids"]).to_csv(out / "traps.csv", index=False)
        cols = [c for c in ["duration", "restart_median_escape"] if c in df]
        print(df[["min_gap", "mean_gap"] + cols].corr("spearman").to_string())
    print(f"{len(traps)} traps; wrote {out}")


if __name__ == "__main__":
    main()
