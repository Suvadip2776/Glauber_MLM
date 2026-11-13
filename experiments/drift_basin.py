"""Slow-mixing certificates for metastable basins (Section 5.4).

Basin defined by a reference sequence r of length N and agreement count
A(x) = #{i : x_i = r_i}:   B_theta = {x : A(x) >= theta N}.
  --ref_token '!'      r = (!,...,!)  (the paper's basin)
  --ref_token '.'      etc.
  --ref_text  "..."    r = a natural trap state (e.g. from experiments/traps.py):
                       B is then a Hamming ball around a natural metastable sentence.

Subcommands
  drift   E[A(x_{t+1}) - A(x_t) | x] = delta_tau(x) on EVERY agreement level of a band,
          with fillers drawn 'random' | 'text' | 'model' | 'adversarial' (hill-climbing
          to MINIMISE the drift). Uniform positive drift on a band [a, b] with unit
          jumps is exactly the hypothesis of the negative-drift theorem
          (Hajek 1982; Oliveto & Witt 2011), giving exp(Omega(b - a)) escape time.
  escape  simulate chains started inside B; time until A(x_t) < theta_out N.
  entry   simulate chains from natural text; time until A(x_t) >= theta_out N.
  certify combine escape+entry into a lower bound that needs NO knowledge of mu(B):
            with E = {A >= theta_out N}, for x in B and natural y,
            ||P^t(x,.) - P^t(y,.)||_TV >= P^t(x,E) - P^t(y,E)
                                       >= 1 - P(T_esc(x) <= t) - P(T_ent(y) <= t),
            and d(t) >= dbar(t)/2, so t_mix(1/4) > t whenever
            P(T_esc <= t) + P(T_ent <= t) < 1/2  (we use Clopper-Pearson upper bounds).

  python -m experiments.drift_basin drift  --ref_token '!' --N 100 --taus 0.5 1 \
         --levels 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1 --fill model --states 100 --out outputs/drift/bang
  python -m experiments.drift_basin escape --ref_token '!' --Ns 10 20 40 60 80 100 --tau 1 \
         --theta 0.9 --theta_out 0.5 --chains 64 --steps 20000 --out outputs/drift/bang
  python -m experiments.drift_basin entry  --ref_token '!' --Ns 100 --tau 1 --theta_out 0.5 \
         --chains 64 --steps 20000 --out outputs/drift/bang
  python -m experiments.drift_basin certify --out outputs/drift/bang
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glauber.chain import glauber_step
from glauber.metrics import clopper_pearson_upper
from glauber.models import MaskedLM, log_softmax_tau
from glauber.utils import load_seed_texts, out_dir, pick_device, save_args, save_json, set_seed


# ----------------------------------------------------------------- helpers
def reference(lm: MaskedLM, args, N=None):
    if args.ref_text:
        c = lm.content_ids(args.ref_text)
        return torch.tensor(c, device=lm.device)
    return torch.full((N,), lm.token_id(args.ref_token), device=lm.device)


def frame_batch(lm: MaskedLM, content: torch.Tensor):
    """content [B, N] -> ids [B, L], positions"""
    e = lm.frame(content[0].tolist())
    ids = e.ids[None].repeat(content.shape[0], 1)
    ids[:, e.positions] = content
    return ids, e.positions


@torch.no_grad()
def ref_logprob(lm, ids, positions, ref, taus):
    """log p_tau(r_i | x_-i) for all sites, rows, taus -> {tau: [B, N]}"""
    lg = lm.site_logits(ids, positions)                             # [B, N, V]
    idx = ref[None, :, None].expand(ids.shape[0], -1, 1)
    return {t: log_softmax_tau(lg, t).gather(-1, idx).squeeze(-1) for t in taus}


def drift_from_lp(lp: torch.Tensor, agree: torch.Tensor) -> torch.Tensor:
    """delta(x) = (1/N)[sum_{i not agree} p(r_i|.) - sum_{i agree} (1 - p(r_i|.))]"""
    p = lp.exp()
    return torch.where(agree, p - 1, p).mean(-1)


def random_fillers(lm, B, N, ref, gen):
    allowed = torch.nonzero(lm.allowed).squeeze(-1)
    f = allowed[torch.randint(len(allowed), (B, N), device=lm.device, generator=gen)]
    clash = f == ref[None]
    while clash.any():
        f[clash] = allowed[torch.randint(len(allowed), (int(clash.sum()),), device=lm.device, generator=gen)]
        clash = f == ref[None]
    return f


def text_fillers(lm, B, N, ref, texts, rng):
    pool = [c[:N] for c in (lm.content_ids(t) for t in texts) if len(c) >= N]
    f = torch.tensor([pool[rng.integers(len(pool))] for _ in range(B)], device=lm.device)
    return torch.where(f == ref[None], (ref[None] + 1) % lm.V, f)


@torch.no_grad()
def model_fillers(lm, content, agree, ref, tau, sweeps, gen):
    """Resample the non-reference sites from the model (forbidding r_i) for a few
    sweeps with the reference sites frozen: the most plausible contaminants."""
    ids, pos = frame_batch(lm, content)
    pos_t = torch.tensor(pos, device=lm.device)
    B, N = content.shape
    for _ in range(sweeps * N):
        k = torch.randint(N, (B,), device=lm.device, generator=gen)
        free = ~agree[torch.arange(B), k]
        if not free.any():
            break
        rows = torch.nonzero(free).squeeze(-1)
        p = pos_t[k[rows]]
        lg = lm.logits_at(ids[rows], p)
        lg[torch.arange(len(rows)), ref[k[rows]]] = float("-inf")
        ids[rows, p] = torch.multinomial(log_softmax_tau(lg, tau).exp(), 1, generator=gen).squeeze(-1)
    return ids[:, pos]


def states_at_level(lm, k, N, ref, B, fill, texts, rng, gen, tau):
    agree = torch.zeros(B, N, dtype=torch.bool, device=lm.device)
    for b in range(B):
        agree[b, torch.tensor(rng.choice(N, k, replace=False), dtype=torch.long)] = True
    if fill == "text":
        f = text_fillers(lm, B, N, ref, texts, rng)
    else:
        f = random_fillers(lm, B, N, ref, gen)
    content = torch.where(agree, ref[None].expand(B, -1), f)
    if fill in ("model", "adversarial") and k < N:
        content = model_fillers(lm, content, agree, ref, tau, sweeps=2, gen=gen)
    return content, agree


@torch.no_grad()
def adversarial(lm, content, agree, ref, tau, iters, props, top_m, gen, rng):
    """Greedy local search that minimises delta_tau at fixed agreement level:
    moves = replace a filler by one of the model's top-m tokens (≠ r_i) or a
    random token, or swap a filler/reference location."""
    B, N = content.shape
    best_c, best_a = content.clone(), agree.clone()
    ids, pos = frame_batch(lm, best_c)
    best = drift_from_lp(ref_logprob(lm, ids, pos, ref, [tau])[tau], best_a)
    pos_t = torch.tensor(pos, device=lm.device)
    for _ in range(iters):
        cand_c = best_c.repeat_interleave(props, 0)
        cand_a = best_a.repeat_interleave(props, 0)
        for r in range(cand_c.shape[0]):
            fill_idx = torch.nonzero(~cand_a[r]).squeeze(-1)
            if len(fill_idx) == 0:
                continue
            if rng.random() < 0.3:          # move a contaminant
                ref_idx = torch.nonzero(cand_a[r]).squeeze(-1)
                if len(ref_idx):
                    i = int(fill_idx[rng.integers(len(fill_idx))]); j = int(ref_idx[rng.integers(len(ref_idx))])
                    cand_c[r, j], cand_c[r, i] = cand_c[r, i].clone(), ref[i]
                    cand_a[r, i], cand_a[r, j] = True, False
                    continue
            i = int(fill_idx[rng.integers(len(fill_idx))])
            ids_r, _ = frame_batch(lm, cand_c[r:r + 1])
            lg = lm.logits_at(ids_r, pos_t[i:i + 1])[0]
            lg[ref[i]] = float("-inf")
            choices = torch.topk(lg, top_m).indices
            cand_c[r, i] = choices[rng.integers(top_m)]
        ids, _ = frame_batch(lm, cand_c)
        d = drift_from_lp(ref_logprob(lm, ids, pos, ref, [tau])[tau], cand_a).view(B, props)
        dmin, arg = d.min(1)
        better = dmin < best
        for b in torch.nonzero(better).squeeze(-1).tolist():
            best[b] = dmin[b]
            best_c[b] = cand_c[b * props + arg[b]]
            best_a[b] = cand_a[b * props + arg[b]]
    return best_c, best_a


# --------------------------------------------------------------- commands
def cmd_drift(args):
    set_seed(args.seed)
    out = out_dir(args.out); save_args(args, out / "args_drift.json")
    lm = MaskedLM(args.model, device=pick_device(args.device))
    ref = reference(lm, args, args.N)
    N = len(ref)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=lm.device).manual_seed(args.seed) if lm.device != "mps" else None
    texts = load_seed_texts(args.seed_source, 300, seed=args.seed) if args.fill == "text" else []
    rows, examples = [], []
    for lev in args.levels:
        k = int(round(lev * N))
        for s0 in range(0, args.states, args.batch):
            B = min(args.batch, args.states - s0)
            content, agree = states_at_level(lm, k, N, ref, B, args.fill, texts, rng, gen, args.taus[0])
            if args.fill == "adversarial" and k < N:
                content, agree = adversarial(lm, content, agree, ref, args.taus[0], args.adv_iters,
                                             args.adv_props, args.adv_top_m, gen, rng)
            ids, pos = frame_batch(lm, content)
            lps = ref_logprob(lm, ids, pos, ref, args.taus)
            for t, lp in lps.items():
                d = drift_from_lp(lp, agree).tolist()
                rows += [{"level": lev, "k": k, "N": N, "tau": t, "fill": args.fill, "drift": v} for v in d]
            if s0 == 0:
                examples.append({"level": lev, "text": lm.decode(ids[0], pos)})
        print(f"level {lev:.2f}: done")
    df = pd.DataFrame(rows)
    tag = f"drift_{args.fill}_N{N}"
    df.to_csv(out / f"{tag}.csv", index=False)
    summ = (df.groupby(["tau", "level"])["drift"]
              .agg(min="min", q05=lambda x: x.quantile(0.05), median="median",
                   frac_positive=lambda x: (x > 0).mean(), n="count").reset_index())
    summ.to_csv(out / f"{tag}_summary.csv", index=False)
    save_json(examples, out / f"{tag}_examples.json")
    print(summ.to_string(index=False))


@torch.no_grad()
def simulate(lm, content, ref, tau, steps, thr, direction, seed, log_every=100):
    """direction='down': first time A < thr (escape); 'up': first time A >= thr (entry)."""
    ids, pos = frame_batch(lm, content)
    pos_t = torch.tensor(pos, device=lm.device)
    gen = torch.Generator(device=lm.device).manual_seed(seed) if lm.device != "mps" else None
    B = ids.shape[0]
    T = torch.full((B,), float("inf"))
    extreme = (ids[:, pos] == ref[None]).sum(-1).float().cpu()
    traj = []
    for t in range(1, steps + 1):
        glauber_step(lm, ids, pos_t, tau, gen)
        A = (ids[:, pos] == ref[None]).sum(-1).float().cpu()
        extreme = torch.minimum(extreme, A) if direction == "down" else torch.maximum(extreme, A)
        hit = (A < thr) if direction == "down" else (A >= thr)
        new = hit & torch.isinf(T)
        T[new] = t
        if t % log_every == 0:
            traj.append(A.mean().item())
    return T.numpy(), extreme.numpy(), traj, lm.decode_batch(ids[:3], pos)


def cmd_escape(args):
    set_seed(args.seed)
    out = out_dir(args.out); save_args(args, out / "args_escape.json")
    lm = MaskedLM(args.model, device=pick_device(args.device))
    gen = torch.Generator(device=lm.device).manual_seed(args.seed) if lm.device != "mps" else None
    rng = np.random.default_rng(args.seed)
    res = []
    for N in (args.Ns if not args.ref_text else [None]):
        ref = reference(lm, args, N)
        N = len(ref)
        if args.start == "full":
            content = ref[None].repeat(args.chains, 1)
        else:  # boundary states with model fillers
            content, _ = states_at_level(lm, math.ceil(args.theta * N), N, ref, args.chains, "model", [], rng,
                                         gen, args.tau)
        thr = math.ceil(args.theta_out * N)
        T, amin, traj, final = simulate(lm, content, ref, args.tau, args.steps, thr, "down", args.seed)
        r = {"N": N, "tau": args.tau, "theta": args.theta, "theta_out": args.theta_out, "steps": args.steps,
             "chains": args.chains, "start": args.start, "escape_times": T.tolist(),
             "frac_escaped": float(np.isfinite(T).mean()), "min_agree_frac": (amin / N).tolist(),
             "mean_agree_traj": traj, "final_examples": final}
        res.append(r)
        print(f"N={N}: escaped {r['frac_escaped']:.2f} within {args.steps}; "
              f"median escape={np.median(T)}; min agree frac (median)={np.median(amin / N):.2f}")
        save_json(res, out / f"escape_tau{args.tau}.json")


def cmd_entry(args):
    set_seed(args.seed)
    out = out_dir(args.out); save_args(args, out / "args_entry.json")
    lm = MaskedLM(args.model, device=pick_device(args.device))
    texts = load_seed_texts(args.seed_source, max(args.chains * 3, 200), seed=args.seed)
    res = []
    for N in (args.Ns if not args.ref_text else [None]):
        ref = reference(lm, args, N)
        N = len(ref)
        pool = [c[:N] for c in (lm.content_ids(t) for t in texts) if len(c) >= N][: args.chains]
        content = torch.tensor(pool, device=lm.device)
        thr = math.ceil(args.theta_out * N)
        T, amax, traj, final = simulate(lm, content, ref, args.tau, args.steps, thr, "up", args.seed)
        r = {"N": N, "tau": args.tau, "theta_out": args.theta_out, "steps": args.steps, "chains": len(pool),
             "entry_times": T.tolist(), "frac_entered": float(np.isfinite(T).mean()),
             "max_agree_frac": (amax / N).tolist(), "final_examples": final}
        res.append(r)
        print(f"N={N}: entered {r['frac_entered']:.2f}; max agree frac (max over chains)={amax.max() / N:.2f}")
        save_json(res, out / f"entry_tau{args.tau}.json")


def cmd_certify(args):
    out = Path(args.out)
    rows = []
    for ef in sorted(out.glob("escape_tau*.json")):
        tau = ef.stem.replace("escape_tau", "")
        nf = out / f"entry_tau{tau}.json"
        if not nf.exists():
            print(f"missing {nf}; run `entry` with the same tau / theta_out"); continue
        esc = {r["N"]: r for r in json.loads(ef.read_text())}
        ent = {r["N"]: r for r in json.loads(nf.read_text())}
        for N in sorted(set(esc) & set(ent)):
            e, n = esc[N], ent[N]
            if e["theta_out"] != n["theta_out"]:
                print(f"N={N}: theta_out mismatch"); continue
            Te, Tn = np.array(e["escape_times"]), np.array(n["entry_times"])
            horizon = min(e["steps"], n["steps"])
            best = 0
            for t in np.unique(np.linspace(1, horizon, 400).astype(int)):
                ue = clopper_pearson_upper(int((Te <= t).sum()), len(Te), args.alpha / 2)
                un = clopper_pearson_upper(int((Tn <= t).sum()), len(Tn), args.alpha / 2)
                if ue + un < 0.5:
                    best = int(t)
            rows.append({"tau": float(tau), "N": N, "theta_out": e["theta_out"],
                         "certified_tmix_quarter_gt": best, "horizon": horizon,
                         "confidence": 1 - args.alpha,
                         "note": "lower bound limited by simulation horizon" if best == horizon else ""})
    df = pd.DataFrame(rows)
    df.to_csv(out / "certificates.csv", index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="bert-base-uncased")
        g = p.add_mutually_exclusive_group()
        g.add_argument("--ref_token", default="!")
        g.add_argument("--ref_text", default=None)
        p.add_argument("--device", default="auto")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--seed_source", default="msmarco")
        p.add_argument("--out", required=True)

    d = sub.add_parser("drift"); common(d)
    d.add_argument("--N", type=int, default=100)
    d.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0])
    d.add_argument("--levels", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
    d.add_argument("--fill", choices=["random", "text", "model", "adversarial"], default="model")
    d.add_argument("--states", type=int, default=100)
    d.add_argument("--batch", type=int, default=8)
    d.add_argument("--adv_iters", type=int, default=30)
    d.add_argument("--adv_props", type=int, default=8)
    d.add_argument("--adv_top_m", type=int, default=20)
    d.set_defaults(func=cmd_drift)

    e = sub.add_parser("escape"); common(e)
    e.add_argument("--Ns", type=int, nargs="+", default=[10, 20, 40, 60, 80, 100])
    e.add_argument("--tau", type=float, default=1.0)
    e.add_argument("--theta", type=float, default=0.9)
    e.add_argument("--theta_out", type=float, default=0.5)
    e.add_argument("--start", choices=["full", "boundary"], default="boundary")
    e.add_argument("--chains", type=int, default=64)
    e.add_argument("--steps", type=int, default=20000)
    e.set_defaults(func=cmd_escape)

    n = sub.add_parser("entry"); common(n)
    n.add_argument("--Ns", type=int, nargs="+", default=[100])
    n.add_argument("--tau", type=float, default=1.0)
    n.add_argument("--theta_out", type=float, default=0.5)
    n.add_argument("--chains", type=int, default=64)
    n.add_argument("--steps", type=int, default=20000)
    n.set_defaults(func=cmd_entry)

    c = sub.add_parser("certify")
    c.add_argument("--out", required=True)
    c.add_argument("--alpha", type=float, default=0.05)
    c.set_defaults(func=cmd_certify)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
