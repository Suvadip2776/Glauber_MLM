"""Rectangle incompatibility test (Section 4, Table 1, Figs 5-6), any model family.

Features:
  * any model family through the same adapter, including masked diffusion LMs, e.g.
    GSAI-ML/LLaDA-8B-Base, Dream-org/Dream-v0-Base-7B, kuleshov-group/mdlm-owt
  * bootstrap CIs; fraction of rectangles that are certificates (|delta| > 1e-3 >> float noise,
    calibrated by the AR control); |delta| vs influence and vs |i-j|.
    (Prop. 4 is deterministic: ONE nonzero delta already certifies incompatibility,
    so a p-value against delta=0 is not the right statistic.)
  * temperature: delta scales exactly as 1/tau (reported analytically, not re-run)
  * GPT-2 control computed from joint log-probs (delta = 0 up to float error)

  python -m experiments.rectangle --models bert-base-uncased roberta-base \
      answerdotai/ModernBERT-base --n_rect 300 --out outputs/rectangle
  python -m experiments.rectangle --models GSAI-ML/LLaDA-8B-Base --n_rect 300 --out outputs/rectangle
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import pandas as pd
import torch

from glauber.metrics import bootstrap_ci
from glauber.models import CausalLM, MaskedLM, log_softmax_tau
from glauber.utils import load_seed_texts, out_dir, pick_device, save_args, save_json, set_seed


@torch.no_grad()
def rectangles_mlm(lm: MaskedLM, texts, n_rect, n_tokens, top_k, rng: random.Random):
    rows = []
    encs = [e for e in (lm.encode(t, n_tokens) for t in texts) if e is not None]
    if not encs:
        raise ValueError("no seed text long enough")
    while len(rows) < n_rect:
        e = rng.choice(encs)
        i, j = rng.sample(e.positions, 2)
        x = e.ids.clone()
        A, B = int(x[i]), int(x[j])
        # replacements: uniform among the model's own top-k at that site (excluding current)
        lg = lm.logits_at(torch.stack([x, x]), torch.tensor([i, j], device=lm.device))
        topi = [t for t in torch.topk(lg[0], top_k).indices.tolist() if t != A]
        topj = [t for t in torch.topk(lg[1], top_k).indices.tolist() if t != B]
        A2, B2 = rng.choice(topi), rng.choice(topj)
        y = x.clone(); y[i] = A2
        w = x.clone(); w[j] = B2
        # contexts: x@i, y@j, x@j, w@i
        L = lm.logits_at(torch.stack([x, y, x, w]), torch.tensor([i, j, j, i], device=lm.device))
        lp = log_softmax_tau(L, 1.0)
        s_xy = (lp[0, A2] - lp[0, A]).item()
        s_yz = (lp[1, B2] - lp[1, B]).item()
        s_xw = (lp[2, B2] - lp[2, B]).item()
        s_wz = (lp[3, A2] - lp[3, A]).item()
        infl = max(abs((lp[0, A] - lp[3, A]).item()), abs((lp[2, B] - lp[1, B]).item()))
        rows.append({"delta": (s_xy + s_yz) - (s_xw + s_wz), "influence": infl,
                     "dist": abs(e.positions.index(i) - e.positions.index(j))})
    return rows


@torch.no_grad()
def rectangles_ar(clm: CausalLM, texts, n_rect, n_tokens, top_k, rng: random.Random):
    """AR control: edge log-ratio = difference of joint log-probs, so delta telescopes."""
    rows = []
    encs = [clm.tok(t, return_tensors="pt")["input_ids"][0][:n_tokens].to(clm.device) for t in texts]
    encs = [e for e in encs if len(e) == n_tokens]
    vocab = clm.model.config.vocab_size
    while len(rows) < n_rect:
        x = rng.choice(encs).clone()
        i, j = rng.sample(range(n_tokens), 2)
        A2, B2 = rng.randrange(vocab), rng.randrange(vocab)
        y = x.clone(); y[i] = A2
        w = x.clone(); w[j] = B2
        z = y.clone(); z[j] = B2
        lx, ly, lw, lz = clm.seq_logprob(torch.stack([x, y, w, z])).tolist()
        rows.append({"delta": ((ly - lx) + (lz - ly)) - ((lw - lx) + (lz - lw)), "influence": float("nan"),
                     "dist": abs(i - j)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["bert-base-uncased", "bert-large-uncased", "roberta-base",
                                                     "roberta-large", "answerdotai/ModernBERT-base"])
    ap.add_argument("--ar_control", default="gpt2", help="'none' to skip")
    ap.add_argument("--n_rect", type=int, default=300)
    ap.add_argument("--n_tokens", type=int, default=64)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed_source", default="msmarco")
    ap.add_argument("--n_texts", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    set_seed(args.seed)
    out = out_dir(args.out)
    save_args(args, out / "args.json")
    texts = load_seed_texts(args.seed_source, args.n_texts, seed=args.seed)

    summary = []
    todo = [(m, "mlm") for m in args.models] + ([(args.ar_control, "ar")] if args.ar_control != "none" else [])
    for name, kind in todo:
        rng = random.Random(args.seed)
        print(f"== {name}")
        if kind == "mlm":
            lm = MaskedLM(name, device=pick_device(args.device))
            rows = rectangles_mlm(lm, texts, args.n_rect, args.n_tokens, args.top_k, rng)
            del lm
        else:
            clm = CausalLM(name, device=pick_device(args.device))
            rows = rectangles_ar(clm, texts, args.n_rect, args.n_tokens, args.top_k, rng)
            del clm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        df = pd.DataFrame(rows)
        df["model"] = name
        df.to_csv(out / f"rect_{name.replace('/', '__')}.csv", index=False)
        a = df["delta"].abs().to_numpy()
        lo, hi = bootstrap_ci(a)
        rho = df[["influence", "delta"]].assign(delta=a).corr("spearman").iloc[0, 1] if kind == "mlm" else float("nan")
        s = {"model": name, "type": kind.upper(), "mean_abs_delta": float(a.mean()), "ci95": [lo, hi],
             "median_abs_delta": float(np.median(a)), "frac_abs_delta_gt_0.1": float((a > 0.1).mean()),
             "frac_certificates_abs_delta_gt_1e-3": float((a > 1e-3).mean()), "spearman_absdelta_influence": float(rho)}
        summary.append(s)
        print(s)
    save_json(summary, out / "summary.json")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print("Note: for temperature tau every edge log-ratio scales by 1/tau, so |delta_tau| = |delta_1|/tau.")


if __name__ == "__main__":
    main()
