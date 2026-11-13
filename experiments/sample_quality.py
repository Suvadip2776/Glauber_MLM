"""Is Glauber/editing dynamics a viable sampler?

Generates from scratch (random tokens) or from text with Glauber dynamics and tracks,
against the number of steps (and sweeps = steps / n), the fluency of samples
(perplexity under an independent causal LM), diversity (distinct-2 across chains)
and degeneracy (fraction of repeated tokens). Optionally runs the MH-corrected chain
targeting exp(PLL) (Goyal et al. 2022) side by side. References: AR samples of the
same length from the causal LM and natural MS MARCO text.

Works for any adapter family, so the same table can be produced for BERT, RoBERTa,
ModernBERT and masked diffusion LMs (LLaDA / Dream / MDLM).

  python -m experiments.sample_quality --model bert-base-uncased --n 32 --chains 32 \
      --tau 1 --steps 6400 --init random --scorer gpt2-large --out outputs/quality/bert
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from glauber.chain import glauber_step, mh_pll_step, pll
from glauber.models import CausalLM, MaskedLM
from glauber.utils import load_seed_texts, out_dir, pick_device, save_args, set_seed


def distinct2(texts):
    bg = [tuple(t.split()[i:i + 2]) for t in texts for i in range(max(0, len(t.split()) - 1))]
    return len(set(bg)) / max(1, len(bg))


def rep_frac(ids):
    ids = ids.tolist()
    return float(np.mean([1 - len(set(r)) / len(r) for r in ids]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--scorer", default="gpt2-large")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--chains", type=int, default=32)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=6400)
    ap.add_argument("--n_checkpoints", type=int, default=12)
    ap.add_argument("--init", choices=["random", "msmarco"], default="random")
    ap.add_argument("--seed_source", default="msmarco")
    ap.add_argument("--mh", action="store_true", help="also run the MH-PLL chain (n+1 forwards / step)")
    ap.add_argument("--mh_steps", type=int, default=None, help="default: same as --steps")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    set_seed(args.seed)
    out = out_dir(args.out); save_args(args, out / "args.json")
    dev = pick_device(args.device)
    lm = MaskedLM(args.model, device=dev)
    scorer = CausalLM(args.scorer, device=dev)
    gen = torch.Generator(device=lm.device).manual_seed(args.seed) if lm.device != "mps" else None

    # init
    allowed = torch.nonzero(lm.allowed).squeeze(-1)
    if args.init == "random":
        content = allowed[torch.randint(len(allowed), (args.chains, args.n), device=lm.device, generator=gen)]
    else:
        texts = load_seed_texts(args.seed_source, args.chains * 3, seed=args.seed)
        content = torch.tensor([c[: args.n] for c in (lm.content_ids(t) for t in texts) if len(c) >= args.n][: args.chains],
                               device=lm.device)
    e = lm.frame(content[0].tolist())
    ids = e.ids[None].repeat(content.shape[0], 1); ids[:, e.positions] = content
    pos_t = torch.tensor(e.positions, device=lm.device)
    ckpts = sorted(set(np.unique(np.geomspace(1, args.steps, args.n_checkpoints).astype(int)).tolist() + [0]))
    rows, samples = [], []

    def measure(kind, step, X):
        tx = lm.decode_batch(X, e.positions)
        ppl = scorer.perplexity(tx)
        rows.append({"sampler": kind, "step": step, "sweeps": step / args.n, "ppl_median": float(np.nanmedian(ppl)),
                     "ppl_mean_log": float(np.nanmean(np.log(ppl))), "distinct2": distinct2(tx),
                     "repeat_frac": rep_frac(X[:, e.positions])})
        samples.extend({"sampler": kind, "step": step, "text": t} for t in tx[:5])
        print(rows[-1])

    X = ids.clone()
    for t in range(args.steps + 1):
        if t > 0:
            glauber_step(lm, X, pos_t, args.tau, gen)
        if t in ckpts:
            measure("glauber", t, X)

    if args.mh:
        Y = ids.clone()
        cur = pll(lm, Y, e.positions, args.tau)
        acc_n = 0
        mh_steps = args.mh_steps or args.steps
        for t in range(mh_steps + 1):
            if t > 0:
                Y, cur, acc = mh_pll_step(lm, Y, cur, e.positions, args.tau, gen)
                acc_n += acc.float().mean().item()
            if t in ckpts:
                measure("mh_pll", t, Y)
                rows[-1]["mh_accept_rate"] = acc_n / max(1, t)

    # references
    nat = load_seed_texts(args.seed_source, args.chains * 3, seed=args.seed + 7)
    nat = [lm.tok.decode(c[: args.n], skip_special_tokens=True) for c in (lm.content_ids(t) for t in nat)
           if len(c) >= args.n][: args.chains]
    p = scorer.perplexity(nat)
    rows.append({"sampler": "natural_text", "step": 0, "ppl_median": float(np.nanmedian(p)),
                 "ppl_mean_log": float(np.nanmean(np.log(p))), "distinct2": distinct2(nat)})
    with torch.no_grad():
        bos = torch.full((args.chains, 1), scorer.bos, device=scorer.device)
        g = scorer.model.generate(bos, do_sample=True, max_new_tokens=args.n, min_new_tokens=args.n,
                                  top_k=0, pad_token_id=scorer.bos)
    ar = scorer.tok.batch_decode(g[:, 1:], skip_special_tokens=True)
    p = scorer.perplexity(ar)
    rows.append({"sampler": f"AR_{args.scorer}_ancestral", "step": args.n, "ppl_median": float(np.nanmedian(p)),
                 "ppl_mean_log": float(np.nanmean(np.log(p))), "distinct2": distinct2(ar),
                 "note": "self-scored: optimistic reference"})
    pd.DataFrame(rows).to_csv(out / "quality.csv", index=False)
    pd.DataFrame(samples).to_csv(out / "samples.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
