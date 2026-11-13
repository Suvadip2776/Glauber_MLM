"""Adapter sanity check — run this before long jobs on a NEW model family
(LLaDA / Dream / MDLM adapters follow the model cards but are not exercised by CI).

  python -m experiments.smoke --model GSAI-ML/LLaDA-8B-Base
Checks: encode/decode round trip, a fill-in-the-blank the model should get right,
Glauber steps run, and the maximal coupling keeps marginals (exact identity test).
"""
import argparse

import torch

from glauber.chain import glauber_step, maximal_coupling_sample
from glauber.models import MaskedLM, log_softmax_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    lm = MaskedLM(a.model, device=a.device)
    print(f"family={lm.family} device={lm.device} V={lm.V} mask_id={lm.mask_id} shift={lm.shift} "
          f"allowed={int(lm.allowed.sum())} prefix={lm._prefix} suffix={lm._suffix}")
    text = "The capital of France is Paris, and the capital of Japan is Tokyo."
    e = lm.encode(text)
    print("round trip:", lm.decode(e.ids, e.positions))
    k = [i for i in e.positions if "paris" in lm.tok.decode([int(e.ids[i])]).lower()]
    if k:
        lg = lm.logits_at(e.ids[None], torch.tensor(k[:1], device=lm.device))[0]
        top = [lm.tok.decode([t]) for t in torch.topk(lg, 5).indices.tolist()]
        print("top-5 for the masked 'Paris' slot:", top, "(should contain Paris)")
    ids = e.ids[None].repeat(2, 1)
    pos = torch.tensor(e.positions, device=lm.device)
    for _ in range(10):
        glauber_step(lm, ids, pos, 1.0)
    print("after 10 Glauber steps:", lm.decode(ids[0], e.positions))
    # coupling marginal check on a toy pair of distributions
    p = torch.tensor([[0.5, 0.3, 0.2]]).repeat(20000, 1); q = torch.tensor([[0.1, 0.3, 0.6]]).repeat(20000, 1)
    x, y = maximal_coupling_sample(p, q)
    print("coupling marginals:", torch.bincount(x, minlength=3) / 20000, torch.bincount(y, minlength=3) / 20000,
          "P(x=y)=", (x == y).float().mean().item(), "(expect 0.6)")


if __name__ == "__main__":
    main()
