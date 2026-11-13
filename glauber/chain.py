"""Batched Glauber dynamics, maximal coupling, and an MH-corrected variant.

State: ids [B, L] (all chains share the same layout / length); `positions` is
the list of updatable content positions (frozen prefixes = just leave them out).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch

from .models import MaskedLM, log_softmax_tau


def _sites(B: int, positions: torch.Tensor, gen: Optional[torch.Generator]) -> torch.Tensor:
    k = torch.randint(len(positions), (B,), device=positions.device, generator=gen)
    return positions[k]


def sample_from_logits(logits: torch.Tensor, tau: float, gen=None) -> torch.Tensor:
    probs = log_softmax_tau(logits, tau).exp()
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)


@torch.no_grad()
def glauber_step(lm: MaskedLM, ids: torch.Tensor, positions: torch.Tensor, tau: float,
                 gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """One Glauber update per chain (random site, resample from p_tau(.|x_-i)). In place."""
    pos = _sites(ids.shape[0], positions, gen)
    tok = sample_from_logits(lm.logits_at(ids, pos), tau, gen)
    ids[torch.arange(ids.shape[0], device=ids.device), pos] = tok
    return ids


@torch.no_grad()
def run_chains(lm: MaskedLM, ids: torch.Tensor, positions: Sequence[int], tau: float, steps: int,
               record_every: int = 0, callback: Optional[Callable[[int, torch.Tensor], None]] = None,
               seed: int = 0) -> torch.Tensor:
    """Run `steps` Glauber steps on every row of ids. callback(step, ids) is
    called at step 0 and every `record_every` steps."""
    gen = torch.Generator(device=ids.device).manual_seed(seed) if ids.device.type != "mps" else None
    pos_t = torch.tensor(list(positions), device=ids.device)
    if callback and record_every:
        callback(0, ids)
    for t in range(1, steps + 1):
        glauber_step(lm, ids, pos_t, tau, gen)
        if callback and record_every and t % record_every == 0:
            callback(t, ids)
    return ids


# --------------------------------------------------------------------------
# Maximal coupling (valid coupling: each marginal is exactly the Glauber kernel)
# --------------------------------------------------------------------------
def maximal_coupling_sample(p: torch.Tensor, q: torch.Tensor, gen=None):
    """p, q [B, V] probability rows. Returns (a, b) with a~p, b~q and
    P(a == b) = sum_k min(p_k, q_k) (the maximum possible)."""
    m = torch.minimum(p, q)
    r = m.sum(-1)
    u = torch.rand(p.shape[0], device=p.device, generator=gen)
    same = u < r
    a = torch.empty(p.shape[0], dtype=torch.long, device=p.device)
    b = torch.empty_like(a)
    if same.any():
        s = torch.multinomial(m[same] / r[same, None], 1, generator=gen).squeeze(-1)
        a[same] = s
        b[same] = s
    diff = ~same
    if diff.any():
        rp = (p[diff] - m[diff]).clamp_min(0)
        rq = (q[diff] - m[diff]).clamp_min(0)
        a[diff] = torch.multinomial(rp / rp.sum(-1, keepdim=True), 1, generator=gen).squeeze(-1)
        b[diff] = torch.multinomial(rq / rq.sum(-1, keepdim=True), 1, generator=gen).squeeze(-1)
    return a, b


@torch.no_grad()
def coupled_step(lm: MaskedLM, X: torch.Tensor, Y: torch.Tensor, positions: torch.Tensor, tau: float,
                 gen=None) -> None:
    """Synchronous maximal coupling (the coupling used in Theorem 5). In place."""
    B = X.shape[0]
    pos = _sites(B, positions, gen)
    lg = lm.logits_at(torch.cat([X, Y]), torch.cat([pos, pos]))
    P = log_softmax_tau(lg, tau).exp()
    a, b = maximal_coupling_sample(P[:B], P[B:], gen)
    ar = torch.arange(B, device=X.device)
    X[ar, pos] = a
    Y[ar, pos] = b


# --------------------------------------------------------------------------
# Pseudo-log-likelihood and MH correction (Goyal et al., ICLR 2022 style)
# --------------------------------------------------------------------------
@torch.no_grad()
def pll(lm: MaskedLM, ids: torch.Tensor, positions: Sequence[int], tau: float = 1.0) -> torch.Tensor:
    """sum_i log p_tau(x_i | x_-i) for every row. [B]"""
    lg = lm.site_logits(ids, positions)                   # [B, n, V]
    lp = log_softmax_tau(lg, tau)
    tgt = ids[:, list(positions)]
    return lp.gather(-1, tgt[..., None]).squeeze(-1).sum(-1)


@torch.no_grad()
def mh_pll_step(lm: MaskedLM, ids: torch.Tensor, cur_pll: torch.Tensor, positions: Sequence[int],
                tau: float, gen=None):
    """MH chain with Glauber proposal targeting pi(x) ∝ exp(PLL_tau(x)).
    Costs n+1 forward passes per step. Returns (ids, pll, accepted[B])."""
    B = ids.shape[0]
    pos_t = torch.tensor(list(positions), device=ids.device)
    pos = _sites(B, pos_t, gen)
    ar = torch.arange(B, device=ids.device)
    lp = log_softmax_tau(lm.logits_at(ids, pos), tau)
    a = torch.multinomial(lp.exp(), 1, generator=gen).squeeze(-1)
    old = ids[ar, pos]
    prop = ids.clone()
    prop[ar, pos] = a
    new_pll = pll(lm, prop, positions, tau)
    log_acc = (new_pll - cur_pll) + lp[ar, old] - lp[ar, a]
    acc = torch.log(torch.rand(B, device=ids.device, generator=gen)) < log_acc
    ids[acc] = prop[acc]
    cur_pll = torch.where(acc, new_pll, cur_pll)
    return ids, cur_pll, acc
