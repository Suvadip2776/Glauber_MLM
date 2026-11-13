"""Exactly solvable MLM Glauber chains: a tractable setting with a computable
stationary distribution.

Construction. Take a real sentence, pick n "slot" positions, and give slot i a
candidate set V_i (the K most likely tokens under the single-mask conditional
in the original sentence, always containing the original token). All other
tokens are frozen context. The chain is Glauber dynamics of the MLM restricted
to X = V_1 x ... x V_n:

    p_i(a | x_-i) ∝ exp(s_i(a; x_-i) / tau),  a in V_i,

i.e. the MLM conditional conditioned on staying in the candidate set. Everything
below is exact: stationary distribution, worst-case TV mixing time, spectrum,
Dobrushin alpha(tau), conductance bottleneck, rectangle defects, and the
PLL-target MH chain of Goyal et al. (2022). Diagnostics used at scale
(coupling, embedding-hitting) can then be simulated on the very same chain and
compared against the exact t_mix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .models import MaskedLM


@dataclass
class SmallChain:
    sizes: Tuple[int, ...]
    scores: List[np.ndarray]      # scores[i][x] = s_i(x_i; x_-i), raw logits (tau=1)
    cand: List[List[int]]         # candidate token ids per slot
    orig: List[int]               # index of the original token in cand[i]
    template: List[int]           # full token ids of the template sentence
    slots: List[int]              # slot positions in template
    positions: List[int]          # all content positions (for decoding)
    meta: Dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.sizes)

    @property
    def S(self) -> int:
        return int(np.prod(self.sizes))

    @property
    def strides(self) -> np.ndarray:
        st = np.ones(self.n, dtype=np.int64)
        for i in range(self.n - 2, -1, -1):
            st[i] = st[i + 1] * self.sizes[i + 1]
        return st

    def state_tokens(self, flat: np.ndarray) -> np.ndarray:
        multi = np.stack(np.unravel_index(flat, self.sizes), -1)        # [m, n]
        return np.stack([np.asarray(self.cand[i])[multi[:, i]] for i in range(self.n)], -1)

    def texts(self, lm: MaskedLM, flat: Optional[np.ndarray] = None) -> List[str]:
        flat = np.arange(self.S) if flat is None else np.asarray(flat)
        toks = self.state_tokens(flat)
        base = torch.tensor(self.template)
        out = []
        for row in toks:
            ids = base.clone()
            ids[self.slots] = torch.tensor(row)
            out.append(lm.decode(ids, self.positions))
        return out


# ---------------------------------------------------------------------- build
@torch.no_grad()
def build_small_chain(lm: MaskedLM, text: str, n_slots: int, K: int, n_tokens: Optional[int] = None,
                      slot_mode: str = "spread", chunk: int = 512) -> SmallChain:
    enc = lm.encode(text, n_tokens)
    if enc is None:
        raise ValueError("seed text too short")
    P = enc.positions
    if n_slots > len(P):
        raise ValueError("more slots than content tokens")
    if slot_mode == "first":
        slot_idx = list(range(n_slots))
    else:
        slot_idx = sorted(set(np.linspace(0, len(P) - 1, n_slots).round().astype(int).tolist()))
    slots = [P[k] for k in slot_idx]
    base_lg = lm.site_logits(enc.ids[None], slots)[0]                   # [n, V]
    cand, orig = [], []
    for i, s in enumerate(slots):
        o = int(enc.ids[s])
        top = torch.topk(base_lg[i], K).indices.tolist()
        if o not in top:
            top[-1] = o
        cand.append(top)
        orig.append(top.index(o))
    sizes = tuple(len(c) for c in cand)
    n = len(slots)

    scores = []
    for i in range(n):
        other = [j for j in range(n) if j != i]
        osz = [sizes[j] for j in other]
        combos = np.array(list(np.ndindex(*osz)), dtype=np.int64).reshape(-1, len(other))
        C = combos.shape[0]
        cand_i = torch.tensor(cand[i], device=lm.device)
        out = []
        for s0 in range(0, C, chunk):
            cb = combos[s0:s0 + chunk]
            ids = enc.ids[None].repeat(len(cb), 1)
            for k, j in enumerate(other):
                ids[:, slots[j]] = torch.tensor(np.asarray(cand[j])[cb[:, k]], device=lm.device)
            pos = torch.full((len(cb),), slots[i], device=lm.device)
            out.append(lm.logits_at(ids, pos)[:, cand_i].cpu())
        arr = torch.cat(out).numpy().astype(np.float64).reshape(*osz, sizes[i])
        scores.append(np.moveaxis(arr, -1, i))
    return SmallChain(sizes, scores, cand, orig, enc.ids.tolist(), slots, list(P),
                      meta={"model": lm.name, "text": text, "K": K, "n_tokens": n_tokens})


# ------------------------------------------------------------------ kernels
def cond_probs(ch: SmallChain, tau: float) -> List[np.ndarray]:
    out = []
    for i, s in enumerate(ch.scores):
        z = s / tau
        z = z - z.max(axis=i, keepdims=True)
        e = np.exp(z)
        out.append(e / e.sum(axis=i, keepdims=True))
    return out


def log_pll(ch: SmallChain, probs: List[np.ndarray]) -> np.ndarray:
    """log of exp(PLL): sum_i log p_i(x_i | x_-i) for every flat state."""
    return sum(np.log(p.ravel()) for p in probs)


def transition_matrix(ch: SmallChain, probs: List[np.ndarray], kind: str = "glauber") -> sp.csr_matrix:
    """kind='glauber': the paper's chain.  kind='mh_pll': MH with Glauber
    proposal targeting pi ∝ exp(PLL) (Goyal et al. 2022)."""
    S, n, st = ch.S, ch.n, ch.strides
    idx = np.arange(S, dtype=np.int64)
    multi = np.unravel_index(idx, ch.sizes)
    rows, cols, vals = [], [], []
    lpi = log_pll(ch, probs) if kind == "mh_pll" else None
    for i in range(n):
        pf = probs[i].ravel()
        for a in range(ch.sizes[i]):
            nb = idx + (a - multi[i]) * st[i]
            q = pf[nb] / n
            if kind == "glauber":
                rows.append(idx); cols.append(nb); vals.append(q)
            else:
                off = nb != idx
                logr = lpi[nb] - lpi + np.log(pf) - np.log(pf[nb])
                acc = np.exp(np.minimum(0.0, logr))
                rows.append(idx[off]); cols.append(nb[off]); vals.append((q * acc)[off])
    P = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(S, S)).tocsr()
    if kind != "glauber":
        P = P + sp.diags(1.0 - np.asarray(P.sum(1)).ravel())
    return P.tocsr()


def stationary(P: sp.csr_matrix, polish: int = 200) -> Tuple[np.ndarray, float]:
    S = P.shape[0]
    A = (P.T - sp.identity(S, format="csr")).tocsr()
    A = sp.vstack([A[:-1], sp.csr_matrix(np.ones((1, S)))]).tocsc()
    b = np.zeros(S); b[-1] = 1.0
    mu = spla.spsolve(A, b)
    mu = np.clip(mu, 0, None); mu /= mu.sum()
    PT = P.T.tocsr()
    for _ in range(polish):
        mu = PT @ mu
    mu /= mu.sum()
    resid = float(np.abs(PT @ mu - mu).sum())
    return mu, resid


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


# ----------------------------------------------------------- exact mixing
def exact_mixing(P: sp.csr_matrix, mu: np.ndarray, eps: float = 0.25, max_doublings: int = 40,
                 device: str = "cpu", dtype=torch.float64):
    """Exact worst-case t_mix(eps) = min{t : max_x ||P^t(x,.) - mu||_TV <= eps}
    by repeated squaring + binary search (d(t) is non-increasing).
    Returns (t_mix or inf, [(t, d(t)) for t = 2^k])."""
    M = torch.tensor(P.toarray(), dtype=dtype, device=device)
    mu_t = torch.tensor(mu, dtype=dtype, device=device)[None]

    def d(X):
        return 0.5 * (X - mu_t).abs().sum(1).max().item()

    powers, curve = [M], [(1, d(M))]
    while curve[-1][1] > eps and len(powers) <= max_doublings:
        M = powers[-1] @ powers[-1]
        powers.append(M)
        curve.append((2 ** (len(powers) - 1), d(M)))
    if curve[-1][1] > eps:
        return math.inf, curve
    k = len(powers) - 1
    if k == 0:
        return 1, curve
    cur, t = powers[k - 1], 2 ** (k - 1)
    for j in range(k - 2, -1, -1):
        cand = cur @ powers[j]
        if d(cand) > eps:
            cur, t = cand, t + 2 ** j
    return t + 1, curve


def spectrum(P: sp.csr_matrix, k: int = 6):
    """Top eigenvalues by modulus, the second-eigenvector (for sweep cuts) and
    t_rel = 1/(1-|lambda_2|). Valid for non-reversible chains:
    t_mix(eps) >= (t_rel - 1) log(1/(2 eps))."""
    S = P.shape[0]
    try:
        vals, vecs = spla.eigs(P, k=min(k, S - 2), which="LM", maxiter=20000, tol=1e-10)
    except Exception:
        dense = P.toarray()
        vals, vecs = np.linalg.eig(dense)
    o = np.argsort(-np.abs(vals))
    vals, vecs = vals[o], vecs[:, o]
    lam2 = vals[1]
    t_rel = 1.0 / max(1e-300, 1 - abs(lam2))
    return vals, np.real(vecs[:, 1]), t_rel


def bottleneck(P: sp.csr_matrix, mu: np.ndarray, order_vec: np.ndarray):
    """Sweep cut along order_vec (both directions). Returns the set S with
    mu(S) <= 1/2 minimising Phi(S) = Q(S,S^c)/mu(S), giving the rigorous
    bound t_mix(1/4) >= 1/(4 Phi(S))."""
    F = sp.diags(mu) @ P
    Fr, Fc = F.tocsr(), F.tocsc()
    best = (math.inf, None)
    for order in (np.argsort(order_vec), np.argsort(-order_vec)):
        inS = np.zeros(len(mu), bool)
        Q = muS = 0.0
        for k, x in enumerate(order):
            r0, r1 = Fr.indptr[x], Fr.indptr[x + 1]
            ri, rd = Fr.indices[r0:r1], Fr.data[r0:r1]
            c0, c1 = Fc.indptr[x], Fc.indptr[x + 1]
            ci, cd = Fc.indices[c0:c1], Fc.data[c0:c1]
            Q += rd[(~inS[ri]) & (ri != x)].sum() - cd[inS[ci]].sum()
            inS[x] = True
            muS += mu[x]
            if muS > 0.5:
                break
            phi = Q / muS if muS > 0 else math.inf
            if phi < best[0]:
                best = (phi, order[:k + 1].copy())
    return best


def exact_alpha(ch: SmallChain, probs: List[np.ndarray]) -> Tuple[float, np.ndarray]:
    """Exact Dobrushin influence c_ij(tau) and alpha(tau) = max_i sum_j c_ij."""
    n = ch.n
    c = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = np.moveaxis(probs[i], [j, i], [-2, -1]).reshape(-1, ch.sizes[j], ch.sizes[i])
            tvm = 0.5 * np.abs(a[:, :, None, :] - a[:, None, :, :]).sum(-1)
            c[i, j] = tvm.max()
    return float(c.sum(1).max()), c


def rectangle_defects(ch: SmallChain, probs: List[np.ndarray], n_rect: int, rng, x_dist=None) -> np.ndarray:
    """|delta| for random rectangles (Section 4) on the exact chain."""
    lp = [np.log(p.ravel()) for p in probs]
    st = ch.strides
    out = []
    for _ in range(n_rect):
        x = rng.choice(ch.S, p=x_dist) if x_dist is not None else rng.integers(ch.S)
        m = np.array(np.unravel_index(x, ch.sizes))
        i, j = rng.choice(ch.n, 2, replace=False)
        a = rng.choice([v for v in range(ch.sizes[i]) if v != m[i]])
        b = rng.choice([v for v in range(ch.sizes[j]) if v != m[j]])
        y = x + (a - m[i]) * st[i]
        w = x + (b - m[j]) * st[j]
        z = y + (b - m[j]) * st[j]
        s_xy = lp[i][y] - lp[i][x]
        s_yz = lp[j][z] - lp[j][y]
        s_xw = lp[j][w] - lp[j][x]
        s_wz = lp[i][z] - lp[i][w]
        out.append((s_xy + s_yz) - (s_xw + s_wz))
    return np.abs(np.array(out))


def flow_asymmetry(P: sp.csr_matrix, mu: np.ndarray) -> float:
    """(1/2) sum_{x,y} |mu(x)P(x,y) - mu(y)P(y,x)|; zero iff reversible
    (a compatible joint would force this to be 0, Theorem 3)."""
    F = sp.diags(mu) @ P
    return 0.5 * float(abs(F - F.T).sum())


# -------------------------------------------------------------- simulation
class SmallSim:
    """Vectorised simulation of the exact chain, for diagnostics."""

    def __init__(self, ch: SmallChain, probs: List[np.ndarray]):
        self.ch = ch
        self.rows, self.ostrides = [], []
        for i in range(ch.n):
            self.rows.append(np.moveaxis(probs[i], i, -1).reshape(-1, ch.sizes[i]))
            osz = [ch.sizes[j] for j in range(ch.n) if j != i]
            st = np.ones(len(osz), np.int64)
            for k in range(len(osz) - 2, -1, -1):
                st[k] = st[k + 1] * osz[k + 1]
            self.ostrides.append(st)

    def _ctx(self, X, i):
        others = [j for j in range(self.ch.n) if j != i]
        return (X[:, others] * self.ostrides[i]).sum(1) if others else np.zeros(len(X), np.int64)

    @staticmethod
    def _draw(p, rng):
        u = rng.random(len(p))[:, None]
        return np.minimum((np.cumsum(p, 1) < u).sum(1), p.shape[1] - 1)

    def multi(self, flat):
        return np.stack(np.unravel_index(np.asarray(flat), self.ch.sizes), -1)

    def flat(self, X):
        return np.ravel_multi_index(X.T, self.ch.sizes)

    def step(self, X, rng):
        sites = rng.integers(self.ch.n, size=len(X))
        for i in range(self.ch.n):
            m = sites == i
            if m.any():
                X[m, i] = self._draw(self.rows[i][self._ctx(X[m], i)], rng)
        return X

    def coupled_step(self, X, Y, rng):
        sites = rng.integers(self.ch.n, size=len(X))
        for i in range(self.ch.n):
            m = np.where(sites == i)[0]
            if not len(m):
                continue
            p = self.rows[i][self._ctx(X[m], i)]
            q = self.rows[i][self._ctx(Y[m], i)]
            mn = np.minimum(p, q)
            r = mn.sum(1)
            same = rng.random(len(m)) < r
            a = np.empty(len(m), np.int64); b = np.empty(len(m), np.int64)
            if same.any():
                s = self._draw(mn[same] / r[same, None], rng)
                a[same] = s; b[same] = s
            d = ~same
            if d.any():
                rp = np.clip(p[d] - mn[d], 0, None); rq = np.clip(q[d] - mn[d], 0, None)
                a[d] = self._draw(rp / rp.sum(1, keepdims=True), rng)
                b[d] = self._draw(rq / rq.sum(1, keepdims=True), rng)
            X[m, i] = a; Y[m, i] = b
        return X, Y

    def coupling_times(self, x0, y0, t_max, rng):
        X, Y = self.multi(x0), self.multi(y0)
        T = np.full(len(X), np.inf)
        T[(X == Y).all(1)] = 0
        for t in range(1, t_max + 1):
            live = ~np.isfinite(T)
            if not live.any():
                break
            Xl, Yl = self.coupled_step(X[live], Y[live], rng)
            X[live], Y[live] = Xl, Yl
            met = (Xl == Yl).all(1)
            idx = np.where(live)[0][met]
            T[idx] = t
        return T

    def hitting_times(self, x0, emb, R, t_max, rng, every: int = 1):
        """First t with 1 - cos(emb[x_t], emb[x_0]) >= R (per threshold)."""
        X = self.multi(x0)
        e0 = emb[np.asarray(x0)]
        R = np.atleast_1d(R)
        T = np.full((len(X), len(R)), np.inf)
        for t in range(1, t_max + 1):
            self.step(X, rng)
            if t % every:
                continue
            d = 1 - (emb[self.flat(X)] * e0).sum(1)
            for k, r in enumerate(R):
                hit = (d >= r) & ~np.isfinite(T[:, k])
                T[hit, k] = t
            if np.isfinite(T).all():
                break
        return T
