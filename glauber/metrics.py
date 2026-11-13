"""Statistics helpers: CIs, survival curves, trap detection, embeddings."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def bootstrap_ci(x, n_boot: int = 2000, ci: float = 95, seed: int = 0, stat=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(x, len(x))) for _ in range(n_boot)]
    return float(np.percentile(bs, (100 - ci) / 2)), float(np.percentile(bs, 100 - (100 - ci) / 2))


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1-alpha) upper confidence bound on a binomial proportion."""
    from scipy.stats import beta
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def survival(times: Sequence[float], grid: Sequence[int]) -> List[float]:
    """P(T > t) for each t in grid; censored runs should be passed as +inf."""
    t = np.asarray(times, float)
    return [float((t > g).mean()) for g in grid]


def censored_median(times: Sequence[float]) -> float:
    """Median that returns inf when more than half are censored (inf)."""
    t = np.sort(np.asarray(times, float))
    return float(t[len(t) // 2]) if len(t) else float("nan")


def detect_traps(emb: np.ndarray, steps: Sequence[int], window: int, thresh: float,
                 min_len: int) -> List[Dict]:
    """Trap = maximal run of recorded points whose mean cosine distance to the
    run's first point over a rolling window stays below `thresh`.
    emb [T, d] L2-normalised; steps [T] Glauber step of each row."""
    T = len(emb)
    steps = np.asarray(steps)
    if T == 0:
        return []
    stride = steps[1] - steps[0] if T > 1 else 1
    w = max(1, window // max(stride, 1))
    d_prev = 1 - (emb[1:] * emb[:-1]).sum(-1)              # step-to-step drift
    d_prev = np.concatenate([[0.0], d_prev])
    roll = np.convolve(d_prev, np.ones(w) / w, mode="same")
    calm = roll < thresh
    traps, s = [], None
    for t in range(T + 1):
        if t < T and calm[t]:
            s = t if s is None else s
        elif s is not None:
            if steps[t - 1] - steps[s] >= min_len:
                traps.append({"start_step": int(steps[s]), "end_step": int(steps[t - 1]),
                              "duration": int(steps[t - 1] - steps[s]), "start_idx": s, "end_idx": t - 1})
            s = None
    return traps


class Embedder:
    """Sentence embeddings (L2-normalised). Default matches the code release
    (jina-embeddings-v2-base-en, 768-d); all-MiniLM-L6-v2 is a fast option."""

    def __init__(self, name: str = "jinaai/jina-embeddings-v2-base-en", device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        from .utils import pick_device
        self.name = name
        self.model = SentenceTransformer(name, device=pick_device(device), trust_remote_code=True)

    def __call__(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        texts = [t if t.strip() else "." for t in texts]
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


def cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - (a * b).sum(-1)
