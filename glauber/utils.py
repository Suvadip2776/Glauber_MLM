"""Shared utilities: env loading, seeding, devices, IO, seed-text loading."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Optional[Path] = None) -> None:
    """Minimal .env loader (no python-dotenv dependency). Existing env vars win."""
    path = Path(path) if path else ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if v and k not in os.environ:
            os.environ[k] = v
    # transformers / huggingface_hub read HF_TOKEN automatically.


load_env()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device: Optional[str] = None) -> str:
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_jsonable(o):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)


def out_dir(base: str, *parts: str) -> Path:
    p = Path(base, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_args(args, path) -> None:
    d = dict(vars(args))
    d.pop("func", None)
    d["_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(d, path)


# --------------------------------------------------------------------------
# Seed texts
# --------------------------------------------------------------------------
def load_seed_texts(source: str = "msmarco", n_texts: int = 200, seed: int = 0,
                    min_chars: int = 200) -> List[str]:
    """Return a list of seed passages.

    source = "msmarco"          -> streams microsoft/ms_marco v2.1 (train)
    source = path/to/file.txt   -> one passage per line
    source = path/to/file.json  -> list of strings
    """
    if source != "msmarco":
        p = Path(source)
        if p.suffix == ".json":
            texts = json.loads(p.read_text())
        else:
            texts = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        rng = random.Random(seed)
        rng.shuffle(texts)
        return texts[:n_texts]

    path = _msmarco_cache(min_chars)
    texts = [l for l in path.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:n_texts]


def _msmarco_cache(min_chars: int = 200, max_passages: int = 50_000) -> Path:
    """Download the MS MARCO v2.1 validation parquet once (~1 file) and cache one
    passage per line under outputs/.cache/. Streaming was too slow in practice."""
    cache = ROOT / "outputs" / ".cache" / f"msmarco_val_min{min_chars}.txt"
    if cache.exists():
        return cache
    import pandas as pd
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("microsoft/ms_marco", "v2.1/validation-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(f, columns=["passages"])
    out: List[str] = []
    for ps in df["passages"]:
        for t in ps["passage_text"]:
            t = " ".join(str(t).split())
            if len(t) >= min_chars:
                out.append(t)
                break  # one passage per query, as before
        if len(out) >= max_passages:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(out))
    return cache


def chunks(xs: list, size: int) -> Iterable[list]:
    for i in range(0, len(xs), size):
        yield xs[i:i + size]
