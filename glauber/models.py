"""Uniform interface to "fill-in-the-mask" models.

Every experiment talks to a `MaskedLM`, which exposes the site conditionals
s_i(a; x_{-i}) (raw logits, tau = 1) for any batch of sequences. Supported families:

  family="mlm"   BERT / RoBERTa / ModernBERT / DeBERTa ... (AutoModelForMaskedLM)
  family="llada" LLaDA masked diffusion LMs, e.g. GSAI-ML/LLaDA-8B-Base   (trust_remote_code)
  family="dream" Dream masked diffusion LMs, e.g. Dream-org/Dream-v0-Base-7B (shifted logits)
  family="mdlm"  kuleshov-group/mdlm-owt (MDLM, Sahoo et al. 2024)        (EXPERIMENTAL)

The family is inferred from the model name; override with family=... .
The MDM adapters follow the public model cards; verify `python -m experiments.smoke
--model <name>` before launching long jobs on a new family.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .utils import pick_device

KNOWN_MASK_IDS = {"llada": 126336, "dream": 151666}


def infer_family(name: str) -> str:
    n = name.lower()
    if "llada" in n:
        return "llada"
    if "dream" in n:
        return "dream"
    if "mdlm" in n:
        return "mdlm"
    return "mlm"


@dataclass
class Encoded:
    ids: torch.Tensor          # [L] full sequence incl. special tokens
    positions: List[int]       # indices of the n content (updatable) tokens


class MaskedLM:
    def __init__(self, name: str, device: Optional[str] = None, family: Optional[str] = None,
                 dtype: Optional[str] = None, max_batch: int = 256):
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

        self.name = name
        self.family = family or infer_family(name)
        self.device = pick_device(device)
        self.max_batch = max_batch
        if dtype is None:
            dtype = "bfloat16" if (self.family in ("llada", "dream") and self.device == "cuda") else "float32"
        torch_dtype = getattr(torch, dtype)
        remote = self.family != "mlm"

        tok_name = "gpt2" if self.family == "mdlm" else name
        self.tok = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=remote)
        if self.family == "mlm":
            self.model = AutoModelForMaskedLM.from_pretrained(name, torch_dtype=torch_dtype)
        elif self.family == "mdlm":
            self.model = AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch_dtype)
        else:
            self.model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch_dtype)
        self.model.to(self.device).eval()

        # ---- mask token -------------------------------------------------------
        if self.family == "mlm":
            self.mask_id = self.tok.mask_token_id
        elif self.family == "mdlm":
            self.mask_id = self.model.config.vocab_size - 1  # MDLM appends [MASK] after GPT-2 vocab
        else:
            self.mask_id = self.tok.mask_token_id if self.tok.mask_token_id is not None else KNOWN_MASK_IDS[self.family]
        if self.mask_id is None:
            raise ValueError(f"{name}: no mask token")

        self.shift = self.family == "dream"  # Dream predicts token p from the output at p-1
        self.V = self._logit_dim()
        self.allowed = self._allowed_mask()
        self._prefix, self._suffix = self._special_frame()

    # ------------------------------------------------------------------ setup
    def _logit_dim(self) -> int:
        ids = torch.tensor([[self.mask_id] * 4], device=self.device)
        return self._forward(ids).shape[-1]

    def _allowed_mask(self) -> torch.Tensor:
        allowed = torch.zeros(self.V, dtype=torch.bool, device=self.device)
        allowed[: min(len(self.tok), self.V)] = True  # drop padded embedding rows
        for t in list(self.tok.all_special_ids) + [self.mask_id]:
            if t is not None and 0 <= t < self.V:
                allowed[t] = False
        for tokstr, t in self.tok.get_vocab().items():
            if t < self.V and (tokstr.startswith("[unused") or tokstr.startswith("<|reserved")):
                allowed[t] = False
        return allowed

    def _special_frame(self) -> Tuple[List[int], List[int]]:
        probe = "hello world"
        full = self.tok(probe)["input_ids"]
        bare = self.tok(probe, add_special_tokens=False)["input_ids"]
        prefix, suffix = [], []
        for s in range(len(full) - len(bare) + 1):
            if full[s:s + len(bare)] == bare:
                prefix, suffix = full[:s], full[s + len(bare):]
                break
        if self.shift and not prefix:  # need a token in front of the first content position
            bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
            prefix = [bos]
        return prefix, suffix

    # --------------------------------------------------------------- encoding
    def content_ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def frame(self, content: Sequence[int]) -> Encoded:
        ids = list(self._prefix) + list(content) + list(self._suffix)
        pos = list(range(len(self._prefix), len(self._prefix) + len(content)))
        return Encoded(torch.tensor(ids, device=self.device), pos)

    def encode(self, text: str, n_tokens: Optional[int] = None) -> Optional[Encoded]:
        c = self.content_ids(text)
        if n_tokens is not None:
            if len(c) < n_tokens:
                return None
            c = c[:n_tokens]
        return self.frame(c)

    def decode(self, ids: torch.Tensor, positions: Sequence[int]) -> str:
        return self.tok.decode(ids[list(positions)].tolist(), skip_special_tokens=True,
                               clean_up_tokenization_spaces=True).strip()

    def decode_batch(self, ids: torch.Tensor, positions: Sequence[int]) -> List[str]:
        return [self.decode(r, positions) for r in ids]

    def token_id(self, s: str) -> int:
        ids = self.tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"{s!r} is not a single token for {self.name}: {ids}")
        return ids[0]

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def _forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self.family == "mlm":
            return self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        if self.family == "mdlm":
            try:
                out = self.model(input_ids=ids, timesteps=torch.zeros(ids.shape[0], device=ids.device))
            except TypeError:
                out = self.model(ids)
            return out.logits if hasattr(out, "logits") else out  # MDLM returns logits directly
        return self.model(ids).logits

    @torch.no_grad()
    def logits_at(self, ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """Raw site scores s_i(.; x_{-i}) at position pos[b] of row b, with that
        position masked. ids [B, L], pos [B] -> float32 [B, V]; disallowed = -inf."""
        out = []
        for s in range(0, ids.shape[0], self.max_batch):
            x = ids[s:s + self.max_batch].clone()
            p = pos[s:s + self.max_batch]
            ar = torch.arange(x.shape[0], device=x.device)
            x[ar, p] = self.mask_id
            logits = self._forward(x)
            q = p - 1 if self.shift else p
            out.append(logits[ar, q].float())
        lg = torch.cat(out, 0)
        return lg.masked_fill(~self.allowed, float("-inf"))

    @torch.no_grad()
    def site_logits(self, ids: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        """For every row b and every site i in positions: s_i(.; x_{-i}).
        ids [B, L] -> [B, n, V]."""
        B, n = ids.shape[0], len(positions)
        rep = ids.repeat_interleave(n, 0)
        pos = torch.tensor(list(positions), device=ids.device).repeat(B)
        return self.logits_at(rep, pos).view(B, n, -1)


def log_softmax_tau(logits: torch.Tensor, tau: float) -> torch.Tensor:
    if tau == float("inf"):
        finite = torch.isfinite(logits)
        return torch.where(finite, -torch.log(finite.sum(-1, keepdim=True).float()), torch.full_like(logits, float("-inf")))
    return F.log_softmax(logits / tau, dim=-1)


class CausalLM:
    """Causal LM for the AR control (rectangle test) and perplexity of samples."""

    def __init__(self, name: str = "gpt2", device: Optional[str] = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name).to(self.device).eval()
        self.bos = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id

    @torch.no_grad()
    def seq_logprob(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B, L] (no BOS) -> log p(ids) [B] (BOS prepended)."""
        bos = torch.full((ids.shape[0], 1), self.bos, device=ids.device, dtype=ids.dtype)
        x = torch.cat([bos, ids], 1)
        lp = F.log_softmax(self.model(x).logits[:, :-1].float(), -1)
        return lp.gather(-1, x[:, 1:, None]).squeeze(-1).sum(-1)

    @torch.no_grad()
    def perplexity(self, texts: List[str], batch: int = 16) -> List[float]:
        out = []
        for s in range(0, len(texts), batch):
            for t in texts[s:s + batch]:
                ids = self.tok(t, return_tensors="pt")["input_ids"].to(self.device)
                if ids.shape[1] == 0:
                    out.append(float("nan"))
                    continue
                lp = self.seq_logprob(ids)[0].item()
                out.append(float(torch.exp(torch.tensor(-lp / ids.shape[1]))))
        return out
