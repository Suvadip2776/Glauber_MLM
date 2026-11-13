"""Topic scorers: HF zero-shot NLI pipelines or an LLM judge via OpenRouter."""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

TOPICS = ["politics", "health", "sport", "science", "technology", "food", "finance", "environment", "art"]


class HFZeroShot:
    def __init__(self, model="facebook/bart-large-mnli", device=None, topics=TOPICS):
        from transformers import pipeline
        from .utils import pick_device
        dev = pick_device(device)
        self.topics = topics
        self.name = model
        self.pipe = pipeline("zero-shot-classification", model=model,
                             device=0 if dev == "cuda" else (dev if dev == "mps" else -1))

    def __call__(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        out = []
        clean = [t if str(t).strip() else "." for t in texts]
        for r in self.pipe(clean, candidate_labels=self.topics, multi_label=False, batch_size=batch_size):
            out.append(dict(zip(r["labels"], r["scores"])))
        return out


JUDGE_PROMPT = """You label the topic of short, possibly ungrammatical text fragments produced by a sampler.
Topics: {topics}, plus "incoherent" for fragments with no identifiable topic.
Return ONLY a JSON object mapping every label above (including "incoherent") to a probability; values sum to 1.

Text: <<<{text}>>>"""


class OpenRouterJudge:
    """LLM-judge topic scores; needs OPENROUTER_API_KEY. Cached on disk by text."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model=None, topics=TOPICS, cache_path="outputs/.openrouter_cache.jsonl", workers=8):
        import requests  # noqa: F401
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY not set (put it in glauber_mlm/.env)")
        self.model = model or os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
        self.name = f"openrouter:{self.model}"
        self.topics = topics
        self.labels = topics + ["incoherent"]
        self.workers = workers
        self.cache_path = Path(cache_path)
        self.cache = {}
        if self.cache_path.exists():
            for l in self.cache_path.read_text().splitlines():
                d = json.loads(l)
                if d.get("model") == self.model:
                    self.cache[d["text"]] = d["scores"]

    def _one(self, text: str) -> Dict[str, float]:
        import requests
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "user", "content": JUDGE_PROMPT.format(topics=", ".join(self.topics), text=text)}]}
        for attempt in range(5):
            try:
                r = requests.post(self.URL, headers={"Authorization": f"Bearer {self.key}"}, json=body, timeout=60)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                d = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
                s = {k: float(d.get(k, 0.0)) for k in self.labels}
                z = sum(s.values()) or 1.0
                return {k: v / z for k, v in s.items()}
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenRouter failed: {err}")

    def __call__(self, texts: List[str], batch_size: int = 0) -> List[Dict[str, float]]:
        todo = sorted({t for t in texts if t not in self.cache})
        if todo:
            with ThreadPoolExecutor(self.workers) as ex:
                for t, s in zip(todo, ex.map(self._one, todo)):
                    self.cache[t] = s
                    with open(self.cache_path, "a") as f:
                        f.write(json.dumps({"model": self.model, "text": t, "scores": s}) + "\n")
        return [self.cache[t] for t in texts]
