"""Semantic basins / political recurrence (Section 6.2), with null baselines.

Extends paper_code/political_drift_experiments.py:
  * generation and scoring are decoupled (generate once, score with several judges);
  * chains of equal token length are batched (fast);
  * null distributions scored by the SAME judge:  natural MS MARCO text of matching
    length ('natural') and uniformly random tokens ('random', the tau=inf chain);
  * summaries for ALL topics: occupancy (argmax label share), enrichment vs the
    natural-text null with chain-bootstrap CIs, dwell times and the dominant-topic
    transition matrix (basins), plus the paper's politics endpoint / ever-crossed stats
    at both the 1/9 baseline and a null-calibrated threshold (95th pct of null).

  python -m experiments.topics run --model bert-base-uncased --tau 1 --steps 10000 \
      --record_every 200 --replicates 1 --out outputs/topics/bert
  python -m experiments.topics score --out outputs/topics/bert --backend hf \
      --classifier facebook/bart-large-mnli
  python -m experiments.topics score --out outputs/topics/bert --backend openrouter   # optional
  python -m experiments.topics summarize --out outputs/topics/bert --scorer facebook/bart-large-mnli
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from glauber.chain import run_chains
from glauber.metrics import bootstrap_ci
from glauber.models import MaskedLM
from glauber.topics import TOPICS
from glauber.utils import ROOT, load_seed_texts, out_dir, pick_device, save_args, set_seed


def run(args):
    set_seed(args.seed)
    out = out_dir(args.out); save_args(args, out / "args_run.json")
    lm = MaskedLM(args.model, device=pick_device(args.device))
    seeds = json.loads(Path(args.seed_json).read_text())
    chains = []  # (topic, seed_id, rep, content ids)
    for topic, lst in seeds.items():
        for sid, t in enumerate(lst[: args.seeds_per_topic]):
            for rep in range(args.replicates):
                chains.append((topic, sid, rep, lm.content_ids(t)))
    groups = defaultdict(list)
    for c in chains:
        groups[len(c[3])].append(c)
    rows = []
    for L, grp in sorted(groups.items()):
        e = lm.frame(grp[0][3])
        ids = e.ids[None].repeat(len(grp), 1)
        ids[:, e.positions] = torch.tensor([g[3] for g in grp], device=lm.device)

        def cb(step, X):
            for g, t in zip(grp, lm.decode_batch(X, e.positions)):
                rows.append({"model": args.model, "tau": args.tau, "seed_topic": g[0], "seed_id": g[1],
                             "rep": g[2], "n_tokens": L, "step": step, "text": t})

        run_chains(lm, ids, e.positions, args.tau, args.steps, record_every=args.record_every, callback=cb,
                   seed=args.seed + L)
        pd.DataFrame(rows).to_csv(out / "texts.csv", index=False)
        print(f"length {L}: {len(grp)} chains done")

    # ---- null texts (same lengths as the chains)
    lengths = sorted(groups)
    nat = load_seed_texts(args.seed_source, args.n_null * 2, seed=args.seed + 1)
    nulls = []
    for k, t in enumerate(nat[: args.n_null]):
        c = lm.content_ids(t)[: lengths[k % len(lengths)]]
        nulls.append({"null": "natural", "text": lm.tok.decode(c, skip_special_tokens=True).strip()})
    allowed = torch.nonzero(lm.allowed).squeeze(-1).cpu()
    g = torch.Generator().manual_seed(args.seed)
    for k in range(args.n_null):
        c = allowed[torch.randint(len(allowed), (lengths[k % len(lengths)],), generator=g)].tolist()
        nulls.append({"null": "random", "text": lm.tok.decode(c, skip_special_tokens=True).strip()})
    pd.DataFrame(nulls).to_csv(out / "null_texts.csv", index=False)
    print(f"wrote {out/'texts.csv'} and null_texts.csv")


def score(args):
    out = Path(args.out)
    if args.backend == "hf":
        from glauber.topics import HFZeroShot
        clf = HFZeroShot(args.classifier, device=args.device)
    else:
        from glauber.topics import OpenRouterJudge
        clf = OpenRouterJudge(args.classifier if args.classifier != "facebook/bart-large-mnli" else None,
                              cache_path=out / ".openrouter_cache.jsonl")
    tag = clf.name.replace("/", "__").replace(":", "_")
    for fname in ["texts.csv", "null_texts.csv"]:
        df = pd.read_csv(out / fname, keep_default_na=False)
        uniq = sorted(set(df["text"]))
        scores = {}
        for s in range(0, len(uniq), 256):
            chunk = uniq[s:s + 256]
            for t, sc in zip(chunk, clf(chunk)):
                scores[t] = sc
            print(f"{fname}: {min(s + 256, len(uniq))}/{len(uniq)}")
        sc = pd.DataFrame([scores[t] for t in df["text"]]).add_suffix("_score")
        pd.concat([df.reset_index(drop=True), sc], axis=1).to_csv(out / f"scored_{tag}_{fname}", index=False)
    print(f"scorer tag: {tag}")


def summarize(args):
    out = Path(args.out)
    tag = args.scorer.replace("/", "__").replace(":", "_")
    df = pd.read_csv(out / f"scored_{tag}_texts.csv", keep_default_na=False)
    nl = pd.read_csv(out / f"scored_{tag}_null_texts.csv", keep_default_na=False)
    labs = [c[:-6] for c in df.columns if c.endswith("_score")]
    cols = [f"{l}_score" for l in labs]
    n0, m0 = len(df), len(nl)
    df, nl = df.dropna(subset=cols), nl.dropna(subset=cols)
    if len(df) < n0 or len(nl) < m0:
        print(f"dropped {n0 - len(df)} chain texts and {m0 - len(nl)} null texts the scorer could not score")
    df["label"] = df[[f"{l}_score" for l in labs]].idxmax(1).str[:-6]
    nl["label"] = nl[[f"{l}_score" for l in labs]].idxmax(1).str[:-6]
    df["chain"] = df["seed_topic"] + "_" + df["seed_id"].astype(str) + "_" + df["rep"].astype(str)
    final = df["step"].max()
    burn = df[df["step"] >= args.burn]
    res = {"scorer": args.scorer, "n_chains": int(df["chain"].nunique()), "final_step": int(final)}

    # occupancy & enrichment vs null (all topics)
    occ_rows = []
    null_nat = nl[nl["null"] == "natural"]["label"].value_counts(normalize=True)
    null_rnd = nl[nl["null"] == "random"]["label"].value_counts(normalize=True)
    per_chain = burn.groupby("chain")["label"].value_counts(normalize=True).unstack(fill_value=0)
    for l in labs:
        v = per_chain[l].to_numpy() if l in per_chain else np.zeros(len(per_chain))
        lo, hi = bootstrap_ci(v)
        occ_rows.append({"topic": l, "chain_occupancy": float(v.mean()), "ci95_lo": lo, "ci95_hi": hi,
                         "null_natural": float(null_nat.get(l, 0)), "null_random": float(null_rnd.get(l, 0)),
                         # NaN when the label never occurs in the natural null (ratio undefined)
                         "enrichment_vs_natural": float(v.mean() / null_nat[l]) if null_nat.get(l, 0) > 0 else float("nan")})
    occ = pd.DataFrame(occ_rows).sort_values("chain_occupancy", ascending=False)
    occ.to_csv(out / f"occupancy_{tag}.csv", index=False)
    print(occ.to_string(index=False))

    # politics: paper stats at 1/9 and at a null-calibrated threshold
    if "politics" in labs:
        thr_null = float(nl[nl["null"] == "natural"]["politics_score"].quantile(0.95))
        pol = []
        for topic, g in df.groupby("seed_topic"):
            st, fn = g[g.step == 0], g[g.step == final]
            mx = g.groupby("chain")["politics_score"].max()
            seed_col = f"{topic.lower()}_score"
            pol.append({"seed_topic": topic, "pol@0": st["politics_score"].mean(), "pol@final": fn["politics_score"].mean(),
                        "seed@0": st[seed_col].mean() if seed_col in g else np.nan,
                        "seed@final": fn[seed_col].mean() if seed_col in g else np.nan,
                        "ever>1/9": int((mx > 1 / 9).sum()), f"ever>null95({thr_null:.3f})": int((mx > thr_null).sum()),
                        "n_chains": int(g["chain"].nunique())})
        pd.DataFrame(pol).to_csv(out / f"politics_{tag}.csv", index=False)
        print(pd.DataFrame(pol).to_string(index=False))
        res["politics_null95_threshold"] = thr_null

    # basins: dwell times and dominant-topic transitions
    prev, nxt = [], []
    for _, g in burn.sort_values("step").groupby("chain"):
        lab = g["label"].tolist()
        prev += lab[:-1]; nxt += lab[1:]
    trans = pd.crosstab(pd.Series(prev, name="from"), pd.Series(nxt, name="to"), normalize="index")
    trans.to_csv(out / f"transitions_{tag}.csv")
    dwell = defaultdict(list)
    rec = int(df["step"].diff().replace(0, np.nan).dropna().abs().min() or 1)
    for _, g in burn.sort_values("step").groupby("chain"):
        lab = g["label"].tolist()
        run = 1
        for a, b in zip(lab, lab[1:] + [None]):
            if a == b:
                run += 1
            else:
                dwell[a].append(run * rec); run = 1
    res["mean_dwell_steps"] = {k: float(np.mean(v)) for k, v in dwell.items()}
    print("mean dwell (steps):", res["mean_dwell_steps"])
    (out / f"summary_{tag}.json").write_text(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="bert-base-uncased")
    r.add_argument("--tau", type=float, default=1.0)
    r.add_argument("--steps", type=int, default=10000)
    r.add_argument("--record_every", type=int, default=200)
    r.add_argument("--seed_json", default=str(ROOT / "configs" / "topic_seeds.json"))
    r.add_argument("--seeds_per_topic", type=int, default=10)
    r.add_argument("--replicates", type=int, default=1)
    r.add_argument("--n_null", type=int, default=500)
    r.add_argument("--seed_source", default="msmarco")
    r.add_argument("--device", default="auto")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--backend", choices=["hf", "openrouter"], default="hf")
    s.add_argument("--classifier", default="facebook/bart-large-mnli",
                   help="HF model, or OpenRouter model id with --backend openrouter")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=score)
    m = sub.add_parser("summarize")
    m.add_argument("--out", required=True)
    m.add_argument("--scorer", default="facebook/bart-large-mnli",
                   help="the scorer tag printed by `score` (e.g. openrouter:anthropic/claude-sonnet-4.5)")
    m.add_argument("--burn", type=int, default=2000)
    m.set_defaults(func=summarize)
    a = ap.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
