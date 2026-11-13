"""
Political drift experiments for MLM Glauber dynamics.

This script reproduces and extends the political drift section experiments:
1. Run Glauber dynamics on masked language models.
2. Score generated text with zero-shot topic classifiers.
3. Summarize endpoint political drift, transient political entry, seed-topic loss,
   and all-topic redistribution.
4. Generate plots for the paper and appendix.

Example:
    python political_drift_experiments.py run \
        --model_name bert-base-uncased \
        --model_label BERT \
        --classifier_model facebook/bart-large-mnli \
        --steps 10000 --record_every 200 --seeds_per_topic 10 \
        --output_dir outputs/bert_bart_10k

    python political_drift_experiments.py summarize \
        --checkpoint_csv outputs/bert_bart_10k/checkpoint_results.csv \
        --output_dir outputs/bert_bart_10k

    python political_drift_experiments.py plot \
        --checkpoint_csv outputs/bert_bart_10k/checkpoint_results.csv \
        --output_dir outputs/bert_bart_10k/figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

import matplotlib.pyplot as plt


CANDIDATE_TOPICS = [
    "politics",
    "health",
    "sport",
    "science",
    "technology",
    "food",
    "finance",
    "environment",
    "art",
]

BASELINE = 1.0 / len(CANDIDATE_TOPICS)

DEFAULT_SEED_GROUPS: Dict[str, List[str]] = {
    "Health": [
        "Doctors recommend eating vegetables and exercising regularly.",
        "The hospital staff monitored the patient after the surgery.",
        "A balanced diet and daily exercise can improve heart health.",
        "The nurse checked the patient's blood pressure and temperature.",
        "Vaccines help protect communities from infectious diseases.",
        "The clinic offers regular checkups for children and adults.",
        "Researchers studied how sleep affects the immune system.",
        "The doctor prescribed medicine to reduce the patient's fever.",
        "Physical therapy helped the athlete recover from the injury.",
        "Public health campaigns encourage people to wash their hands.",
    ],
    "Sport": [
        "The team scored three goals in the final minutes to win.",
        "The coach made several substitutions during the second half.",
        "The tennis player served an ace to win the match.",
        "Fans cheered loudly after the championship game ended.",
        "The runner trained every morning before the marathon.",
        "The basketball team practiced shooting and defense after school.",
        "The goalkeeper blocked a powerful shot in the final minute.",
        "The baseball player hit a home run into left field.",
        "The swimmer broke the record in the final race.",
        "The referee stopped the game after a foul was committed.",
    ],
    "Science": [
        "Researchers discovered a new species of deep sea fish.",
        "The experiment measured how plants respond to sunlight.",
        "Astronomers observed a distant galaxy through the telescope.",
        "The laboratory tested samples collected from the river.",
        "Scientists developed a model to explain the chemical reaction.",
        "The fossil record provides evidence about ancient animals.",
        "The physics class studied motion, energy, and force.",
        "A new satellite collected data about Earth's atmosphere.",
        "Biologists analyzed the behavior of insects in the forest.",
        "The microscope revealed tiny structures inside the cell.",
    ],
    "Food": [
        "The chef prepared a dish using fresh tomatoes and garlic.",
        "The bakery sold warm bread and sweet pastries every morning.",
        "She cooked rice with vegetables and spices for dinner.",
        "The restaurant served soup, salad, and grilled chicken.",
        "Fresh fruit and yogurt made a simple breakfast.",
        "The recipe called for butter, flour, sugar, and eggs.",
        "They roasted potatoes with herbs and olive oil.",
        "The kitchen smelled like cinnamon and baked apples.",
        "He sliced onions and peppers before making the sauce.",
        "The family shared pasta, bread, and soup at the table.",
    ],
    "Technology": [
        "The new smartphone has a faster processor and better camera.",
        "Engineers tested the software update before releasing it.",
        "The computer stored files on a secure cloud server.",
        "A new battery design improved the electric car's range.",
        "The website loaded quickly after the database was optimized.",
        "The robot used sensors to navigate through the room.",
        "Programmers fixed bugs in the mobile application.",
        "The wireless network connected laptops, phones, and tablets.",
        "The company designed a chip for artificial intelligence systems.",
        "The camera software improved image quality in low light.",
    ],
}


@dataclass
class RunConfig:
    model_name: str
    model_label: str
    classifier_model: str
    steps: int
    record_every: int
    temperature: float
    seeds_per_topic: int
    random_seed: int
    output_dir: str


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_string() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_seed_groups(path: Optional[str]) -> Dict[str, List[str]]:
    if path is None:
        return DEFAULT_SEED_GROUPS
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Seed JSON must be a dict mapping topic names to lists of seed texts.")
    return {str(k): list(v) for k, v in data.items()}


def build_allowed_token_mask(tokenizer, model, device: str) -> torch.Tensor:
    vocab_size = model.get_input_embeddings().num_embeddings
    allowed = torch.ones(vocab_size, dtype=torch.bool, device=device)

    for tid in tokenizer.all_special_ids:
        if tid is not None and 0 <= tid < vocab_size:
            allowed[tid] = False

    vocab = tokenizer.get_vocab()
    for token, tid in vocab.items():
        if 0 <= tid < vocab_size:
            # Avoid BERT unused tokens and obvious control tokens.
            if token.startswith("[unused"):
                allowed[tid] = False

    return allowed


def get_update_positions(input_ids: torch.Tensor, tokenizer) -> List[int]:
    special = set(tokenizer.all_special_ids)
    ids = input_ids[0].tolist()
    return [i for i, tid in enumerate(ids) if tid not in special]


def decode_ids(input_ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


@torch.no_grad()
def glauber_step(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    update_positions: List[int],
    tokenizer,
    model,
    allowed_token_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    pos = random.choice(update_positions)
    masked = input_ids.clone()
    masked[0, pos] = tokenizer.mask_token_id

    outputs = model(input_ids=masked, attention_mask=attention_mask)
    logits = outputs.logits[0, pos].clone()

    # Mask out special or unusable tokens.
    logits[~allowed_token_mask] = float("-inf")

    probs = torch.softmax(logits / temperature, dim=-1)
    sampled = torch.multinomial(probs, 1).item()
    input_ids[0, pos] = sampled
    return input_ids


def classify_topics(text: str, classifier) -> Dict[str, float]:
    text = str(text).strip()
    if not text:
        return {topic: BASELINE for topic in CANDIDATE_TOPICS}
    result = classifier(text, candidate_labels=CANDIDATE_TOPICS, multi_label=False)
    return dict(zip(result["labels"], result["scores"]))


def completed_chains_from_csv(path: Path) -> set[Tuple[str, int]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    max_step = df["step"].max()
    done = df[df["step"] == max_step][["seed_topic", "seed_id"]].drop_duplicates()
    return set((str(r.seed_topic), int(r.seed_id)) for r in done.itertuples(index=False))


def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_csv = output_dir / "checkpoint_results.csv"
    config_path = output_dir / "run_config.json"

    config = RunConfig(
        model_name=args.model_name,
        model_label=args.model_label,
        classifier_model=args.classifier_model,
        steps=args.steps,
        record_every=args.record_every,
        temperature=args.temperature,
        seeds_per_topic=args.seeds_per_topic,
        random_seed=args.random_seed,
        output_dir=str(output_dir),
    )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    set_global_seed(args.random_seed)
    dev = device_string()
    print(f"Device: {dev}")
    print(f"Loading MLM: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(dev)
    model.eval()

    if tokenizer.mask_token_id is None:
        raise ValueError(f"Tokenizer for {args.model_name} has no mask token.")

    allowed_token_mask = build_allowed_token_mask(tokenizer, model, dev)

    print(f"Loading classifier: {args.classifier_model}")
    classifier = pipeline(
        "zero-shot-classification",
        model=args.classifier_model,
        device=0 if dev == "cuda" else -1,
    )

    seed_groups = load_seed_groups(args.seed_json)
    completed = completed_chains_from_csv(checkpoint_csv) if args.resume else set()
    rows_buffer: List[dict] = []

    if checkpoint_csv.exists() and not args.resume:
        raise FileExistsError(
            f"{checkpoint_csv} already exists. Use --resume to append/skip completed chains, "
            "or choose a new output_dir."
        )

    start_time = time.time()
    topic_items = list(seed_groups.items())

    for seed_topic, seeds in topic_items:
        chosen = list(seeds)[: args.seeds_per_topic]
        for seed_id, seed_text in enumerate(tqdm(chosen, desc=f"{args.model_label} | {seed_topic}")):
            if (seed_topic, seed_id) in completed:
                print(f"Skipping completed chain: {seed_topic} seed {seed_id}")
                continue

            enc = tokenizer(seed_text, return_tensors="pt", truncation=True)
            input_ids = enc["input_ids"].to(dev)
            attention_mask = enc["attention_mask"].to(dev)
            update_positions = get_update_positions(input_ids, tokenizer)
            if not update_positions:
                raise ValueError(f"No update positions found for seed: {seed_text}")

            chain_rows: List[dict] = []

            for step in range(0, args.steps + 1):
                if step > 0:
                    input_ids = glauber_step(
                        input_ids,
                        attention_mask,
                        update_positions,
                        tokenizer,
                        model,
                        allowed_token_mask,
                        args.temperature,
                    )

                if step % args.record_every == 0:
                    text = decode_ids(input_ids, tokenizer)
                    scores = classify_topics(text, classifier)
                    row = {
                        "model_label": args.model_label,
                        "model_name": args.model_name,
                        "classifier_model": args.classifier_model,
                        "seed_topic": seed_topic,
                        "seed_id": seed_id,
                        "step": step,
                        "temperature": args.temperature,
                        "text": text,
                    }
                    for topic in CANDIDATE_TOPICS:
                        row[f"{topic}_score"] = float(scores[topic])
                    chain_rows.append(row)

            rows_buffer.extend(chain_rows)

            # Save after every chain to survive Colab disconnects.
            df_chain = pd.DataFrame(chain_rows)
            if checkpoint_csv.exists():
                df_chain.to_csv(checkpoint_csv, mode="a", header=False, index=False)
            else:
                df_chain.to_csv(checkpoint_csv, index=False)

            final_pol = chain_rows[-1]["politics_score"]
            max_pol = max(row["politics_score"] for row in chain_rows)
            print(
                f"Finished {seed_topic} seed {seed_id}: "
                f"final politics={final_pol:.3f}, max politics={max_pol:.3f}"
            )

    elapsed = (time.time() - start_time) / 60.0
    print(f"Run complete in {elapsed:.1f} minutes.")
    summarize_checkpoint_csv(str(checkpoint_csv), str(output_dir))
    plot_from_checkpoint_csv(str(checkpoint_csv), str(output_dir / "figures"))


def load_checkpoint_df(checkpoint_csv: str) -> pd.DataFrame:
    df = pd.read_csv(checkpoint_csv)
    required = {"seed_topic", "seed_id", "step", "politics_score", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in checkpoint CSV: {missing}")
    return df


def bootstrap_ci(values: np.ndarray, n_boot: int = 3000, ci: float = 95.0, seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(sample.mean())
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return float(lo), float(hi)


def summarize_checkpoint_csv(checkpoint_csv: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = load_checkpoint_df(checkpoint_csv)
    final_step = int(df["step"].max())
    topic_cols = [f"{t}_score" for t in CANDIDATE_TOPICS if f"{t}_score" in df.columns]

    start_df = df[df["step"] == 0].copy()
    final_df = df[df["step"] == final_step].copy()

    # Endpoint politics by seed topic.
    endpoint_rows = []
    for seed_topic, group in final_df.groupby("seed_topic"):
        start_group = start_df[start_df["seed_topic"] == seed_topic]
        vals = group["politics_score"].to_numpy()
        ci_lo, ci_hi = bootstrap_ci(vals)
        endpoint_rows.append(
            {
                "seed_topic": seed_topic,
                "n_chains": len(group),
                "politics_start_mean": start_group["politics_score"].mean(),
                "politics_final_mean": group["politics_score"].mean(),
                "politics_delta_mean": group["politics_score"].mean() - start_group["politics_score"].mean(),
                "politics_final_ci_low": ci_lo,
                "politics_final_ci_high": ci_hi,
                "final_crossed_count": int((group["politics_score"] > BASELINE).sum()),
                "final_crossed_rate": float((group["politics_score"] > BASELINE).mean()),
            }
        )
    endpoint_summary = pd.DataFrame(endpoint_rows).sort_values("seed_topic")
    endpoint_summary.to_csv(output / "endpoint_politics_summary.csv", index=False)

    # Maximum politics reached by each chain.
    idx = df.groupby(["seed_topic", "seed_id"])["politics_score"].idxmax()
    peak_df = df.loc[idx].copy().reset_index(drop=True)
    peak_df = peak_df.rename(columns={"step": "peak_step", "politics_score": "peak_politics_score", "text": "peak_text"})
    peak_df.to_csv(output / "peak_politics_by_chain.csv", index=False)

    max_rows = []
    for seed_topic, group in peak_df.groupby("seed_topic"):
        max_rows.append(
            {
                "seed_topic": seed_topic,
                "n_chains": len(group),
                "mean_max_politics": group["peak_politics_score"].mean(),
                "median_max_politics": group["peak_politics_score"].median(),
                "ever_crossed_count": int((group["peak_politics_score"] > BASELINE).sum()),
                "ever_crossed_rate": float((group["peak_politics_score"] > BASELINE).mean()),
                "max_peak": group["peak_politics_score"].max(),
            }
        )
    max_summary = pd.DataFrame(max_rows).sort_values("seed_topic")
    max_summary.to_csv(output / "max_politics_summary.csv", index=False)

    # Seed topic identity loss.
    seed_rows = []
    for seed_topic, group in final_df.groupby("seed_topic"):
        score_col = f"{seed_topic.lower()}_score"
        if score_col not in df.columns:
            continue
        start_group = start_df[start_df["seed_topic"] == seed_topic]
        seed_rows.append(
            {
                "seed_topic": seed_topic,
                "seed_score_start_mean": start_group[score_col].mean(),
                "seed_score_final_mean": group[score_col].mean(),
                "seed_score_delta_mean": group[score_col].mean() - start_group[score_col].mean(),
            }
        )
    pd.DataFrame(seed_rows).sort_values("seed_topic").to_csv(output / "seed_topic_identity_loss.csv", index=False)

    # Full nine-topic deltas from step 0 to final.
    merge_cols = ["seed_topic", "seed_id"] + topic_cols
    merged = start_df[merge_cols].merge(
        final_df[merge_cols],
        on=["seed_topic", "seed_id"],
        suffixes=("_start", "_final"),
    )
    delta_rows = []
    top_gain_rows = []
    for seed_topic, group in merged.groupby("seed_topic"):
        row = {"seed_topic": seed_topic}
        gains = []
        for topic in CANDIDATE_TOPICS:
            start_col = f"{topic}_score_start"
            final_col = f"{topic}_score_final"
            if start_col not in group.columns:
                continue
            delta = (group[final_col] - group[start_col]).mean()
            row[f"{topic}_delta"] = delta
            gains.append((topic, delta))
        delta_rows.append(row)
        gains_sorted = sorted(gains, key=lambda x: x[1], reverse=True)
        top_gain_rows.append(
            {
                "seed_topic": seed_topic,
                "top_gain_1": gains_sorted[0][0],
                "delta_1": gains_sorted[0][1],
                "top_gain_2": gains_sorted[1][0],
                "delta_2": gains_sorted[1][1],
                "top_gain_3": gains_sorted[2][0],
                "delta_3": gains_sorted[2][1],
                "politics_delta": row.get("politics_delta", float("nan")),
            }
        )
    pd.DataFrame(delta_rows).sort_values("seed_topic").to_csv(output / "all_topic_score_changes.csv", index=False)
    pd.DataFrame(top_gain_rows).sort_values("seed_topic").to_csv(output / "top_topic_gains.csv", index=False)

    # Final politics values for distribution plots and manual inspection.
    final_out = final_df[
        ["model_label", "model_name", "classifier_model", "seed_topic", "seed_id", "step", "politics_score", "text"]
    ].sort_values("politics_score")
    final_out.to_csv(output / "final_politics_by_chain.csv", index=False)

    print(f"Wrote summaries to: {output}")
    print("Endpoint politics summary:")
    print(endpoint_summary)
    print("Max politics summary:")
    print(max_summary)


def plot_from_checkpoint_csv(checkpoint_csv: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = load_checkpoint_df(checkpoint_csv)
    final_step = int(df["step"].max())
    final_df = df[df["step"] == final_step].copy().sort_values("politics_score").reset_index(drop=True)

    plt.figure(figsize=(8, 4))
    plt.bar(range(1, len(final_df) + 1), final_df["politics_score"])
    plt.axhline(BASELINE, linestyle="--", linewidth=1, label="Baseline = 1/9")
    plt.xlabel("Chains sorted by final politics score")
    plt.ylabel(f"Politics score at step {final_step:,}")
    plt.title(f"Final politics scores across {len(final_df)} chains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f"final_politics_score_sorted_bar_step_{final_step}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(final_df["politics_score"], bins=10, edgecolor="black")
    plt.axvline(BASELINE, linestyle="--", linewidth=1, label="Baseline = 1/9")
    plt.xlabel(f"Politics score at step {final_step:,}")
    plt.ylabel("Number of chains")
    plt.title(f"Distribution of final politics scores across {len(final_df)} chains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / f"final_politics_score_histogram_step_{final_step}.png", dpi=200)
    plt.close()

    # Plot all-topic score changes by seed topic.
    topic_cols = [f"{t}_score" for t in CANDIDATE_TOPICS if f"{t}_score" in df.columns]
    start_df = df[df["step"] == 0].copy()
    merged = start_df[["seed_topic", "seed_id"] + topic_cols].merge(
        final_df[["seed_topic", "seed_id"] + topic_cols],
        on=["seed_topic", "seed_id"],
        suffixes=("_start", "_final"),
    )

    for seed_topic, group in merged.groupby("seed_topic"):
        values = []
        labels = []
        for topic in CANDIDATE_TOPICS:
            start_col = f"{topic}_score_start"
            final_col = f"{topic}_score_final"
            if start_col in group.columns:
                labels.append(topic)
                values.append((group[final_col] - group[start_col]).mean())
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.axhline(0, linewidth=1)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(f"Mean score change, step {final_step:,} minus step 0")
        plt.title(f"Topic score changes from {seed_topic} initializations")
        plt.tight_layout()
        safe_topic = str(seed_topic).replace(" ", "_")
        plt.savefig(output / f"topic_score_changes_{safe_topic}_step_{final_step}.png", dpi=200)
        plt.close()

    print(f"Wrote figures to: {output}")


def rescore_checkpoint_csv(args: argparse.Namespace) -> None:
    input_csv = Path(args.checkpoint_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_checkpoint_df(str(input_csv))

    dev = device_string()
    print(f"Device: {dev}")
    print(f"Loading secondary classifier: {args.classifier_model}")
    classifier = pipeline(
        "zero-shot-classification",
        model=args.classifier_model,
        device=0 if dev == "cuda" else -1,
    )

    # Score all checkpoint texts with a second classifier. This can be slow.
    rows = []
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="Rescoring texts"):
        scores = classify_topics(r.text, classifier)
        row = r._asdict()
        for topic in CANDIDATE_TOPICS:
            row[f"{topic}_score"] = float(scores[topic])
        row["classifier_model"] = args.classifier_model
        rows.append(row)

    out_csv = output_dir / "checkpoint_results_rescored.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote rescored checkpoints to: {out_csv}")
    summarize_checkpoint_csv(str(out_csv), str(output_dir))
    plot_from_checkpoint_csv(str(out_csv), str(output_dir / "figures"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Political drift experiments for MLM Glauber dynamics.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run Glauber dynamics and score topic trajectories.")
    run.add_argument("--model_name", type=str, default="bert-base-uncased")
    run.add_argument("--model_label", type=str, default="BERT")
    run.add_argument("--classifier_model", type=str, default="facebook/bart-large-mnli")
    run.add_argument("--steps", type=int, default=10000)
    run.add_argument("--record_every", type=int, default=200)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--seeds_per_topic", type=int, default=10)
    run.add_argument("--random_seed", type=int, default=42)
    run.add_argument("--seed_json", type=str, default=None, help="Optional JSON file with custom seed groups.")
    run.add_argument("--output_dir", type=str, required=True)
    run.add_argument("--resume", action="store_true", help="Skip chains already present in checkpoint_results.csv.")
    run.set_defaults(func=run_experiment)

    summ = sub.add_parser("summarize", help="Create summary CSVs from checkpoint_results.csv.")
    summ.add_argument("--checkpoint_csv", type=str, required=True)
    summ.add_argument("--output_dir", type=str, required=True)
    summ.set_defaults(func=lambda args: summarize_checkpoint_csv(args.checkpoint_csv, args.output_dir))

    plot = sub.add_parser("plot", help="Create figures from checkpoint_results.csv.")
    plot.add_argument("--checkpoint_csv", type=str, required=True)
    plot.add_argument("--output_dir", type=str, required=True)
    plot.set_defaults(func=lambda args: plot_from_checkpoint_csv(args.checkpoint_csv, args.output_dir))

    rescore = sub.add_parser("rescore", help="Rescore saved checkpoint texts with a different zero-shot classifier.")
    rescore.add_argument("--checkpoint_csv", type=str, required=True)
    rescore.add_argument("--classifier_model", type=str, default="MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33")
    rescore.add_argument("--output_dir", type=str, required=True)
    rescore.set_defaults(func=rescore_checkpoint_csv)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
