# Political Drift Experiments for MLM Glauber Dynamics

This folder contains the code needed to reproduce the political drift section of the paper. The experiments study whether masked-token Glauber dynamics on masked language models enters political regions of text space from non-political initializations.

The main script is:

```text
political_drift_experiments.py
```

It supports four workflows:

1. Run Glauber dynamics and score topic trajectories.
2. Summarize endpoint political drift, transient political entry, and seed-topic loss.
3. Plot the final politics-score distribution and all-topic score changes.
4. Rescore saved checkpoint texts with a second zero-shot classifier for robustness checks.

## Dependencies

Recommended Python version: 3.10 or newer.

Install dependencies:

```bash
pip install torch transformers pandas numpy matplotlib tqdm sentencepiece accelerate
```

If running on Colab, use a GPU runtime. The experiments were originally run on Colab T4/A100 GPUs.

## Candidate topics

All zero-shot classifiers use the same nine candidate labels:

```text
politics, health, sport, science, technology, food, finance, environment, art
```

The uniform-label baseline is therefore:

```text
1/9 = 0.111...
```

## Default seed topics

The script includes 10 non-political seed sentences for each of five initialization topics:

```text
Health, Sport, Science, Food, Technology
```

To use custom seeds, pass a JSON file with this format:

```json
{
  "Health": ["seed sentence 1", "seed sentence 2"],
  "Sport": ["seed sentence 1", "seed sentence 2"]
}
```

and run with:

```bash
--seed_json path/to/seeds.json
```

## Main BERT experiment with BART-MNLI scoring

This is the main experiment used for the paper tables:

```bash
python political_drift_experiments.py run \
  --model_name bert-base-uncased \
  --model_label BERT \
  --classifier_model facebook/bart-large-mnli \
  --steps 10000 \
  --record_every 200 \
  --temperature 1.0 \
  --seeds_per_topic 10 \
  --output_dir outputs/bert_bart_10k
```

The script saves after every chain, so it can survive Colab disconnects. If the runtime disconnects, rerun with `--resume`:

```bash
python political_drift_experiments.py run \
  --model_name bert-base-uncased \
  --model_label BERT \
  --classifier_model facebook/bart-large-mnli \
  --steps 10000 \
  --record_every 200 \
  --temperature 1.0 \
  --seeds_per_topic 10 \
  --output_dir outputs/bert_bart_10k \
  --resume
```

## Outputs

A run creates:

```text
checkpoint_results.csv
run_config.json
endpoint_politics_summary.csv
max_politics_summary.csv
peak_politics_by_chain.csv
seed_topic_identity_loss.csv
all_topic_score_changes.csv
top_topic_gains.csv
final_politics_by_chain.csv
figures/
```

Important files:

- `endpoint_politics_summary.csv`: mean politics score at step 0 and step 10,000 by seed topic.
- `max_politics_summary.csv`: maximum politics score reached by each chain, summarized by seed topic.
- `seed_topic_identity_loss.csv`: drop in the original seed-topic score.
- `all_topic_score_changes.csv`: changes for all nine topic scores.
- `top_topic_gains.csv`: top three gaining topics per seed class.
- `final_politics_by_chain.csv`: final politics score for every chain.
- `figures/final_politics_score_sorted_bar_step_10000.png`: sorted bar chart of final politics scores.
- `figures/final_politics_score_histogram_step_10000.png`: histogram of final politics scores.

## Summarize or plot from an existing checkpoint CSV

If you already have `checkpoint_results.csv`, no rerun is needed.

Create summaries:

```bash
python political_drift_experiments.py summarize \
  --checkpoint_csv outputs/bert_bart_10k/checkpoint_results.csv \
  --output_dir outputs/bert_bart_10k
```

Create figures:

```bash
python political_drift_experiments.py plot \
  --checkpoint_csv outputs/bert_bart_10k/checkpoint_results.csv \
  --output_dir outputs/bert_bart_10k/figures
```

## DeBERTa-v3 classifier robustness check

To check that the political drift result is not specific to BART-MNLI, rescore the saved checkpoint texts with DeBERTa-v3:

```bash
python political_drift_experiments.py rescore \
  --checkpoint_csv outputs/bert_bart_10k/checkpoint_results.csv \
  --classifier_model MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33 \
  --output_dir outputs/bert_deberta_rescore_10k
```

This uses the same generated texts and changes only the classifier.

## Cross-model endpoint comparison

To compare models, run the same protocol with `seeds_per_topic=5` for each model.

BERT:

```bash
python political_drift_experiments.py run \
  --model_name bert-base-uncased \
  --model_label BERT \
  --classifier_model facebook/bart-large-mnli \
  --steps 10000 --record_every 200 --temperature 1.0 \
  --seeds_per_topic 5 \
  --output_dir outputs/cross_model_bert_bart
```

RoBERTa:

```bash
python political_drift_experiments.py run \
  --model_name roberta-base \
  --model_label RoBERTa \
  --classifier_model facebook/bart-large-mnli \
  --steps 10000 --record_every 200 --temperature 1.0 \
  --seeds_per_topic 5 \
  --output_dir outputs/cross_model_roberta_bart
```

ModernBERT:

```bash
python political_drift_experiments.py run \
  --model_name answerdotai/ModernBERT-base \
  --model_label ModernBERT \
  --classifier_model facebook/bart-large-mnli \
  --steps 10000 --record_every 200 --temperature 1.0 \
  --seeds_per_topic 5 \
  --output_dir outputs/cross_model_modernbert_bart
```

For a cleaner cross-model table, use the same `seeds_per_topic`, `steps`, `record_every`, and classifier for all models.

## 2,000-step diagnostic run

To generate the shorter 2,000-step distribution requested during review discussions:

```bash
python political_drift_experiments.py run \
  --model_name bert-base-uncased \
  --model_label BERT \
  --classifier_model facebook/bart-large-mnli \
  --steps 2000 \
  --record_every 200 \
  --temperature 1.0 \
  --seeds_per_topic 10 \
  --output_dir outputs/bert_bart_2k
```

This produces `figures/final_politics_score_sorted_bar_step_2000.png`.

## Notes for paper interpretation

The key political drift claim supported by these experiments is:

```text
Political content is not a universal endpoint, but it is a frequently reachable and sometimes sticky attractor in BERT's induced text dynamics.
```

The most important statistics are:

- endpoint politics scores by seed topic,
- number of chains that ever cross the politics baseline,
- maximum politics score reached by each chain,
- final politics-score distribution,
- drop in original seed-topic scores,
- all-topic score changes to show that politics is one attractor among several.

## Reproducibility caveats

- The experiments use stochastic token sampling, so exact values can vary across runs even with fixed seeds, depending on hardware and library versions.
- Zero-shot classifier scores are used as topic proxies, not ground-truth annotations.
- The cross-model endpoint comparison should use matching protocols across models for the cleanest interpretation.
- Full 10,000-step runs can be slow because every recorded checkpoint is passed through a zero-shot classifier.
