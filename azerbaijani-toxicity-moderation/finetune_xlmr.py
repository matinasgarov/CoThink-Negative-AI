"""
Multi-label fine-tuning of XLM-RoBERTa on the toxicity dataset.

NOTE: this machine has CPU-only torch (no CUDA). Fine-tuning xlm-roberta-base
on ~60k rows on CPU is realistically hours-per-epoch, not minutes. For a real
training run, move this to a GPU machine (Colab/cloud) -- the script itself
doesn't change, torch will just pick up CUDA automatically if available.

Use --max_train_samples for a fast CPU smoke test to confirm the pipeline
works end-to-end before running the full job on a GPU.
"""

import argparse

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABEL_COLS = [
    "identity_attack",
    "insult",
    "obscene",
    "severe_toxicity",
    "sexual_explicit",
    "threat",
    "toxicity",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="xlm-roberta-base")
    p.add_argument("--train_file", default="azerbaijani_toxicity_train.csv")
    p.add_argument("--val_file", default="azerbaijani_toxicity_val.csv")
    p.add_argument("--output_dir", default="xlmr_toxicity_model")
    p.add_argument("--max_length", type=int, default=96)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_train_samples", type=int, default=None,
                    help="Subsample the train set for a quick smoke test.")
    return p.parse_args()


def load_dataset(path, max_samples=None):
    ds = Dataset.from_csv(path)
    if max_samples is not None:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    labels = np.stack([np.array(ds[c], dtype=np.float32) for c in LABEL_COLS], axis=1)
    ds = ds.add_column("labels", labels.tolist())
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    labels = labels.astype(int)

    results = {}
    f1s = []
    for i, col in enumerate(LABEL_COLS):
        p, r, f1, _ = precision_recall_fscore_support(
            labels[:, i], preds[:, i], average="binary", zero_division=0
        )
        results[f"{col}_precision"] = p
        results[f"{col}_recall"] = r
        results[f"{col}_f1"] = f1
        f1s.append(f1)
    results["macro_f1"] = float(np.mean(f1s))
    return results


def main():
    args = parse_args()
    print(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    )

    train_ds = load_dataset(args.train_file, args.max_train_samples)
    val_ds = load_dataset(args.val_file)

    def tokenize(batch):
        return tokenizer(
            batch["comment"], truncation=True, max_length=args.max_length, padding="max_length"
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("\n=== Final validation metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
