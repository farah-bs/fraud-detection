"""
Comprehensive evaluation of the Content Moderation Model on test dataset.

Calculates all relevant metrics:
- Accuracy (subset and hamming)
- Precision, Recall, F1 (macro, micro, weighted)
- ROC-AUC scores
- Per-label metrics
- Confusion matrices
- Classification reports

Usage:
    python evaluate_model.py
    python evaluate_model.py --checkpoint path/to/checkpoint.pt
    python evaluate_model.py --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model.content_moderation_model import (
    BASE_MODEL,
    LABEL_NAMES,
    NUM_LABELS,
    ContentModerationModel,
    load_from_checkpoint,
)


# ── Paths ──────────────────────────────────────────────────────────────────

DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_PATH = ROOT / "checkpoints" / "content_moderation_best.pt"
RESULTS_DIR = ROOT / "evaluation_results"

LABEL_COLS = [f"label_{l}" for l in LABEL_NAMES[:-2]] + ["label_spam", "label_safe"]


# ── Dataset ────────────────────────────────────────────────────────────────

class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128) -> None:
        self.texts = df["text"].tolist()
        self.labels = df[LABEL_COLS].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    test_loader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Run evaluation on test set and compute all metrics."""

    print("\n" + "=" * 70)
    print("EVALUATING MODEL ON TEST DATASET")
    print("=" * 70)

    model.eval()
    all_logits = []
    all_labels = []

    print("\n📊 Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * len(batch['input_ids']):,} samples")

    all_logits_arr = np.vstack(all_logits)  # (N, 8)
    all_labels_arr = np.vstack(all_labels)  # (N, 8)

    print(f"  ✓ Total samples: {len(all_logits_arr):,}")

    # Convert to probabilities
    probs = 1 / (1 + np.exp(-all_logits_arr))  # Sigmoid
    preds = (probs >= threshold).astype(int)

    # ── MULTI-LABEL METRICS ────────────────────────────────────────────────

    results = {
        "metadata": {
            "num_samples": len(all_logits_arr),
            "num_labels": NUM_LABELS,
            "threshold": threshold,
            "label_names": LABEL_NAMES,
        }
    }

    print("\n" + "-" * 70)
    print("MULTI-LABEL METRICS")
    print("-" * 70)

    # 1. Subset Accuracy (exact match)
    subset_acc = accuracy_score(all_labels_arr, preds)
    print(f"\n📌 Subset Accuracy (Exact Match): {subset_acc:.4f}")
    print(f"    Meaning: % of samples where ALL labels predicted correctly")
    results["subset_accuracy"] = float(subset_acc)

    # 2. Hamming Loss
    ham_loss = hamming_loss(all_labels_arr, preds)
    print(f"\n📌 Hamming Loss: {ham_loss:.4f}")
    print(f"    Meaning: Fraction of incorrect labels (lower is better)")
    print(f"    Hamming Accuracy: {1 - ham_loss:.4f}")
    results["hamming_loss"] = float(ham_loss)
    results["hamming_accuracy"] = float(1 - ham_loss)

    # 3. Per-Sample Label Count
    pred_label_counts = preds.sum(axis=1)
    true_label_counts = all_labels_arr.sum(axis=1)
    print(f"\n📌 Label Count Statistics:")
    print(f"    True avg labels per sample: {true_label_counts.mean():.2f}")
    print(f"    Pred avg labels per sample: {pred_label_counts.mean():.2f}")
    results["avg_true_labels"] = float(true_label_counts.mean())
    results["avg_pred_labels"] = float(pred_label_counts.mean())

    # ── PER-LABEL METRICS ──────────────────────────────────────────────────

    print("\n" + "-" * 70)
    print("PER-LABEL METRICS")
    print("-" * 70)

    per_label_metrics = {}

    for label_idx, label_name in enumerate(LABEL_NAMES):
        y_true_label = all_labels_arr[:, label_idx]
        y_pred_label = preds[:, label_idx]
        y_prob_label = probs[:, label_idx]

        # Calculate metrics
        precision = precision_score(y_true_label, y_pred_label, zero_division=0)
        recall = recall_score(y_true_label, y_pred_label, zero_division=0)
        f1 = f1_score(y_true_label, y_pred_label, zero_division=0)

        # ROC-AUC (only if label has both positive and negative samples)
        if len(np.unique(y_true_label)) > 1:
            roc_auc = roc_auc_score(y_true_label, y_prob_label)
        else:
            roc_auc = np.nan

        per_label_metrics[label_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
            "support": int(y_true_label.sum()),
        }

    # Print per-label metrics in table
    print("\n{:<20} {:<12} {:<12} {:<12} {:<12} {:<10}".format(
        "Label", "Precision", "Recall", "F1", "ROC-AUC", "Support"
    ))
    print("-" * 78)
    for label_name in LABEL_NAMES:
        metrics = per_label_metrics[label_name]
        roc_auc_str = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "N/A"
        print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12} {:<10}".format(
            label_name,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            roc_auc_str,
            metrics["support"]
        ))

    results["per_label_metrics"] = per_label_metrics

    # ── AGGREGATE METRICS ──────────────────────────────────────────────────

    print("\n" + "-" * 70)
    print("AGGREGATE METRICS")
    print("-" * 70)

    # Macro-averaged
    macro_precision = np.mean([m["precision"] for m in per_label_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in per_label_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in per_label_metrics.values()])

    print(f"\n📊 Macro-Averaged (unweighted average across labels):")
    print(f"    Precision: {macro_precision:.4f}")
    print(f"    Recall:    {macro_recall:.4f}")
    print(f"    F1:        {macro_f1:.4f}")

    results["macro_metrics"] = {
        "precision": float(macro_precision),
        "recall": float(macro_recall),
        "f1": float(macro_f1),
    }

    # Micro-averaged
    tp = np.sum((preds == 1) & (all_labels_arr == 1))
    fp = np.sum((preds == 1) & (all_labels_arr == 0))
    fn = np.sum((preds == 0) & (all_labels_arr == 1))

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    print(f"\n📊 Micro-Averaged (treating all predictions as one problem):")
    print(f"    Precision: {micro_precision:.4f}")
    print(f"    Recall:    {micro_recall:.4f}")
    print(f"    F1:        {micro_f1:.4f}")

    results["micro_metrics"] = {
        "precision": float(micro_precision),
        "recall": float(micro_recall),
        "f1": float(micro_f1),
    }

    # Weighted-averaged (by support)
    supports = np.array([per_label_metrics[name]["support"] for name in LABEL_NAMES])
    total_support = supports.sum()

    weighted_precision = np.sum([
        per_label_metrics[name]["precision"] * per_label_metrics[name]["support"]
        for name in LABEL_NAMES
    ]) / total_support if total_support > 0 else 0

    weighted_recall = np.sum([
        per_label_metrics[name]["recall"] * per_label_metrics[name]["support"]
        for name in LABEL_NAMES
    ]) / total_support if total_support > 0 else 0

    weighted_f1 = np.sum([
        per_label_metrics[name]["f1"] * per_label_metrics[name]["support"]
        for name in LABEL_NAMES
    ]) / total_support if total_support > 0 else 0

    print(f"\n📊 Weighted-Averaged (weighted by label support):")
    print(f"    Precision: {weighted_precision:.4f}")
    print(f"    Recall:    {weighted_recall:.4f}")
    print(f"    F1:        {weighted_f1:.4f}")

    results["weighted_metrics"] = {
        "precision": float(weighted_precision),
        "recall": float(weighted_recall),
        "f1": float(weighted_f1),
    }

    # ── ADDITIONAL METRICS ─────────────────────────────────────────────────

    print("\n" + "-" * 70)
    print("ADDITIONAL METRICS")
    print("-" * 70)

    # Label distribution
    true_positive_rate = all_labels_arr.mean(axis=0)
    pred_positive_rate = preds.mean(axis=0)

    print(f"\n📊 Label Distribution:")
    print(f"{'Label':<20} {'True %':<15} {'Pred %':<15} {'Diff':<10}")
    print("-" * 60)
    for label_idx, label_name in enumerate(LABEL_NAMES):
        true_pct = true_positive_rate[label_idx] * 100
        pred_pct = pred_positive_rate[label_idx] * 100
        diff = abs(true_pct - pred_pct)
        print(f"{label_name:<20} {true_pct:<14.2f}% {pred_pct:<14.2f}% {diff:>8.2f}%")

    results["label_distribution"] = {
        "true_positive_rate": [float(x) for x in true_positive_rate],
        "pred_positive_rate": [float(x) for x in pred_positive_rate],
    }

    # Mean of probabilities
    print(f"\n📊 Prediction Confidence:")
    print(f"    Mean probability (correct predictions): {probs[preds == all_labels_arr].mean():.4f}")
    print(f"    Mean probability (incorrect predictions): {probs[preds != all_labels_arr].mean():.4f}")

    results["confidence"] = {
        "correct_mean_prob": float(probs[preds == all_labels_arr].mean()),
        "incorrect_mean_prob": float(probs[preds != all_labels_arr].mean()),
    }

    return results, all_logits_arr, all_labels_arr, preds, probs


def save_results(results: dict, all_labels: np.ndarray, preds: np.ndarray) -> None:
    """Save evaluation results to JSON and CSV."""

    RESULTS_DIR.mkdir(exist_ok=True)

    # Save JSON
    json_path = RESULTS_DIR / "evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics saved: {json_path}")

    # Save per-label metrics as CSV
    per_label_df = pd.DataFrame(results["per_label_metrics"]).T
    csv_path = RESULTS_DIR / "per_label_metrics.csv"
    per_label_df.to_csv(csv_path)
    print(f"✓ Per-label metrics saved: {csv_path}")

    # Save confusion matrices
    for label_idx, label_name in enumerate(LABEL_NAMES):
        y_true = all_labels[:, label_idx]
        y_pred = preds[:, label_idx]
        cm = confusion_matrix(y_true, y_pred)

        cm_path = RESULTS_DIR / f"confusion_matrix_{label_name}.npy"
        np.save(cm_path, cm)

    print(f"✓ Confusion matrices saved to {RESULTS_DIR}/")


def main(
    checkpoint_path: str | None = None,
    batch_size: int = 32,
    device_str: str = "auto",
) -> None:
    """Main evaluation pipeline."""

    # Setup paths
    if checkpoint_path is None:
        checkpoint_path = str(CHECKPOINT_PATH)

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load test data
    test_path = DATA_DIR / "tweets_test_labeled.csv"
    if not test_path.exists():
        print(f"❌ Test data not found: {test_path}")
        print("   Run: python data/preprocess_tweets.py && python data/pseudo_label.py")
        sys.exit(1)

    test_df = pd.read_csv(test_path)
    print(f"\n📂 Test set loaded: {len(test_df):,} samples")

    # Setup device
    if device_str == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device_str)

    print(f"📊 Device: {device}")

    # Load model
    print(f"\n🔄 Loading model from: {checkpoint_path}")
    model, cfg = load_from_checkpoint(checkpoint_path, device=device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("base_model", BASE_MODEL))

    # Create dataset and dataloader
    test_dataset = TweetDataset(test_df, tokenizer, max_length=128)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Evaluate
    results, logits, labels, preds, probs = evaluate_model(
        model,
        test_loader,
        device,
        threshold=cfg.get("threshold", 0.5),
    )

    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    save_results(results, labels, preds)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n✅ Subset Accuracy:     {results['subset_accuracy']:.4f}")
    print(f"✅ Hamming Accuracy:    {results['hamming_accuracy']:.4f}")
    print(f"✅ Macro F1:            {results['macro_metrics']['f1']:.4f}")
    print(f"✅ Micro F1:            {results['micro_metrics']['f1']:.4f}")
    print(f"✅ Weighted F1:         {results['weighted_metrics']['f1']:.4f}")

    print("\n📊 Top-3 Best Performing Labels:")
    sorted_labels = sorted(
        results["per_label_metrics"].items(),
        key=lambda x: x[1]["f1"],
        reverse=True
    )[:3]
    for rank, (label, metrics) in enumerate(sorted_labels, 1):
        print(f"    {rank}. {label:<20} F1: {metrics['f1']:.4f}")

    print("\n📊 Top-3 Worst Performing Labels:")
    sorted_labels = sorted(
        results["per_label_metrics"].items(),
        key=lambda x: x[1]["f1"],
    )[:3]
    for rank, (label, metrics) in enumerate(sorted_labels, 1):
        print(f"    {rank}. {label:<20} F1: {metrics['f1']:.4f}")

    print("\n" + "=" * 70)
    print("✓ Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate content moderation model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: checkpoints/content_moderation_best.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use (default: auto-detect)",
    )

    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        device_str=args.device,
    )
