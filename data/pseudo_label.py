"""
Pseudo-labeling: enrich the preprocessed tweets with fine-grained toxicity scores.

Uses  unitary/toxic-bert  (multi-label toxicity classifier) to generate soft labels
for categories beyond the binary spam/legitimate source label.

Output columns added to CSV:
  toxic | severe_toxic | obscene | threat | insult | identity_hate

Then threshold at 0.5 to produce hard binary columns:
  label_toxic | label_severe_toxic | label_obscene |
  label_threat | label_insult | label_identity_hate | label_spam | label_safe

Run after preprocess_tweets.py:
    python data/pseudo_label.py
    python data/pseudo_label.py --batch-size 64  # tune for your GPU VRAM
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data", "processed")

SPLITS = ["tweets_train.csv", "tweets_val.csv", "tweets_test.csv"]

# unitary/toxic-bert label order (matches model config)
TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

TOXICITY_THRESHOLD = 0.5


# ── Inference helpers ─────────────────────────────────────────────────────────

def load_toxic_bert(device: torch.device):
    model_name = "unitary/toxic-bert"
    print(f"  Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()
    print(f"  Model on {device}")
    return tokenizer, model


@torch.no_grad()
def score_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """Return sigmoid scores shape (N, 6) for TOXIC_LABELS."""
    all_scores: list[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  Scoring batches"):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits          # (B, 6)
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_scores.append(probs)

    return np.vstack(all_scores)              # (N, 6)


# ── Per-split processing ──────────────────────────────────────────────────────

def process_split(
    path: str,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> None:
    print(f"\n  Processing {os.path.basename(path)} …")
    df = pd.read_csv(path)

    scores = score_texts(
        df["text"].tolist(), tokenizer, model, device, batch_size=batch_size
    )

    # Add soft scores
    for j, lbl in enumerate(TOXIC_LABELS):
        df[lbl] = scores[:, j].round(4)

    # Add hard binary labels
    for lbl in TOXIC_LABELS:
        df[f"label_{lbl}"] = (df[lbl] >= TOXICITY_THRESHOLD).astype(int)

    # spam / safe come from source_label
    df["label_spam"] = df["source_label"]
    df["label_safe"] = (
        (df["source_label"] == 0) &
        (np.array(df[[f"label_{l}" for l in TOXIC_LABELS]]).sum(axis=1) == 0)
    ).astype(int)

    out_path = path.replace(".csv", "_labeled.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved → {os.path.basename(out_path)}")

    # Distribution summary
    label_cols = [f"label_{l}" for l in TOXIC_LABELS] + ["label_spam", "label_safe"]
    print("  Label distribution:")
    for col in label_cols:
        pos = df[col].sum()
        print(f"    {col:<22} {pos:>7,}  ({100*pos/len(df):.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(batch_size: int = 64) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"=== Phase 2: Pseudo-labeling with toxic-bert  [{device}] ===")

    tokenizer, model = load_toxic_bert(device)

    for split_file in SPLITS:
        path = os.path.join(DATA_DIR, split_file)
        if not os.path.exists(path):
            print(f"  Skipping {split_file} — file not found (run preprocess_tweets.py first)")
            continue
        process_split(path, tokenizer, model, device, batch_size)

    print("\n=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fine-grained toxicity pseudo-labels")
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Tokenizer/inference batch size (reduce if OOM, default 64)"
    )
    args = parser.parse_args()
    main(batch_size=args.batch_size)
