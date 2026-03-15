"""
Preprocessing pipeline for content moderation training data.

Input  : social_honeypot_icwsm_2011 TSV files (user_id, tweet_id, text, timestamp)
Output : data/processed/tweets_train.csv
         data/processed/tweets_val.csv
         data/processed/tweets_test.csv

Columns in output:
  text          — cleaned tweet text
  source_label  — 0 = legitimate, 1 = content_polluter (spam)

Run:
    python data/preprocess_tweets.py
    python data/preprocess_tweets.py --sample 150000  # tweets per class
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")

POLLUTER_FILE = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "social_honeypot_icwsm_2011",
    "content_polluters_tweets.txt",
)
LEGITIMATE_FILE = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "social_honeypot_icwsm_2011",
    "legitimate_users_tweets.txt",
)

# ── Text cleaning ─────────────────────────────────────────────────────────────

_URL_RE    = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HTML_RE   = re.compile(r"<[^>]+>")
_MULTI_WS  = re.compile(r"\s{2,}")


def clean_tweet(text: str) -> str:
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub("[URL]", text)        # keep placeholder so model knows link existed
    text = _MENTION_RE.sub("[USER]", text)   # anonymise mentions
    text = text.replace("\n", " ").replace("\r", " ")
    text = _MULTI_WS.sub(" ", text)
    return text.strip()


# ── Loader ────────────────────────────────────────────────────────────────────

def load_file(path: str, label: int, sample: int | None = None) -> pd.DataFrame:
    """Read a honeypot TSV and return a DataFrame with clean text + label."""
    print(f"  Loading {'polluter' if label else 'legitimate'} tweets from {path} …")
    rows = []
    seen: set[str] = set()

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            raw_text = parts[2].strip()
            if not raw_text:
                continue
            cleaned = clean_tweet(raw_text)
            if not cleaned or cleaned in seen:   # deduplicate
                continue
            seen.add(cleaned)
            rows.append(cleaned)

    print(f"    → {len(rows):,} unique tweets after dedup")

    if sample and len(rows) > sample:
        import random
        random.seed(42)
        random.shuffle(rows)
        rows = rows[:sample]
        print(f"    → sampled down to {len(rows):,}")

    return pd.DataFrame({"text": rows, "source_label": label})


# ── Main ──────────────────────────────────────────────────────────────────────

def main(sample: int | None = None) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=== Phase 1: Loading datasets ===")
    df_polluters  = load_file(POLLUTER_FILE,  label=1, sample=sample)
    df_legitimate = load_file(LEGITIMATE_FILE, label=0, sample=sample)

    # Balance to the smaller class size so training isn't skewed
    n = min(len(df_polluters), len(df_legitimate))
    df_polluters  = df_polluters.sample(n=n,  random_state=42)
    df_legitimate = df_legitimate.sample(n=n, random_state=42)
    print(f"\n  Balanced to {n:,} tweets per class ({2*n:,} total)")

    df = pd.concat([df_polluters, df_legitimate], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print("\n=== Phase 2: Train / Val / Test split (80 / 10 / 10) ===")
    train_df, tmp_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["source_label"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.5, random_state=42, stratify=tmp_df["source_label"]
    )

    print(f"  Train : {len(train_df):,}")
    print(f"  Val   : {len(val_df):,}")
    print(f"  Test  : {len(test_df):,}")

    train_path = os.path.join(DATA_DIR, "tweets_train.csv")
    val_path   = os.path.join(DATA_DIR, "tweets_val.csv")
    test_path  = os.path.join(DATA_DIR, "tweets_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"\n=== Done — files saved to {DATA_DIR}/ ===")
    print(f"  {os.path.basename(train_path)}")
    print(f"  {os.path.basename(val_path)}")
    print(f"  {os.path.basename(test_path)}")

    # Quick class distribution check
    print("\nLabel distribution (train):")
    print(train_df["source_label"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess honeypot tweet dataset")
    parser.add_argument(
        "--sample",
        type=int,
        default=150_000,
        help="Max tweets to sample per class (default 150000, use 0 for all)",
    )
    args = parser.parse_args()
    sample = args.sample if args.sample > 0 else None
    main(sample=sample)
