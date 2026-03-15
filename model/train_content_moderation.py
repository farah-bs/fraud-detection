"""
Training script for the content moderation model.

Expects the labeled CSVs produced by:
    python data/preprocess_tweets.py
    python data/pseudo_label.py

Usage:
    python model/train_content_moderation.py
    python model/train_content_moderation.py --epochs 5 --batch-size 16 --lr 2e-5
    python model/train_content_moderation.py --freeze-base   # probe only, fast iteration

Outputs:
    checkpoints/content_moderation_best.pt   ← best val macro-F1
    checkpoints/content_moderation_last.pt   ← last epoch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model.content_moderation_model import (
    BASE_MODEL,
    LABEL_NAMES,
    NUM_LABELS,
    ContentModerationModel,
)

# ── Config defaults ───────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "base_model":  BASE_MODEL,
    "num_labels":  NUM_LABELS,
    "hidden_dim":  256,
    "dropout":     0.3,
    "max_length":  128,
    "batch_size":  32,
    "lr":          2e-5,
    "epochs":      4,
    "warmup_ratio": 0.06,
    "weight_decay": 0.01,
    "threshold":   0.5,
    "num_workers": 0 if sys.platform == "darwin" else 4,
    "log_every_steps": 50,
    "early_stopping_patience": 2,
    "early_stopping_min_delta": 1e-4,
}

DATA_DIR  = os.path.join(ROOT, "data", "processed")
CKPT_DIR  = os.path.join(ROOT, "checkpoints")
RUNS_DIR  = os.path.join(ROOT, "runs", "content_moderation")

LABEL_COLS = [f"label_{l}" for l in LABEL_NAMES[:-2]] + ["label_spam", "label_safe"]


# ── Tracking helpers ──────────────────────────────────────────────────────────

def _build_run_name(run_name: str | None = None) -> str:
    if run_name:
        return run_name
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _init_trackers(cfg: dict) -> tuple[str, str, SummaryWriter | None, Any]:
    """
    Initialize local record keeping + optional TensorBoard / W&B trackers.

    Returns: (run_name, run_dir, tb_writer, wandb_module_or_none)
    """
    run_name = _build_run_name(cfg.get("run_name"))
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Persist config for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)

    # Create CSV with header for durable metrics history
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "epoch", "global_step", "train_loss", "val_macro_f1",
            *[f"f1_{label}" for label in LABEL_NAMES],
        ])

    tracking_mode = cfg.get("tracking", "tensorboard")

    tb_writer: SummaryWriter | None = None
    if tracking_mode in {"tensorboard", "both"}:
        tb_writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))

    wandb_module: Any = None
    if tracking_mode in {"wandb", "both"}:
        try:
            import wandb  # type: ignore
            wandb.init(
                project=cfg.get("wandb_project", "content-moderation"),
                name=run_name,
                config=cfg,
                dir=run_dir,
                reinit=True,
            )
            wandb_module = wandb
        except ImportError:
            print("[WARN] wandb not installed. Falling back to local records only.")
        except Exception as exc:
            print(f"[WARN] Failed to initialize wandb ({exc}). Falling back to local records only.")

    return run_name, run_dir, tb_writer, wandb_module


def _append_metrics_row(run_dir: str, epoch: int, global_step: int, train_loss: float, metrics: dict[str, float]) -> None:
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            epoch,
            global_step,
            round(train_loss, 6),
            round(metrics["macro_f1"], 6),
            *[round(metrics[f"f1_{label}"], 6) for label in LABEL_NAMES],
        ])


def _log_epoch(tb_writer: SummaryWriter | None, wandb_module: Any, epoch: int, global_step: int, train_loss: float, metrics: dict[str, float]) -> None:
    if tb_writer is not None:
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("val/macro_f1", metrics["macro_f1"], epoch)
        for label in LABEL_NAMES:
            tb_writer.add_scalar(f"val/f1_{label}", metrics[f"f1_{label}"], epoch)

    if wandb_module is not None:
        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "train/loss": train_loss,
            "val/macro_f1": metrics["macro_f1"],
        }
        for label in LABEL_NAMES:
            payload[f"val/f1_{label}"] = metrics[f"f1_{label}"]
        wandb_module.log(payload, step=global_step)


def _log_step(tb_writer: SummaryWriter | None, wandb_module: Any, global_step: int, loss_value: float, lr: float) -> None:
    if tb_writer is not None:
        tb_writer.add_scalar("train/step_loss", loss_value, global_step)
        tb_writer.add_scalar("train/lr", lr, global_step)

    if wandb_module is not None:
        wandb_module.log(
            {
                "train/step_loss": loss_value,
                "train/lr": lr,
                "global_step": global_step,
            },
            step=global_step,
        )


# ── Dataset ───────────────────────────────────────────────────────────────────

class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128) -> None:
        self.texts      = df["text"].tolist()
        self.labels     = df[LABEL_COLS].values.astype(np.float32)
        self.tokenizer  = tokenizer
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
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Weighted sampler (combat class imbalance on the spam label) ───────────────

def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Up-sample minority classes based on spam/toxic presence."""
    spam_col   = "label_spam"
    toxic_cols = [c for c in LABEL_COLS if c != "label_safe" and c != "label_spam"]
    is_toxic   = np.array(df[toxic_cols]).sum(axis=1) > 0
    is_harmful = (df[spam_col] == 1) | is_toxic
    weights    = np.where(is_harmful, 2.0, 1.0).astype(np.float64)
    weights   /= weights.sum()
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(df),
        replacement=True,
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    all_logits: np.ndarray,
    all_labels: np.ndarray,
    threshold:  float = 0.5,
) -> dict[str, float]:
    probs = 1 / (1 + np.exp(-all_logits))          # sigmoid
    preds = (probs >= threshold).astype(int)

    per_label_f1 = f1_score(all_labels, preds, average=None, zero_division=0)
    macro_f1     = float(np.mean(per_label_f1))

    metrics: dict[str, float] = {"macro_f1": macro_f1}
    for name, score in zip(LABEL_NAMES, list(np.asarray(per_label_f1).flat)):
        metrics[f"f1_{name}"] = float(score)
    return metrics


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n=== Training content moderation model  [{device}] ===\n")

    # -- Data --
    train_df = pd.read_csv(os.path.join(DATA_DIR, "tweets_train_labeled.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "tweets_val_labeled.csv"))
    print(f"  Train: {len(train_df):,}   Val: {len(val_df):,}")

    # Sanity-check all required columns exist
    for col in LABEL_COLS:
        if col not in train_df.columns:
            raise ValueError(
                f"Column '{col}' missing from training data.\n"
                "Run: python data/pseudo_label.py"
            )

    tokenizer  = AutoTokenizer.from_pretrained(cfg["base_model"])
    train_ds   = TweetDataset(train_df, tokenizer, max_length=cfg["max_length"])
    val_ds     = TweetDataset(val_df,   tokenizer, max_length=cfg["max_length"])

    sampler    = make_sampler(train_df)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], sampler=sampler,
        num_workers=cfg["num_workers"], pin_memory=(device.type != "mps"),
    )
    val_loader   = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=(device.type != "mps"),
    )

    # -- Model --
    model = ContentModerationModel(
        base_model  = cfg["base_model"],
        num_labels  = cfg["num_labels"],
        hidden_dim  = cfg["hidden_dim"],
        dropout     = cfg["dropout"],
        freeze_base = cfg.get("freeze_base", False),
    ).to(device)

    # -- Optimizer / scheduler --
    no_decay   = ["bias", "LayerNorm.weight"]
    opt_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer  = torch.optim.AdamW(opt_groups, lr=cfg["lr"])
    total_steps   = len(train_loader) * cfg["epochs"]
    warmup_steps  = int(total_steps * cfg["warmup_ratio"])
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    run_name, run_dir, tb_writer, wandb_module = _init_trackers(cfg)
    print(f"  Run: {run_name}")
    print(f"  Records: {run_dir}")

    best_macro_f1 = 0.0
    epochs_without_improvement = 0
    global_step = 0

    # -- Epoch loop --
    for epoch in range(1, cfg["epochs"] + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            if global_step % cfg["log_every_steps"] == 0:
                current_lr = float(optimizer.param_groups[0]["lr"])
                _log_step(tb_writer, wandb_module, global_step, float(loss.item()), current_lr)
            if step % 200 == 0:
                print(f"  Epoch {epoch} | step {step}/{len(train_loader)} | loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits         = model(input_ids, attention_mask)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(batch["labels"].numpy())

        all_logits_arr = np.vstack(all_logits)
        all_labels_arr = np.vstack(all_labels)
        metrics        = compute_metrics(all_logits_arr, all_labels_arr, cfg["threshold"])

        print(
            f"\nEpoch {epoch}/{cfg['epochs']} — "
            f"train_loss={avg_loss:.4f}  val_macro_f1={metrics['macro_f1']:.4f}"
        )
        for label_name in LABEL_NAMES:
            print(f"   f1_{label_name:<18} {metrics[f'f1_{label_name}']:.4f}")
        print()

        _append_metrics_row(run_dir, epoch, global_step, avg_loss, metrics)
        _log_epoch(tb_writer, wandb_module, epoch, global_step, avg_loss, metrics)

        # ── Checkpoint ──
        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch":            epoch,
            "val_macro_f1":     metrics["macro_f1"],
            "model_config":     {k: cfg[k] for k in ("base_model", "num_labels", "hidden_dim", "dropout")},
            "label_names":      LABEL_NAMES,
            "threshold":        cfg["threshold"],
        }

        last_path = os.path.join(CKPT_DIR, "content_moderation_last.pt")
        torch.save(ckpt, last_path)

        if metrics["macro_f1"] > (best_macro_f1 + cfg["early_stopping_min_delta"]):
            best_macro_f1 = metrics["macro_f1"]
            epochs_without_improvement = 0
            best_path = os.path.join(CKPT_DIR, "content_moderation_best.pt")
            torch.save(ckpt, best_path)
            print(f"  ✓ New best macro-F1 = {best_macro_f1:.4f} — saved to {best_path}")
        else:
            epochs_without_improvement += 1
            print(
                f"  No val improvement for {epochs_without_improvement} epoch(s) "
                f"(patience={cfg['early_stopping_patience']})"
            )

            if epochs_without_improvement >= cfg["early_stopping_patience"]:
                print(
                    "  Early stopping triggered: "
                    f"no macro-F1 improvement >= {cfg['early_stopping_min_delta']} "
                    f"for {cfg['early_stopping_patience']} consecutive epoch(s)."
                )
                break

    if tb_writer is not None:
        tb_writer.close()
    if wandb_module is not None:
        wandb_module.finish()

    print(f"\n=== Training complete — best val macro-F1: {best_macro_f1:.4f} ===")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train content moderation model")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--hidden-dim",  type=int,   default=DEFAULT_CONFIG["hidden_dim"])
    parser.add_argument("--dropout",     type=float, default=DEFAULT_CONFIG["dropout"])
    parser.add_argument("--max-length",  type=int,   default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--threshold",   type=float, default=DEFAULT_CONFIG["threshold"])
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_CONFIG["num_workers"],
        help="DataLoader worker processes (default: 0 on macOS, 4 elsewhere)",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=50,
        help="Log training step loss every N optimizer steps (default: 50)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_CONFIG["early_stopping_patience"],
        help="Stop training after N epochs without val macro-F1 improvement (default: 2)",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=DEFAULT_CONFIG["early_stopping_min_delta"],
        help="Minimum val macro-F1 improvement to reset early stopping (default: 1e-4)",
    )
    parser.add_argument(
        "--tracking",
        type=str,
        choices=["none", "tensorboard", "wandb", "both"],
        default="tensorboard",
        help="Metrics tracking backend (default: tensorboard)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Default: timestamp-based run_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="content-moderation",
        help="W&B project name when tracking includes wandb",
    )
    parser.add_argument(
        "--freeze-base", action="store_true",
        help="Freeze the RoBERTa encoder (train classification head only)"
    )
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg["epochs"]      = args.epochs
    cfg["batch_size"]  = args.batch_size
    cfg["lr"]          = args.lr
    cfg["hidden_dim"]  = args.hidden_dim
    cfg["dropout"]     = args.dropout
    cfg["max_length"]  = args.max_length
    cfg["threshold"]   = args.threshold
    cfg["num_workers"] = args.num_workers
    cfg["log_every_steps"] = args.log_every_steps
    cfg["early_stopping_patience"] = args.early_stopping_patience
    cfg["early_stopping_min_delta"] = args.early_stopping_min_delta
    cfg["tracking"]    = args.tracking
    cfg["run_name"]    = args.run_name
    cfg["wandb_project"] = args.wandb_project
    cfg["freeze_base"] = args.freeze_base

    train(cfg)
