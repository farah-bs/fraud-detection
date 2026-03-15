"""
Content moderation model — TwitterRoBERTa fine-tuned for multi-label classification.

Architecture:
    cardiffnlp/twitter-roberta-base  (tweet-native encoder)
        ↓  [CLS] embedding (768-d)
        ↓  Dense(256) + GELU + Dropout
        ↓  Dense(N_LABELS) + Sigmoid   ← one probability per moderation category

Labels (8):
    toxic | severe_toxic | obscene | threat | insult | identity_hate | spam | safe
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ── Label schema ──────────────────────────────────────────────────────────────

LABEL_NAMES: list[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "spam",
    "safe",
]

NUM_LABELS = len(LABEL_NAMES)

BASE_MODEL = "cardiffnlp/twitter-roberta-base"


# ── Model ─────────────────────────────────────────────────────────────────────

class ContentModerationModel(nn.Module):
    """
    Multi-label tweet content moderator.

    Args:
        base_model: HuggingFace model name / local path (default: TwitterRoBERTa)
        num_labels: number of binary output heads (default: 8)
        hidden_dim: intermediate projection size (default: 256)
        dropout:    dropout rate after projection (default: 0.3)
        freeze_base: if True, encoder weights are frozen (useful for quick probing)
    """

    def __init__(
        self,
        base_model:  str   = BASE_MODEL,
        num_labels:  int   = NUM_LABELS,
        hidden_dim:  int   = 256,
        dropout:     float = 0.3,
        freeze_base: bool  = False,
    ) -> None:
        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model)

        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        encoder_dim = self.encoder.config.hidden_size  # 768 for roberta-base

        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns raw logits (B, num_labels). Apply sigmoid for probabilities.
        BCEWithLogitsLoss expects logits, not probabilities.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        logits   = self.classifier(cls_repr)             # (B, num_labels)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns sigmoid probabilities (B, num_labels) for inference."""
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.sigmoid(logits)


# ── Tokenizer factory ─────────────────────────────────────────────────────────

def get_tokenizer(base_model: str = BASE_MODEL) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(base_model)


# ── Convenience: load a saved checkpoint ─────────────────────────────────────

def load_from_checkpoint(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
) -> tuple[ContentModerationModel, dict]:
    """
    Load model + metadata from a .pt checkpoint saved by train_content_moderation.py.

    Returns (model, config_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt["model_config"]

    model = ContentModerationModel(
        base_model  = cfg.get("base_model", BASE_MODEL),
        num_labels  = cfg.get("num_labels", NUM_LABELS),
        hidden_dim  = cfg.get("hidden_dim", 256),
        dropout     = cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg
