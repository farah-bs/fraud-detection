"""
Unified Inference API for Fraud Detection & Content Moderation

Two models integrated:
1. GNN Fraud Detector: Detects fraudulent accounts using graph neural networks
2. Content Moderation: Classifies tweets across 8 moderation categories

Endpoints organized by model type:
- /fraud/* → GNN fraud detection endpoints
- /moderation/* → Content moderation endpoints
- /health → API health & model status
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import sys
from contextlib import asynccontextmanager

import numpy as np
import torch
from uuid import UUID
from torch_geometric.data import Data
from fastapi import FastAPI, HTTPException, Depends
from neo4j import GraphDatabase
from pydantic import BaseModel
from sqlalchemy.orm import subqueryload, Session
from transformers import AutoTokenizer
import uvicorn

from core.database import get_db_prod
from entities.prod.user_prod import UserProd
from api.service import extract_features_prod_database, build_graph

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import FRAUD_THRESHOLD, MODEL_PATH, MODERATION_MODEL_PATH, MODEL_CONFIG, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from model.gnn_model import FraudGNN
from model.content_moderation_model import (
    LABEL_NAMES as MOD_LABEL_NAMES,
    ContentModerationModel,
    load_from_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# MODERATION_MODEL_PATH = os.path.join(
#     os.path.dirname(os.path.dirname(__file__)),
#     "checkpoints",
#     "content_moderation_best.pt",
# )
MODERATION_THRESHOLD = 0.5

# ── État global (chargé une seule fois au démarrage) ─────────────────────────

_model:         FraudGNN | None     = None
_data_x:        torch.Tensor | None = None
_data_edge:     torch.Tensor | None = None
_user_ids:      list[int]           = []
_id2idx:        dict[int, int]      = {}

_mod_model:     ContentModerationModel | None = None
_mod_tokenizer                                = None
_mod_device:    torch.device                  = torch.device("cpu")


def _neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _load_model() -> None:
    global _model, _data_x, _data_edge, _user_ids, _id2idx

    if not os.path.exists(MODEL_PATH):
        print(
            f"[API] GNN Fraud detection model not found: {MODEL_PATH}. "
            "Endpoint /score will be unavailable. "
            "Train it with: python model/train_gnn.py"
        )
        return

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("model_config", MODEL_CONFIG)

    _model = FraudGNN(
        in_channels     = cfg["node_feature_dim"],
        hidden_channels = cfg["hidden_dim"],
        num_layers      = cfg["num_layers"],
        dropout         = cfg["dropout"],
    )
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()

    # node_features = checkpoint["node_features"]
    # edge_index    = checkpoint["edge_index"]
    _user_ids     = checkpoint["user_ids"]

    # _data_x    = torch.tensor(node_features, dtype=torch.float)
    # _data_edge = torch.tensor(edge_index,    dtype=torch.long)
    _id2idx    = {uid: i for i, uid in enumerate(_user_ids)}

    print(f"[API] Modèle chargé — {len(_user_ids)} utilisateurs indexés.")


def _load_moderation_model() -> None:
    global _mod_model, _mod_tokenizer, _mod_device

    if not os.path.exists(MODERATION_MODEL_PATH):
        print(
            "[API] Modèle de modération introuvable — "
            "endpoint /moderate désactivé. "
            "Lancez d'abord: python data/preprocess_tweets.py && "
            "python data/pseudo_label.py && python model/train_content_moderation.py"
        )
        return

    _mod_device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    _mod_model, cfg = load_from_checkpoint(MODERATION_MODEL_PATH, device=_mod_device)
    base_model      = cfg.get("base_model", "cardiffnlp/twitter-roberta-base")
    _mod_tokenizer  = AutoTokenizer.from_pretrained(base_model)

    print(f"[API] Modèle de modération chargé [{_mod_device}]  {MODERATION_MODEL_PATH}")


# ── Lifespan (FastAPI ≥ 0.93) ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    _load_moderation_model()
    yield


app = FastAPI(
    title="Fraud Detection & Content Moderation API",
    description=(
        "Unified API for detecting fraudulent accounts (GNN) and moderating tweet content (RoBERTa). "
        "Two independent models: (1) GNN-based account fraud detection via /fraud/* endpoints, "
        "(2) TwitterRoBERTa multi-label content moderation via /moderation/* endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schémas ───────────────────────────────────────────────────────────────────

class FraudScoreResponse(BaseModel):
    userId:        UUID
    fraudScore:    float   # probabilité [0, 1]
    isSuspicious:  bool    # True si score >= FRAUD_THRESHOLD
    threshold:      float
    message:        str


class BatchScoreRequest(BaseModel):
    user_ids: list[int]


class BatchScoreResponse(BaseModel):
    results: list[FraudScoreResponse]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_user(user_id: int) -> float:
    """Retourne le score de fraude [0,1] pour un utilisateur donné."""
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="GNN fraud detection model not loaded. Train it first: python model/train.py",
        )
    if user_id not in _id2idx:
        raise HTTPException(
            status_code=404,
            detail=f"Utilisateur {user_id} inconnu du graphe. "
                   "Relancez l'entraînement pour intégrer les nouveaux comptes.",
        )
    idx   = _id2idx[user_id]
    proba = _model.predict_proba(_data_x, _data_edge)
    return float(proba[idx].item())


def score_users_batch(data: Data, index_by_id: dict[UUID, int]) -> dict[UUID, float]:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="GNN fraud detection model not loaded. Train it first: python model/train.py",
        )

    scores = _model.predict_proba(data.x, data.edge_index)

    score_by_user_id = {}

    for user_id, idx in index_by_id.items():
        score_by_user_id[user_id] = float(scores[idx].item())

    return score_by_user_id


def _build_response(user_id: UUID, score: float) -> FraudScoreResponse:
    suspicious = score >= FRAUD_THRESHOLD
    message = (
        "Compte suspect : vérification supplémentaire recommandée."
        if suspicious
        else "Compte normal."
    )
    return FraudScoreResponse(
        userId       = user_id,
        fraudScore   = round(score, 4),
        isSuspicious = suspicious,
        threshold     = FRAUD_THRESHOLD,
        message       = message,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """
    Check API health and model availability status.

    Returns status of both GNN fraud detection and content moderation models.
    """
    return {
        "status": "ok",
        "models": {
            "gnn_fraud_detection": {
                "loaded": _model is not None,
                "indexed_users": len(_user_ids),
                "threshold": FRAUD_THRESHOLD,
            },
            "content_moderation": {
                "loaded": _mod_model is not None,
                "threshold": MODERATION_THRESHOLD,
            },
        },
    }


@app.get("/fraud/score/{user_id}", response_model=FraudScoreResponse, tags=["GNN Fraud Detection"])
def get_fraud_score(user_id: int):
    """
    **GNN Model** - Get fraud score for a single user account.

    Uses Graph Neural Network to analyze user's network patterns, account features,
    and behavioral signals to detect fraudulent accounts.

    Returns:
    - `fraud_score`: Probability [0, 1] that account is fraudulent
    - `is_suspicious`: Boolean flag if score >= threshold
    - `threshold`: Decision threshold used
    - `message`: Human-readable verdict

    Example: `GET /fraud/score/12345`
    """
    score = _score_user(user_id)
    return _build_response(user_id, score)


@app.post("/fraud/score/batch", response_model=list[FraudScoreResponse], tags=["GNN Fraud Detection"])
def get_fraud_scores_batch(db: Session = Depends(get_db_prod)):
    """
    **GNN Model** - Get fraud scores for multiple user accounts in a single request.

    Efficiently score a batch of user IDs using the GNN model.
    Useful for admin dashboards, bulk user reviews, or onboarding analysis.

    Request body:
    ```json
    {
      "user_ids": [123, 456, 789]
    }
    ```

    Returns: List of fraud scores for each user ID.
    Failed queries return score=-1.0 with error details.
    """
    results = []

    node_features, edge_index, index_by_uuid = extract_features_prod_database(db)

    data = build_graph(node_features, edge_index)

    score_by_user_id = score_users_batch(data, index_by_uuid)

    for user_id, score in score_by_user_id.items():
        results.append(_build_response(user_id, score))

    return results


@app.get("/fraud/top-suspicious", response_model=BatchScoreResponse, tags=["GNN Fraud Detection"])
def get_top_suspicious(limit: int = 20):
    """
    **GNN Model** - Get top N most suspicious user accounts ranked by fraud score.

    Returns accounts with highest fraud probability scores.
    Useful for moderation dashboards, priority review lists, or risk assessment.

    Query parameters:
    - `limit`: Number of top accounts to return (default: 20, max: 1000)

    Example: `GET /fraud/top-suspicious?limit=50`
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="GNN fraud detection model not loaded. Train it first: python model/train_gnn.py",
        )
    proba = _model.predict_proba(_data_x, _data_edge).numpy()
    top_indices = np.argsort(proba)[::-1][:limit]
    results = [
        _build_response(_user_ids[i], float(proba[i]))
        for i in top_indices
    ]
    return BatchScoreResponse(results=results)


# ── Content moderation endpoint ───────────────────────────────────────────────

class ModerationRequest(BaseModel):
    content_post: str
    post_id: int


class ModerationResponse(BaseModel):
    content_post:   str
    post_id: int
    scores:  dict[str, float]   # label → probability [0, 1]
    verdict: str                # highest-scoring harmful label, or "safe"


@app.post("/moderation/check", response_model=ModerationResponse, tags=["Content Moderation"])
def moderate_tweet(request: ModerationRequest):
    """
    **Content Moderation Model** - Analyze tweet content across 8 moderation categories.

    Uses fine-tuned TwitterRoBERTa to detect harmful content:
    - **toxic**: General toxicity/hostility
    - **severe_toxic**: Severe harmful content
    - **obscene**: Obscene language
    - **threat**: Threats of violence
    - **insult**: Insults/name-calling
    - **identity_hate**: Hateful speech targeting groups
    - **spam**: Spam/promotional content
    - **safe**: Safe/benign content

    Returns probability scores [0, 1] for each category and overall verdict.

    Request body:
    ```json
    {
      "tweet": "Your tweet text to analyze"
    }
    ```

    Response includes:
    - `verdict`: Highest-scoring harmful label or "safe"
    - `scores`: Dict of all category probabilities
    - `tweet`: Original tweet text

    Example: `POST /moderation/check`
    """
    if _mod_model is None or _mod_tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Content moderation model not loaded. "
                "Train it first: python model/train_content_moderation.py"
            ),
        )

    post = request.content_post.strip()
    if not post:
        raise HTTPException(status_code=422, detail="tweet text must not be empty")

    enc = _mod_tokenizer(
        post,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(_mod_device) for k, v in enc.items()}

    probs = _mod_model.predict_proba(
        enc["input_ids"], enc["attention_mask"]
    ).squeeze(0).cpu().tolist()

    scores = {label: round(prob, 4) for label, prob in zip(MOD_LABEL_NAMES, probs)}

    # Verdict = highest-scoring harmful label (exclude "safe" from this ranking)
    harmful_labels = [l for l in MOD_LABEL_NAMES if l != "safe"]
    verdict = max(harmful_labels, key=lambda l: scores[l])
    if scores[verdict] < MODERATION_THRESHOLD:
        verdict = "safe"

    return ModerationResponse(content_post=post, post_id=request.post_id, scores=scores, verdict=verdict)


# ── Backward Compatibility Aliases (deprecated, redirects to new endpoints) ──

@app.get("/score/{user_id}", response_model=FraudScoreResponse, deprecated=True, tags=["GNN Fraud Detection (Deprecated)"])
def get_fraud_score_deprecated(user_id: int):
    """
    **DEPRECATED** - Use `/fraud/score/{user_id}` instead.

    This endpoint is maintained for backward compatibility only.
    All new integrations should use the `/fraud/` endpoints.
    """
    return get_fraud_score(user_id)


@app.post("/score/batch", response_model=BatchScoreResponse, deprecated=True, tags=["GNN Fraud Detection (Deprecated)"])
def get_fraud_scores_batch_deprecated(request: BatchScoreRequest):
    """
    **DEPRECATED** - Use `/fraud/score/batch` instead.

    This endpoint is maintained for backward compatibility only.
    All new integrations should use the `/fraud/` endpoints.
    """
    return get_fraud_scores_batch(request)


@app.get("/top-suspicious", response_model=BatchScoreResponse, deprecated=True, tags=["GNN Fraud Detection (Deprecated)"])
def get_top_suspicious_deprecated(limit: int = 20):
    """
    **DEPRECATED** - Use `/fraud/top-suspicious` instead.

    This endpoint is maintained for backward compatibility only.
    All new integrations should use the `/fraud/` endpoints.
    """
    return get_top_suspicious(limit)


@app.post("/moderate", response_model=ModerationResponse, deprecated=True, tags=["Content Moderation (Deprecated)"])
def moderate_tweet_deprecated(request: ModerationRequest):
    """
    **DEPRECATED** - Use `/moderation/check` instead.

    This endpoint is maintained for backward compatibility only.
    All new integrations should use the `/moderation/` endpoints.
    """
    return moderate_tweet(request)


# ── Lancement ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.inference:app", host="0.0.0.0", port=8000, reload=False)
