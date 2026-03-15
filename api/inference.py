"""
API FastAPI d'inférence du GNN de détection de fraude
+ modération de contenu (TwitterRoBERTa multi-label).
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import uvicorn


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import FRAUD_THRESHOLD, MODEL_PATH, MODEL_CONFIG
from model.gnn_model import FraudGNN
from model.content_moderation_model import (
    LABEL_NAMES as MOD_LABEL_NAMES,
    ContentModerationModel,
    load_from_checkpoint,
)

MODERATION_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "checkpoints",
    "content_moderation_best.pt",
)
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


def _load_model() -> None:
    global _model, _data_x, _data_edge, _user_ids, _id2idx

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. "
            "Lancez d'abord : python model/train.py"
        )

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

    node_features = checkpoint["node_features"]
    edge_index    = checkpoint["edge_index"]
    _user_ids     = checkpoint["user_ids"]

    _data_x    = torch.tensor(node_features, dtype=torch.float)
    _data_edge = torch.tensor(edge_index,    dtype=torch.long)
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
    title="Fraud Detection API",
    description="Score de confiance de fraude basé sur GNN",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schémas ───────────────────────────────────────────────────────────────────

class FraudScoreResponse(BaseModel):
    user_id:        int
    fraud_score:    float   # probabilité [0, 1]
    is_suspicious:  bool    # True si score >= FRAUD_THRESHOLD
    threshold:      float
    message:        str


class BatchScoreRequest(BaseModel):
    user_ids: list[int]


class BatchScoreResponse(BaseModel):
    results: list[FraudScoreResponse]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_user(user_id: int) -> float:
    """Retourne le score de fraude [0,1] pour un utilisateur donné."""
    if user_id not in _id2idx:
        raise HTTPException(
            status_code=404,
            detail=f"Utilisateur {user_id} inconnu du graphe. "
                   "Relancez l'entraînement pour intégrer les nouveaux comptes.",
        )
    idx   = _id2idx[user_id]
    proba = _model.predict_proba(_data_x, _data_edge)
    return float(proba[idx].item())


def _build_response(user_id: int, score: float) -> FraudScoreResponse:
    suspicious = score >= FRAUD_THRESHOLD
    message = (
        "Compte suspect : vérification supplémentaire recommandée."
        if suspicious
        else "Compte normal."
    )
    return FraudScoreResponse(
        user_id       = user_id,
        fraud_score   = round(score, 4),
        is_suspicious = suspicious,
        threshold     = FRAUD_THRESHOLD,
        message       = message,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status":           "ok",
        "model_loaded":     _model is not None,
        "indexed_users":    len(_user_ids),
        "fraud_threshold":  FRAUD_THRESHOLD,
    }


@app.get("/score/{user_id}", response_model=FraudScoreResponse)
def get_fraud_score(user_id: int):
    """
    Calcule le score de fraude pour un utilisateur.
    Appelé par l'app mobile après chaque action sensible.
    """
    score = _score_user(user_id)
    return _build_response(user_id, score)


@app.post("/score/batch", response_model=BatchScoreResponse)
def get_fraud_scores_batch(request: BatchScoreRequest):
    """
    Calcule les scores de fraude pour une liste d'utilisateurs en une seule passe.
    Utile pour l'analyse côté admin ou lors d'un onboarding en masse.
    """
    results = []
    for uid in request.user_ids:
        try:
            score = _score_user(uid)
            results.append(_build_response(uid, score))
        except HTTPException as exc:
            results.append(FraudScoreResponse(
                user_id       = uid,
                fraud_score   = -1.0,
                is_suspicious = False,
                threshold     = FRAUD_THRESHOLD,
                message       = exc.detail,
            ))
    return BatchScoreResponse(results=results)


@app.get("/top-suspicious", response_model=BatchScoreResponse)
def get_top_suspicious(limit: int = 20):
    """
    Retourne les `limit` utilisateurs avec le score de fraude le plus élevé.
    Utile pour un tableau de bord de modération.
    """
    proba = _model.predict_proba(_data_x, _data_edge).numpy()
    top_indices = np.argsort(proba)[::-1][:limit]
    results = [
        _build_response(_user_ids[i], float(proba[i]))
        for i in top_indices
    ]
    return BatchScoreResponse(results=results)


# ── Content moderation endpoint ───────────────────────────────────────────────

class ModerationRequest(BaseModel):
    tweet: str


class ModerationResponse(BaseModel):
    tweet:   str
    scores:  dict[str, float]   # label → probability [0, 1]
    verdict: str                # highest-scoring harmful label, or "safe"


@app.post("/moderate", response_model=ModerationResponse)
def moderate_tweet(request: ModerationRequest):
    """
    Score a tweet across 8 moderation categories and return the top verdict.

    Categories: toxic | severe_toxic | obscene | threat | insult |
                identity_hate | spam | safe
    """
    if _mod_model is None or _mod_tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Content moderation model not loaded. "
                "Train it first: python model/train_content_moderation.py"
            ),
        )

    tweet = request.tweet.strip()
    if not tweet:
        raise HTTPException(status_code=422, detail="tweet text must not be empty")

    enc = _mod_tokenizer(
        tweet,
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

    return ModerationResponse(tweet=tweet, scores=scores, verdict=verdict)


# ── Lancement ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api.inference:app", host="0.0.0.0", port=8000, reload=False)
