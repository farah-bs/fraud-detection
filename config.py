"""
Configuration centralisée : connexions PostgreSQL, Neo4j et paramètres du modèle
"""

import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Variable d'environnement manquante : {name}\n"
            "Assurez-vous d'avoir copié .env.example en .env et de l'avoir rempli."
        )
    return value


# ── PostgreSQL ───────────────────────────────────────────────────────────────
POSTGRES_CONFIG = {
    "host":     _require("PG_HOST"),
    "port":     int(os.getenv("PG_PORT", "5432")),
    "dbname":   _require("PG_DB"),
    "user":     _require("PG_USER"),
    "password": _require("PG_PASS"),
}

# ── Neo4j ────────────────────────────────────────────────────────────────────
NEO4J_URI      = _require("NEO4J_URI")
NEO4J_USER     = _require("NEO4J_USER")
NEO4J_PASSWORD = _require("NEO4J_PASS")

# ── Modèle GNN ───────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "node_feature_dim": int(os.getenv("GNN_FEATURE_DIM",   8)),
    "hidden_dim":       int(os.getenv("GNN_HIDDEN_DIM",    64)),
    "num_layers":       int(os.getenv("GNN_NUM_LAYERS",    3)),
    "dropout":          float(os.getenv("GNN_DROPOUT",     0.3)),
    "learning_rate":    float(os.getenv("GNN_LR",          1e-3)),
    "epochs":           int(os.getenv("GNN_EPOCHS",        100)),
    "batch_size":       int(os.getenv("GNN_BATCH_SIZE",    32)),
}

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/gnn_fraud_detector.pt")

# ── Seuil de décision ────────────────────────────────────────────────────────
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", 0.6))
