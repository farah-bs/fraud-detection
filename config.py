"""
Configuration centralisée : connexions PostgreSQL, Neo4j et paramètres du modèle.
"""

import os

# ── PostgreSQL ───────────────────────────────────────────────────────────────
POSTGRES_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB",   "social_app"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASS", "postgres"),
}

# ── Neo4j ────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASS", "password")

# ── Modèle GNN ───────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "node_feature_dim": 8,    # nombre de features par nœud (voir graph_features.py)
    "hidden_dim":       64,
    "num_layers":       3,
    "dropout":          0.3,
    "learning_rate":    1e-3,
    "epochs":           100,
    "batch_size":       32,
}

MODEL_PATH = os.getenv("MODEL_PATH", "model/gnn_fraud_detector.pt")

# ── Seuil de décision ────────────────────────────────────────────────────────
# Score de confiance au-dessus duquel un compte est considéré frauduleux
FRAUD_THRESHOLD = 0.6 # on peut l'ajuster selon les besoins
