"""
Extraction des features de graphe depuis Neo4j

Pour chaque utilisateur, on calcule :
  degree_out          — nombre de connexions sortantes (KNOWS)
  degree_in           — nombre de connexions entrantes
  msg_sent            — messages envoyés
  msg_received        — messages reçus
  clustering_coeff    — coefficient de clustering local (triangles / paires de voisins)
  is_verified         — compte vérifié (0/1)
  account_age_days    — ancienneté du compte (normalisée)
  report_count        — nombre de signalements reçus

Ces 8 valeurs constituent le vecteur de features x_i de chaque nœud
alimentant le GNN (node_feature_dim = 8 dans config.py)
"""

from __future__ import annotations

import numpy as np
import psycopg2
from neo4j import GraphDatabase

from config import POSTGRES_CONFIG, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# ── Connexions ────────────────────────────────────────────────────────────────

def _pg_conn():
    return psycopg2.connect(**POSTGRES_CONFIG)

def _neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Requêtes Neo4j ────────────────────────────────────────────────────────────

DEGREE_QUERY = """
MATCH (u:User)
OPTIONAL MATCH (u)-[:KNOWS]->(out)
OPTIONAL MATCH (inc)-[:KNOWS]->(u)
RETURN u.id AS id,
       count(DISTINCT out) AS degree_out,
       count(DISTINCT inc) AS degree_in
"""

MSG_QUERY = """
MATCH (u:User)
OPTIONAL MATCH (u)-[s:SENT_MSG]->()
OPTIONAL MATCH ()-[r:SENT_MSG]->(u)
RETURN u.id AS id,
       coalesce(sum(s.count), 0) AS msg_sent,
       coalesce(sum(r.count), 0) AS msg_received
"""

CLUSTERING_QUERY = """
MATCH (u:User)-[:KNOWS]->(v:User)-[:KNOWS]->(w:User)-[:KNOWS]->(u)
WITH u, count(*) AS triangles
MATCH (u)-[:KNOWS]->(neighbor)
WITH u, triangles, count(DISTINCT neighbor) AS deg
RETURN u.id AS id,
       CASE WHEN deg > 1
            THEN toFloat(triangles) / (deg * (deg - 1))
            ELSE 0.0
       END AS clustering_coeff
"""


def _fetch_neo4j(query: str, driver) -> dict[int, dict]:
    with driver.session() as session:
        result = session.run(query)
        return {row["id"]: dict(row) for row in result}


# ── Feature extraction complète ───────────────────────────────────────────────

def extract_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Retourne :
      node_features  : np.ndarray (N, 8)   — features normalisées
      edge_index     : np.ndarray (2, E)   — arêtes [src, dst] (indices locaux)
      labels         : np.ndarray (N,)     — 1=fraude, 0=normal
      user_ids       : list[int]           — identifiants originaux (même ordre que node_features)
    """
    driver = _neo4j_driver()

    # ── Données Neo4j ──────────────────────────────────────────────────────────
    degrees   = _fetch_neo4j(DEGREE_QUERY,    driver)
    msgs      = _fetch_neo4j(MSG_QUERY,       driver)
    clusterings = _fetch_neo4j(CLUSTERING_QUERY, driver)

    # Arêtes
    with driver.session() as session:
        raw_edges = session.run(
            "MATCH (a:User)-[:KNOWS]->(b:User) RETURN a.id AS src, b.id AS dst"
        ).data()
    driver.close()

    # ── Données PostgreSQL ─────────────────────────────────────────────────────
    conn = _pg_conn()
    cur  = conn.cursor()
    cur.execute(
        "SELECT id, is_verified, created_at, report_count, is_fraud FROM users"
    )
    pg_rows = {row[0]: row for row in cur.fetchall()}
    cur.close()
    conn.close()

    # ── Construction de la matrice de features ─────────────────────────────────
    from datetime import datetime, timezone

    now    = datetime.now(timezone.utc)
    all_ids = sorted(set(degrees.keys()) | set(pg_rows.keys()))
    id2idx  = {uid: i for i, uid in enumerate(all_ids)}

    feature_rows = []
    label_rows   = []

    for uid in all_ids:
        d  = degrees.get(uid, {})
        m  = msgs.get(uid, {})
        cl = clusterings.get(uid, {})
        pg = pg_rows.get(uid)

        deg_out   = d.get("degree_out",  0)
        deg_in    = d.get("degree_in",   0)
        sent      = m.get("msg_sent",    0)
        received  = m.get("msg_received",0)
        clust     = cl.get("clustering_coeff", 0.0)

        if pg:
            is_verified  = float(pg[1])
            created_at   = pg[2]
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_days     = (now - created_at).days
            report_count = float(pg[3])
            is_fraud     = int(pg[4])
        else:
            is_verified  = 0.0
            age_days     = 0
            report_count = 0.0
            is_fraud     = 0

        feature_rows.append([
            float(deg_out),
            float(deg_in),
            float(sent),
            float(received),
            float(clust),
            is_verified,
            float(age_days),
            report_count,
        ])
        label_rows.append(is_fraud)

    node_features = np.array(feature_rows, dtype=np.float32)
    labels        = np.array(label_rows,   dtype=np.int64)

    # ── Normalisation min-max par colonne ──────────────────────────────────────
    col_min = node_features.min(axis=0)
    col_max = node_features.max(axis=0)
    denom   = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    node_features = (node_features - col_min) / denom

    # ── Edge index (indices locaux) ────────────────────────────────────────────
    src_list, dst_list = [], []
    for e in raw_edges:
        if e["src"] in id2idx and e["dst"] in id2idx:
            src_list.append(id2idx[e["src"]])
            dst_list.append(id2idx[e["dst"]])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    print(f"[Features] {len(all_ids)} nœuds, {edge_index.shape[1]} arêtes extraits.")
    print(f"           Frauduleux : {labels.sum()} / Normaux : {(labels == 0).sum()}")

    return node_features, edge_index, labels, all_ids
