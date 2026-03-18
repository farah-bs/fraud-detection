"""
Extraction des features de graphe depuis Neo4j

Pour chaque utilisateur, on calcule :
  followers_number                  — nombre de followers
  followings_number                 — nombre de followings
  ratio_following_followers         — ratio entre les deux (nombre de following / nombre de followers)
  posts_number                      — nombre de posts
  posts_number_per_day              — nombre de posts par jour
  is_verified                       — compte vérifié (0/1)
  account_age_days                  — ancienneté du compte (normalisée)
  report_count                      — nombre de signalements reçus
  link_number_per_post              — nombre de liens (url) pour les posts
  mention_number_per_post           — nombre de mentions @username dans un post

Ces 8 valeurs constituent le vecteur de features x_i de chaque nœud
alimentant le GNN (node_feature_dim = 8 dans config.py)
"""

from __future__ import annotations

from datetime import datetime, timezone
import re

import numpy as np
import psycopg2
import torch
from neo4j import GraphDatabase
from sqlalchemy.orm import subqueryload, Session
from torch_geometric.data import Data
from tqdm import tqdm
import logging

from config import POSTGRES_CONFIG_TRAIN, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from core.database import get_db_train
from entities.prod.user_prod import UserProd
from entities.train.post_train import PostTrain
from entities.train.user_train import UserTrain


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Connexions ────────────────────────────────────────────────────────────────

def _pg_conn():
    return psycopg2.connect(**POSTGRES_CONFIG_TRAIN)

def _neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Requêtes Neo4j ────────────────────────────────────────────────────────────

DEGREE_QUERY = """
MATCH (u:User)
OPTIONAL MATCH (u)-[:FOLLOWS]->(out)
OPTIONAL MATCH (inc)-[:FOLLOWS]->(u)
RETURN u.id AS id,
       count(DISTINCT out) AS followings,
       count(DISTINCT inc) AS followers
"""

DEGREE_QUERY_PROD = """
MATCH (u:UserNode)
OPTIONAL MATCH (u)-[:CONNECTED_TO]->(out)
OPTIONAL MATCH (inc)-[:CONNECTED_TO]->(u)
RETURN u.id AS id,
       count(DISTINCT out) AS followings,
       count(DISTINCT inc) AS followers
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
MATCH (u:User)
// 1. Récupération du degré sortant (remplacement de size() par COUNT {} pour Neo4j 5+)
WITH u, COUNT { (u)-[:FOLLOWS]->() } AS deg

// 2. Élimination précoce : pas de triangles possibles si deg < 2
WHERE deg > 1

// 3. Recherche des triangles dirigés (u->v->w->u) uniquement pour les nœuds pertinents
MATCH (u)-[:FOLLOWS]->(v)-[:FOLLOWS]->(w)
WHERE (w)-[:FOLLOWS]->(u)

WITH u, deg, count(*) AS triangles

RETURN u.id AS id,
       toFloat(triangles) / (deg * (deg - 1)) AS clustering_coeff
"""


def _fetch_neo4j(query: str, driver) -> dict[int, dict]:
    with driver.session() as session:
        result = session.run(query)
        return {row["id"]: dict(row) for row in result}


# ── Feature extraction complète ───────────────────────────────────────────────

def extract_features_prod_database(db: Session) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    log.info("Extraction des données depuis PostgreSQL...")

    users = db.query(UserProd).options(subqueryload(UserProd.posts)).all()

    index_by_id = {user.id: index for index, user in enumerate(users)}

    log.info(f"{len(users)} utilisateurs extraits de PostgreSQL.")

    driver = _neo4j_driver()

    log.info("Extraction des données depuis Neo4j et création de la liste d'arrêtes...")

    degrees   = _fetch_neo4j(DEGREE_QUERY_PROD,    driver)

    # Arêtes
    with driver.session() as session:
        raw_edges = session.run(
            "MATCH (a:UserNode)-[:CONNECTED_TO]->(b:UserNode) RETURN a.id AS src, b.id AS dst"
        ).data()
    driver.close()

    current_date = datetime.now(timezone.utc)

    feature_rows = []

    for index, user in tqdm(enumerate(users)):
        number_of_followings   = degrees.get(user.id, {}).get("followings", 0)
        number_of_followers    = degrees.get(user.id, {}).get("followers", 0)

        ratio_following_followers = (number_of_followings / number_of_followers) if number_of_followers > 0 else 0.0

        account_age_days = (current_date - user.created_at).days

        total_links = 0
        total_mentions = 0
        posts_count = len(user.posts)

        for post in user.posts:
            if post.content:
                # Regex pour trouver les URLs (http, https, www)
                links = re.findall(r'https?://\S+|www\.\S+', post.content)
                # Regex pour trouver les mentions @username
                mentions = re.findall(r'@\w+', post.content)

                total_links += len(links)
                total_mentions += len(mentions)

        if posts_count > 0:
            avg_links = total_links / posts_count
            avg_mentions = total_mentions / posts_count
            posts_per_day = posts_count / account_age_days if account_age_days > 0 else 0.0
        else:
            avg_links = 0.0
            avg_mentions = 0.0
            posts_per_day = 0.0

        feature_rows.append([
            number_of_followers, # followers_number
            number_of_followings, # followings_number
            ratio_following_followers, # ratio_following_followers
            posts_count, # posts_number
            posts_per_day, # posts_number_per_day
            user.is_verified, # is_verified
            account_age_days, # account_age_days
            user.report_count, # report_count
            avg_links, # link_number_per_post
            avg_mentions, # mention_number_per_post
        ])

    node_features = np.array(feature_rows, dtype=np.float32)

    # ── Normalisation min-max par colonne ──────────────────────────────────────
    col_min = node_features.min(axis=0)
    col_max = node_features.max(axis=0)
    denom   = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    node_features = (node_features - col_min) / denom

    # ── Edge index (indices locaux) ────────────────────────────────────────────
    src_list, dst_list = [], []
    for e in raw_edges:
        if e["src"] in index_by_id and e["dst"] in index_by_id:
            src_list.append(index_by_id[e["src"]])
            dst_list.append(index_by_id[e["dst"]])

    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    print(f"[Features] {len(index_by_id)} nœuds, {edge_index.shape[1]} arêtes extraits.")

    return node_features, edge_index, index_by_id


def extract_features_train_database() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    db_train = next(get_db_train())

    log.info("Extraction des données depuis PostgreSQL...")

    # 2 minutes environ pour récupérer 40 000 utilisateurs, c'est supportable
    result = db_train.query(UserTrain).options(subqueryload(UserTrain.posts)).all()

    log.info(f"{len(result)} utilisateurs extraits de PostgreSQL.")

    # ── Données Neo4j ──────────────────────────────────────────────────────────
    # Trop long malheureusement (+10 minutes sans avoir fini le traitement), il faudrait utiliser GDS (Graph Data Science) un plugin neo4j
    # clusterings = _fetch_neo4j(CLUSTERING_QUERY, driver)

    # log.info(f"Récupération des {len(clusterings)} clusters")

    current_date = datetime.now(timezone.utc)

    feature_rows = []
    label_rows   = []

    index_by_uuid = {str(user.uuid): index for index, user in enumerate(result)}

    for index, user in tqdm(enumerate(result)):
        uuid = user.uuid

        # cl = clusterings.get(uuid, {})

        # clust = cl.get("clustering_coeff", 0.0)

        number_of_followers = user.number_of_followers
        number_of_followings = user.number_of_followings

        ratio_following_followers = (number_of_followings / number_of_followers) if number_of_followers > 0 else 0.0

        account_age_days = (current_date - user.created_at).days

        total_links = 0
        total_mentions = 0
        posts_count = len(user.posts)

        for post in user.posts:
            if post.content:
                # Regex pour trouver les URLs (http, https, www)
                links = re.findall(r'https?://\S+|www\.\S+', post.content)
                # Regex pour trouver les mentions @username
                mentions = re.findall(r'@\w+', post.content)

                total_links += len(links)
                total_mentions += len(mentions)

        if posts_count > 0:
            avg_links = total_links / posts_count
            avg_mentions = total_mentions / posts_count
            posts_per_day = posts_count / account_age_days if account_age_days > 0 else 0.0
        else:
            avg_links = 0.0
            avg_mentions = 0.0
            posts_per_day = 0.0

        feature_rows.append([
            number_of_followers, # followers_number
            number_of_followings, # followings_number
            ratio_following_followers, # ratio_following_followers
            posts_count, # posts_number
            posts_per_day, # posts_number_per_day
            # clust, # clustering_coeff
            user.is_verified, # is_verified
            account_age_days, # account_age_days
            user.report_count, # report_count
            avg_links, # link_number_per_post
            avg_mentions, # mention_number_per_post
        ])
        label_rows.append(user.is_fraud)

    node_features = np.array(feature_rows, dtype=np.float32)
    labels = np.array(label_rows, dtype=np.int64)

    # ── Normalisation min-max par colonne ──────────────────────────────────────
    col_min = node_features.min(axis=0)
    col_max = node_features.max(axis=0)
    denom = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
    node_features = (node_features - col_min) / denom

    # ── Edge index (indices locaux) ────────────────────────────────────────────
    driver = _neo4j_driver()

    log.info("Extraction des données depuis Neo4j et création de la liste d'arrêtes...")

    all_uuids = list(index_by_uuid.keys())
    batch_size = 5000

    src_list, dst_list = [], []

    # Arêtes
    with driver.session() as session:
        for i in tqdm(range(0, len(all_uuids), batch_size)):
            batch_uuids = all_uuids[i:i + batch_size]

            batch_result = session.run(
                """
                    MATCH (a:User)-[:FOLLOWS]->(b:User)
                    WHERE a.user_id IN $user_ids
                    RETURN a.user_id AS src, b.user_id AS dst
                    """,
                user_ids=batch_uuids
            ).data()

            for e in batch_result:
                if e["src"] in index_by_uuid and e["dst"] in index_by_uuid:
                    src_list.append(index_by_uuid[e["src"]])
                    dst_list.append(index_by_uuid[e["dst"]])

    driver.close()

    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    print(f"[Features] {len(index_by_uuid)} nœuds, {edge_index.shape[1]} arêtes extraits.")
    print(f"           Frauduleux : {labels.sum()} / Normaux : {(labels == 0).sum()}")

    return node_features, edge_index, labels, index_by_uuid


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


def build_graph(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
) -> Data:
    x          = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index,    dtype=torch.long)
    y          = torch.tensor(labels,        dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


if __name__ == "__main__":
    node_features, edge_index, labels, index_by_uuid = extract_features_train_database()

    print("Node features shape:", node_features.shape)
    print("Edge index shape:", edge_index.shape)
    print("Labels shape:", labels.shape)

    data = build_graph(node_features, edge_index, labels)

    tensor_list = [data.x, data.edge_index, data.y]

    torch.save(tensor_list, "graph_features.pt")
