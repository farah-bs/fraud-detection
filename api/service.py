import logging
import re

from datetime import datetime, timezone

import numpy as np
import torch
from neo4j import GraphDatabase
from sqlalchemy import UUID
from sqlalchemy.orm import Session, subqueryload
from torch_geometric.data import Data
from tqdm import tqdm

from entities.prod.user_prod import UserProd
from entities.prod.post_prod import PostProd # Nécessaire pour le subqueryload des posts liés à chaque utilisateur

from config import POSTGRES_CONFIG_TRAIN, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


DEGREE_QUERY_PROD = """
MATCH (u:UserNode)
OPTIONAL MATCH (u)-[:CONNECTED_TO]->(out)
OPTIONAL MATCH (inc)-[:CONNECTED_TO]->(u)
RETURN u.id AS id,
       count(DISTINCT out) AS followings,
       count(DISTINCT inc) AS followers
"""


def _neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def _fetch_neo4j(query: str, driver) -> dict[int, dict]:
    with driver.session() as session:
        result = session.run(query)
        return {row["id"]: dict(row) for row in result}


def extract_features_prod_database(db: Session) -> tuple[np.ndarray, np.ndarray, dict[UUID, int]]:
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

    current_date = datetime.now(timezone.utc).date()

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
            if post.text:
                # Regex pour trouver les URLs (http, https, www)
                links = re.findall(r'https?://\S+|www\.\S+', post.text)
                # Regex pour trouver les mentions @username
                mentions = re.findall(r'@\w+', post.text)

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


def build_graph(
    node_features: np.ndarray,
    edge_index: np.ndarray,
) -> Data:
    x          = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index,    dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


