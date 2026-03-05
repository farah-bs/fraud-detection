"""
Simulation de données hybrides pour l'entraînement du détecteur de fraude.

Génère :
  - des profils utilisateurs dans PostgreSQL
  - un graphe social dans Neo4j avec des patterns normaux et frauduleux
    * Clusters isolés (faux comptes qui ne se connectent qu'entre eux)
    * Anneaux de fraude  (cycles fermés sans connexion vers l'extérieur)
    * Comptes normaux    (réseau organique, degrés variés)
"""

import random
import string
from datetime import datetime, timedelta

import psycopg2
from neo4j import GraphDatabase

from config import POSTGRES_CONFIG, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ── Helpers ──────────────────────────────────────────────────────────────────

def rand_str(n=8):
    return "".join(random.choices(string.ascii_lowercase, k=n))

def rand_date(days_back=730):
    return datetime.now() - timedelta(days=random.randint(0, days_back))

# ── PostgreSQL ───────────────────────────────────────────────────────────────

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(64) UNIQUE NOT NULL,
    email         VARCHAR(128) UNIQUE NOT NULL,
    created_at    TIMESTAMP NOT NULL,
    is_verified   BOOLEAN NOT NULL DEFAULT FALSE,
    report_count  INTEGER NOT NULL DEFAULT 0,
    is_fraud      BOOLEAN NOT NULL DEFAULT FALSE   -- label ground-truth
);
"""

def populate_postgres(users: list[dict]) -> None:
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cur  = conn.cursor()
    cur.execute(PG_SCHEMA)

    insert = """
        INSERT INTO users (username, email, created_at, is_verified, report_count, is_fraud)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (username) DO NOTHING
    """
    for u in users:
        cur.execute(insert, (
            u["username"], u["email"], u["created_at"],
            u["is_verified"], u["report_count"], u["is_fraud"],
        ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"[PG]  {len(users)} utilisateurs insérés.")


# ── Neo4j ────────────────────────────────────────────────────────────────────

def populate_neo4j(users: list[dict], edges: list[tuple]) -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Contrainte d'unicité
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")

        # Nœuds
        session.run("UNWIND $users AS u MERGE (n:User {id: u.id}) SET n += u", users=users)

        # Relations KNOWS (connexions sociales)
        session.run("""
            UNWIND $edges AS e
            MATCH (a:User {id: e.src}), (b:User {id: e.dst})
            MERGE (a)-[:KNOWS]->(b)
        """, edges=[{"src": s, "dst": d} for s, d in edges])

        # Relations MESSAGE (simulation d'envoi de messages)
        msg_edges = random.sample(edges, k=min(len(edges) // 2, 500))
        session.run("""
            UNWIND $edges AS e
            MATCH (a:User {id: e.src}), (b:User {id: e.dst})
            MERGE (a)-[r:SENT_MSG]->(b)
            ON CREATE SET r.count = toInteger(rand()*20 + 1)
        """, edges=[{"src": s, "dst": d} for s, d in msg_edges])

    driver.close()
    print(f"[Neo4j] {len(users)} nœuds, {len(edges)} arêtes insérés.")


# ── Génération des patterns ───────────────────────────────────────────────────

def build_normal_users(n: int, id_offset: int = 0) -> tuple[list[dict], list[tuple]]:
    """Réseau social organique : distribution de degrés variée (loi puissance approximée)."""
    users = []
    for i in range(n):
        uid = id_offset + i
        users.append({
            "id":           uid,
            "username":     f"user_{uid}",
            "email":        f"user_{uid}@example.com",
            "created_at":   rand_date(730).isoformat(),
            "is_verified":  random.random() > 0.2,
            "report_count": random.randint(0, 2),
            "is_fraud":     False,
        })

    # Connexions aléatoires (graphe de Barabási-Albert simplifié)
    edges = set()
    ids = [u["id"] for u in users]
    for uid in ids:
        nb_friends = max(1, int(random.expovariate(1 / 5)))
        friends = random.sample([x for x in ids if x != uid], k=min(nb_friends, len(ids) - 1))
        for f in friends:
            edges.add((uid, f))

    return users, list(edges)


def build_isolated_cluster(size: int, id_offset: int) -> tuple[list[dict], list[tuple]]:
    """
    Cluster isolé : groupe de faux comptes très récents, non vérifiés,
    connectés uniquement entre eux.
    """
    users = []
    for i in range(size):
        uid = id_offset + i
        users.append({
            "id":           uid,
            "username":     f"fake_{rand_str()}_{uid}",
            "email":        f"fake_{uid}@disposable.io",
            "created_at":   rand_date(30).isoformat(),   # comptes très récents
            "is_verified":  False,
            "report_count": random.randint(1, 5),
            "is_fraud":     True,
        })

    ids = [u["id"] for u in users]
    # Tous connectés entre eux (clique)
    edges = [(a, b) for a in ids for b in ids if a != b]
    return users, edges


def build_fraud_ring(size: int, id_offset: int) -> tuple[list[dict], list[tuple]]:
    """
    Anneau de fraude : cycle fermé, aucune connexion vers l'extérieur.
    Comptes avec taux de messages très élevé entre eux.
    """
    users = []
    for i in range(size):
        uid = id_offset + i
        users.append({
            "id":           uid,
            "username":     f"ring_{rand_str()}_{uid}",
            "email":        f"ring_{uid}@tempmail.com",
            "created_at":   rand_date(60).isoformat(),
            "is_verified":  False,
            "report_count": random.randint(2, 8),
            "is_fraud":     True,
        })

    ids = [u["id"] for u in users]
    # Cycle : 0→1→2→...→n-1→0
    edges = [(ids[i], ids[(i + 1) % size]) for i in range(size)]
    # Quelques liens en diagonale pour densifier
    extra = random.sample(
        [(ids[i], ids[j]) for i in range(size) for j in range(size) if abs(i - j) > 1],
        k=min(size, len(ids) ** 2 // 4),
    )
    return users, edges + extra


# ── Point d'entrée ────────────────────────────────────────────────────────────

def simulate(
    n_normal: int = 500,
    n_clusters: int = 5,
    cluster_size: int = 10,
    n_rings: int = 4,
    ring_size: int = 8,
) -> None:
    all_users: list[dict] = []
    all_edges: list[tuple] = []
    offset = 0

    # Comptes normaux
    u, e = build_normal_users(n_normal, offset)
    all_users.extend(u); all_edges.extend(e); offset += n_normal

    # Clusters isolés
    for _ in range(n_clusters):
        u, e = build_isolated_cluster(cluster_size, offset)
        all_users.extend(u); all_edges.extend(e); offset += cluster_size

    # Anneaux de fraude
    for _ in range(n_rings):
        u, e = build_fraud_ring(ring_size, offset)
        all_users.extend(u); all_edges.extend(e); offset += ring_size

    random.shuffle(all_users)

    populate_postgres(all_users)
    populate_neo4j(all_users, all_edges)

    fraud_count = sum(1 for u in all_users if u["is_fraud"])
    print(f"\nTotal : {len(all_users)} utilisateurs "
          f"({fraud_count} frauduleux / {len(all_users) - fraud_count} normaux)")


if __name__ == "__main__":
    simulate()
