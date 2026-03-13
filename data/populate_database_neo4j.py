"""
Génération de liens FOLLOWS entre utilisateurs et import des données dans Neo4j
Dataset : Social Honeypot Twitter (Lee et al., 2011)

Structure du dataset attendue :
- content_polluters.txt       : user_id | date_created | date_collected | nombre_followings | nombre_followers | nombre_tweets | longueur_screen_name | longueur_description
- legitimate_users.txt        : user_id | date_created | date_collected | nombre_followings | nombre_followers | nombre_tweets | longueur_screen_name | longueur_description

Lien généré : (source)-[:FOLLOWS]->(destination)
"""
from pathlib import Path

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import logging

from tqdm import tqdm

from config import (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
                    DATASET_DIR, FOLLOW_SAMPLE_RATIO, RECIP_LEGITIMATE, RECIP_POLLUTER, BATCH_SIZE, RANDOM_SEED,
                    IMPORT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
POLLUTERS_FILE = "content_polluters.txt"
LEGITIMATE_FILE = "legitimate_users.txt"

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# 1. CHARGEMENT DU DATASET
# ─────────────────────────────────────────────
COLUMNS = ["user_id", "date_created", "date_collected", "following_count",
           "followers_count", "tweets_count", "length_of_screen_name", "length_of_description"]


def dataset_file(filename: str) -> str:
    return str(Path(DATASET_DIR) / filename)


def load_users() -> pd.DataFrame:
    log.info("Chargement des polluters...")
    polluters = pd.read_csv(dataset_file(POLLUTERS_FILE), sep="\t", header=None, names=COLUMNS)
    polluters["label"] = "polluter"

    log.info("Chargement des légitimes...")
    legit = pd.read_csv(dataset_file(LEGITIMATE_FILE), sep="\t", header=None, names=COLUMNS)
    legit["label"] = "legitimate"

    df = pd.concat([polluters, legit], ignore_index=True)

    # Nettoyage
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["followers_count"] = pd.to_numeric(df["followers_count"], errors="coerce").fillna(0).astype(int)
    df["following_count"] = pd.to_numeric(df["following_count"], errors="coerce").fillna(0).astype(int)

    log.info(f"Dataset chargé : {len(df)} utilisateurs "
             f"({(df.label == 'polluter').sum()} polluters, "
             f"{(df.label == 'legitimate').sum()} légitimes)")
    return df


# ─────────────────────────────────────────────
# 2. GÉNÉRATION DES LIENS
# ─────────────────────────────────────────────
def compute_destination_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Preferential attachment : la popularité (followers_count) détermine
    la probabilité d'être choisi comme destination.
    """
    weights = df["followers_count"].values.astype(float) + 1  # Laplace smoothing
    return weights / weights.sum()


def generate_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque utilisateur, simule FOLLOW_SAMPLE_RATIO × following_count
    liens sortants, pondérés par la popularité des destinations.
    Ajoute ensuite la réciprocité selon le label.
    """
    user_ids = df["user_id"].values
    dest_probs = compute_destination_weights(df)

    edges = []

    log.info("Génération des liens primaires (following)...")
    for i, row in df.iterrows():
        n_follows = max(1, int(row["following_count"] * FOLLOW_SAMPLE_RATIO))
        n_follows = min(n_follows, len(user_ids) - 1)

        # Exclure self-loop
        mask = user_ids != row["user_id"]
        cands = user_ids[mask]
        probs = dest_probs[mask]
        probs = probs / probs.sum()

        n_follows = min(n_follows, len(cands))
        targets = np.random.choice(cands, size=n_follows, replace=False, p=probs)

        for t in targets:
            edges.append((row["user_id"], t))

    log.info(f"  → {len(edges)} liens primaires générés")

    # ── Réciprocité ──────────────────────────
    log.info("Ajout de la réciprocité...")
    label_map = dict(zip(df["user_id"], df["label"]))
    recip_edges = []

    for (src, dst) in edges:
        rate = RECIP_POLLUTER if label_map.get(src) == "polluter" else RECIP_LEGITIMATE
        if np.random.random() < rate:
            recip_edges.append((dst, src))

    log.info(f"  → {len(recip_edges)} liens réciproques ajoutés")

    all_edges = pd.DataFrame(edges + recip_edges, columns=["source", "destination"])
    all_edges = all_edges.drop_duplicates()
    # Supprimer self-loops résiduels
    all_edges = all_edges[all_edges["source"] != all_edges["destination"]]

    log.info(f"Total liens uniques : {len(all_edges)}")
    return all_edges


# ─────────────────────────────────────────────
# 3. INJECTION DANS NEO4J
# ─────────────────────────────────────────────
class Neo4jLoader:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """
        Supprime tous les nœuds et relations de la base.
        Stratégie en deux étapes (Relations puis Nœuds) pour éviter les erreurs de mémoire
        sur les graphes denses (supernodes).
        """
        log.info("Nettoyage de la base de données...")

        # Étape 1 : Supprimer toutes les relations
        total_rels = 0
        while True:
            with self.driver.session() as session:
                # On limite la suppression des relations pour ne pas surcharger la RAM
                result = session.run("""
                    MATCH ()-[r]->()
                    WITH r LIMIT 50000
                    DELETE r
                    RETURN count(r) as deleted
                """)
                count = result.single()["deleted"]
                total_rels += count

                # Si 0 relations trouvées, on a fini l'étape 1
                if count == 0:
                    break
                log.info(f"  {count} relations supprimées (Total: {total_rels})")

        # Étape 2 : Supprimer tous les nœuds (maintenant isolés)
        total_nodes = 0
        while True:
            with self.driver.session() as session:
                # DELETE simple suffit car il n'y a plus de relations
                result = session.run("""
                    MATCH (n)
                    WITH n LIMIT 10000
                    DELETE n
                    RETURN count(n) as deleted
                """)
                count = result.single()["deleted"]
                total_nodes += count

                if count == 0:
                    break
                log.info(f"  {count} nœuds supprimés (Total: {total_nodes})")

        log.info("Base de données entièrement vidée")

    def create_constraints(self):
        """Index d'unicité sur user_id pour accélérer les MERGEs."""
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT IF NOT EXISTS
                FOR (u:User) REQUIRE u.user_id IS UNIQUE
            """)
            log.info("Contrainte d'unicité créée sur User.user_id")

    def load_users(self, df: pd.DataFrame):
        """
        Crée les nœuds User avec leurs propriétés et leur label métier.
        """
        records = df[[
            "user_id"
        ]].to_dict("records")

        query = """
        UNWIND $rows AS row
        MERGE (u:User {user_id: row.user_id})
        """

        self._batch_write(records, query, "nœuds User")

    def load_edges(self, edges_df: pd.DataFrame):
        """
        Crée les relations FOLLOWS entre utilisateurs.
        """
        records = edges_df.to_dict("records")

        query = """
        UNWIND $rows AS row
        MATCH (src:User {user_id: row.source})
        MATCH (dst:User {user_id: row.destination})
        MERGE (src)-[:FOLLOWS]->(dst)
        """

        self._batch_write(records, query, "liens FOLLOWS")

    def load_edges_from_csv(self, csv_filenames: list):
        """
        Injection des liens FOLLOWS via LOAD CSV de manière itérative.
        """
        log.info(f"Injection des liens FOLLOWS via {len(csv_filenames)} fichiers CSV...")

        for filename in tqdm(csv_filenames, desc="Import Neo4j"):
            query = f"""
            LOAD CSV WITH HEADERS FROM 'file:///{filename}' AS row
            CALL {{
                WITH row
                MATCH (src:User {{user_id: row.source}})
                MATCH (dst:User {{user_id: row.destination}})
                CREATE (src)-[:FOLLOWS]->(dst)
            }} IN TRANSACTIONS OF 50000 ROWS
            """

            with self.driver.session() as session:
                session.run(query)

        log.info("Liens FOLLOWS écrits avec succès")

    def _batch_write(self, records: list, query: str, label: str):
        total = len(records)
        log.info(f"Écriture de {total} {label} par batchs de {BATCH_SIZE}...")

        with self.driver.session() as session:
            # TODO tqdm
            for start in tqdm(range(0, total, BATCH_SIZE)):
                batch = records[start:start + BATCH_SIZE]
                session.run(query, rows=batch)
                log.info(f"  {min(start + BATCH_SIZE, total)}/{total}")

        log.info(f"✓ {label} écrits avec succès")

    def verify(self):
        with self.driver.session() as session:
            n_users = session.run("MATCH (u:User) RETURN count(u) AS n").single()["n"]
            n_edges = session.run("MATCH ()-[r:FOLLOWS]->() RETURN count(r) AS n").single()["n"]
            # Prend trop de temps à s'éxécuter
            # recip = session.run("""
            #     MATCH (a)-[:FOLLOWS]->(b)-[:FOLLOWS]->(a)
            #     RETURN count(*)/2 AS n
            # """).single()["n"]

        log.info("─── Vérification Neo4j ───────────────────")
        log.info(f"  Nœuds User     : {n_users:,}")
        log.info(f"  Liens FOLLOWS  : {n_edges:,}")
        # log.info(f"  Liens réciproques : {recip:,}")
        log.info("──────────────────────────────────────────")


# ─────────────────────────────────────────────
# 4. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def main():
    # Chargement
    df = load_users()
    edges = generate_edges(df)

    # Export dans le dossier monté dans Docker
    import_dir = Path(IMPORT_DIR)
    import_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = 100000
    total_edges = len(edges)
    num_chunks = (total_edges + chunk_size - 1) // chunk_size
    csv_filenames = []

    log.info(f"Exportation des liens en {num_chunks} fichiers de {chunk_size} lignes...")
    for i in tqdm(range(num_chunks), desc="Export CSV"):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_edges)
        chunk = edges.iloc[start:end]

        filename = f"generated_edges_{i:03d}.csv"
        file_path = import_dir / filename
        chunk.to_csv(file_path, index=False)
        csv_filenames.append(filename)

    log.info(f"Liens exportés dans {import_dir}")

    # Injection Neo4j
    loader = Neo4jLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        loader.create_constraints()
        loader.clear_database()
        loader.load_users(df)
        loader.load_edges_from_csv(csv_filenames)
        loader.verify()
    finally:
        loader.close()

    log.info("Pipeline terminé ✓")


if __name__ == "__main__":
    main()
