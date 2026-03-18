"""
Ajout des données du dataset de honeypot social dans une base de données PostgreSQL.

Structure du dataset attendue :
- content_polluters.txt       : user_id | date_created | date_collected | nombre_followings | nombre_followers | nombre_tweets | longueur_screen_name | longueur_description
- legitimate_users.txt        : user_id | date_created | date_collected | nombre_followings | nombre_followers | nombre_tweets | longueur_screen_name | longueur_description
- content_polluters_tweets.txt : user_id | tweet_id | content | date_created
- legitimate_users_tweets.txt  : user_id | tweet_id | content | date_created

Les user_id des utilisateurs seront stockés dans la colonne uuid de la table users.
"""

from itertools import islice
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

from config import POSTGRES_CONFIG, DATASET_DIR

# ── PostgreSQL ───────────────────────────────────────────────────────────────

PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    uuid          BIGINT UNIQUE NOT NULL, -- identifiant d'origine du dataset
    created_at    TIMESTAMP NOT NULL,
    is_verified   BOOLEAN NOT NULL DEFAULT FALSE,
    report_count  INTEGER NOT NULL DEFAULT 0,
    -- Il est normalement préférable de calculer les valeurs plutôt que les stocker,
    -- cependant on a pas le vrai nombre de followers/followings donc ça peut être une information intéressante
    number_of_followings INTEGER NOT NULL DEFAULT 0,
    number_of_followers  INTEGER NOT NULL DEFAULT 0,
    is_fraud      BOOLEAN NOT NULL DEFAULT FALSE   -- label ground-truth
);

CREATE TABLE IF NOT EXISTS posts (
    id            SERIAL PRIMARY KEY,
    uuid          BIGINT UNIQUE NOT NULL,
    user_id       INTEGER NOT NULL REFERENCES users(id),
    content       TEXT NOT NULL,
    created_at    TIMESTAMP NOT NULL
);
"""


def get_database_connection() -> psycopg2.extensions.connection:
    """Retourne une connexion à la base de données PostgreSQL en utilisant les paramètres de configuration."""

    database=POSTGRES_CONFIG["dbname"]
    user=POSTGRES_CONFIG["user"]
    password=POSTGRES_CONFIG["password"]
    host=POSTGRES_CONFIG["host"]
    port=POSTGRES_CONFIG["port"]

    return psycopg2.connect(database=database, user=user, password=password, host=host, port=port)


def create_database_schema():
    """Crée le schéma de la base de données PostgreSQL pour stocker les utilisateurs et leurs posts."""

    conn = get_database_connection()
    cur  = conn.cursor()

    # Suppression des données existantes
    print(f"[PG] Suppression des tables existantes (si elles existent)")
    cur.execute("DROP TABLE IF EXISTS posts")
    cur.execute("DROP TABLE IF EXISTS users")
    conn.commit()

    # Création du schéma
    cur.execute(PG_SCHEMA)
    conn.commit()
    cur.close()
    conn.close()
    print(f"[PG] Schéma de base de données créé ou vérifié avec succès.")


def dataset_file(filename: str) -> str:
    return str(Path(DATASET_DIR) / filename)


def populate_postgres_users(batch_size = 1000) -> None:
    """Insère les utilisateurs du dataset de honeypot social dans la base de données PostgreSQL."""

    conn = get_database_connection()
    cur  = conn.cursor()

    print("Insertions des utilisateurs légitimes")

    legitimate_user_file_path = dataset_file("legitimate_users.txt")

    lines_number = count_lines(legitimate_user_file_path)

    print(f"Lecture de {lines_number} lignes par tranche de {batch_size}")

    for index in tqdm(range(0, lines_number, batch_size)):
        users = extract_users_from_dataset(legitimate_user_file_path, is_polluters=False, start_index=index, end_index=index + batch_size)

        insert_users_in_dataset(users, cur, conn, batch_size)

    print("Insertions des utilisateurs polleurs")

    polluters_user_file_path = dataset_file("content_polluters.txt")

    lines_number = count_lines(polluters_user_file_path)

    print(f"Lecture de {lines_number} lignes par tranche de {batch_size}")

    for index in tqdm(range(0, lines_number, batch_size)):
        users = extract_users_from_dataset(polluters_user_file_path, is_polluters=True, start_index=index,
                                           end_index=index + batch_size)

        insert_users_in_dataset(users, cur, conn, batch_size)

    cur.close()
    conn.close()


def populate_postgres_posts(batch_size = 1000) -> None:
    """Insère les posts des utilisateurs dans la base de données PostgreSQL."""

    conn = get_database_connection()
    cur  = conn.cursor()

    posts_legitimate_users_file_path = dataset_file("legitimate_users_tweets.txt")

    print("Insertions des posts des utilisateurs légitimes")

    lines_number = count_lines(posts_legitimate_users_file_path)

    print(f"Lecture de {lines_number} lignes par tranche de {batch_size}")

    for index in tqdm(range(0, lines_number, batch_size)):
        posts = extract_tweets_from_dataset(posts_legitimate_users_file_path, start_index=index, end_index=index + batch_size)

        insert_posts_in_dataset(posts, cur, conn, batch_size)

    posts_polluters_users_file_path = dataset_file("content_polluters_tweets.txt")

    print("Insertions des posts des utilisateurs pollueurs")

    lines_number = count_lines(posts_polluters_users_file_path)

    print(f"Lecture de {lines_number} lignes par tranche de {batch_size}")

    for index in tqdm(range(0, lines_number, batch_size)):
        posts = extract_tweets_from_dataset(posts_polluters_users_file_path, start_index=index, end_index=index + batch_size)

        insert_posts_in_dataset(posts, cur, conn, batch_size)

    cur.close()
    conn.close()


def insert_users_in_dataset(users: list[dict], cur, current_connection: psycopg2.extensions.connection, batch_size: int) -> None:
    """Insère une liste d'utilisateurs dans la base de données PostgreSQL."""

    if not users:
        print("Aucun utilisateur à insérer")
        return

    rows = [
        (u["uuid"], u["created_at"], u["is_verified"], u["report_count"], u["is_fraud"])
        for u in users
    ]

    execute_values(
        cur,
        """INSERT INTO users (uuid, created_at, is_verified, report_count, is_fraud)
        VALUES %s
        ON CONFLICT (uuid) DO NOTHING
        """,
        rows,
        page_size=batch_size
    )
    current_connection.commit()


def insert_posts_in_dataset(posts: list[dict], cur, current_connection: psycopg2.extensions.connection, batch_size: int) -> None:
    """Insère une liste de posts dans la base de données PostgreSQL."""

    if not posts:
        print("Aucun post à insérer")
        return

    # Il faut récupérer les id des utilisateurs insérés pour faire le lien avec les posts
    cur.execute("SELECT id, uuid FROM users WHERE uuid IN %s", (tuple(set(p["user_id"] for p in posts)),))
    uuid_to_id = {uuid: id for id, uuid in cur.fetchall()}

    # Filtrer les posts dont l'user_id est inconnu
    rows = [
        (p["uuid"], uuid_to_id[p["user_id"]], p["content"], p["created_at"])
        for p in posts
        if p["user_id"] in uuid_to_id
    ]

    if len(rows) < len(posts):
        print(f"{len(posts) - len(rows)} posts ignorés car leur user_id est inconnu (non inséré dans users)")

    if not rows:
        print("Aucun post à insérer après filtrage des user_id inconnus")
        return

    execute_values(
        cur,
        """INSERT INTO posts (uuid, user_id, content, created_at)
        VALUES %s
        ON CONFLICT (uuid) DO NOTHING
        """,
        rows,
        page_size=batch_size
    )
    current_connection.commit()

# ── Extraction ────────────────────────────────────────────────────────────────────

def count_lines(path: str) -> int:
    """Retourne le nombre de lignes dans un fichier."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {dataset_path.resolve()}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def iter_lines_range(path: str, x: int, y: int):
    """Générateur qui lit les lignes d'un fichier de x à y (exclusif) et les retourne sous forme de tuples (index, ligne)."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {dataset_path.resolve()}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        # lit les lignes x, x+1, ..., y-1
        for i, line in enumerate(islice(f, x, y), start=x):
            yield i, line.rstrip("\n")


def extract_users_from_dataset(dataset_path: str, is_polluters: bool, start_index: int, end_index: int) -> list[dict]:
    """
    Extrait les utilisateurs du dataset de honeypot social et les formate pour l'insertion en base.

    :param dataset_path: chemin vers le fichier de données
    :param is_polluters: indique si les utilisateurs extraits sont des pollueurs (fraudeurs) ou non
    :param start_index: index de la première ligne à lire
    :param end_index: index de la dernière ligne à ne pas lire (exclusif)
    :return: liste de dictionnaires représentant les utilisateurs
    """
    users = []

    for idx, line in iter_lines_range(dataset_path, x=start_index, y=end_index):
        items = line.split("\t")

        number_of_followings = int(items[3])
        number_of_followers = int(items[4])
        number_of_tweets = int(items[5])

        is_verified = generate_is_verified(number_of_followers, is_polluters)
        report_count = generate_report_count(number_of_tweets, number_of_followers, number_of_followings, is_polluters)

        user = {
            "uuid": int(items[0]),
            "created_at": items[1],
            "number_of_followings": number_of_followings,
            "number_of_followers": number_of_followers,
            "number_of_tweets": number_of_tweets,
            "is_fraud": is_polluters,
            "is_verified": is_verified,
            "report_count": report_count,
        }

        users.append(user)

    return users


def extract_tweets_from_dataset(dataset_path: str, start_index: int, end_index: int) -> list[dict]:
    """
    Extrait les tweets du dataset de honeypot social et les formate pour l'insertion en base.

    :param dataset_path: chemin vers le fichier de données
    :param start_index: index de la première ligne à lire
    :param end_index: index de la dernière ligne à ne pas lire (exclusif)
    :return: liste de dictionnaires représentant les tweets
    """
    posts = []

    for idx, line in iter_lines_range(dataset_path, x=start_index, y=end_index):
        items = line.split("\t")

        post = {
            "user_id": int(items[0]),
            "uuid": int(items[1]),
            "content": items[2],
            "created_at": items[3],
        }

        posts.append(post)

    return posts

# ── Génération is_verified et report_count ────────────────────────────────────

rng = np.random.default_rng(42)

def generate_is_verified(n_followers: int, is_polluter: bool) -> bool:
    """
    Génère une valeur booléenne pour is_verified en fonction du nombre de followers et du statut de pollueur.

    :param n_followers: nombre de followers de l'utilisateur
    :param is_polluter: booléen indiquant si l'utilisateur est un pollueur ou non
    :return: booléen indiquant si l'utilisateur est vérifié ou non
    """
    if is_polluter:
        # Très rare, quasi indépendant des followers
        p = 0.0005
    else:
        # Dépend des followers (loi logistique)
        # ~0.1% pour <1k, ~1% pour 1k-10k, ~5% pour >10k
        log_followers = np.log1p(n_followers)
        p = 1 / (1 + np.exp(-(log_followers - 9)))  # sigmoid centrée ~8k followers
        p = np.clip(p * 0.06, 0.001, 0.07)  # plafond à 7%
    return bool(rng.random() < p)


def generate_report_count(n_tweets: int, n_followers: int, n_followings: int, is_polluter: bool) -> int:
    """
    Génère un nombre de signalements (report_count) en fonction du nombre de tweets, followers, followings et du statut de pollueur.

    :param n_tweets: nombre de tweets de l'utilisateur
    :param n_followers: nombre de followers de l'utilisateur
    :param n_followings: nombre de followings de l'utilisateur
    :param is_polluter: booléen indiquant si l'utilisateur est un pollueur ou non
    :return: nombre de signalements généré
    """

    if is_polluter:
        # Mu de base élevé, corrélé aux tweets et au ratio following/followers
        ratio = np.log1p(n_followings) / np.log1p(n_followers + 1)
        mu = 2 + 0.003 * n_tweets + 1.5 * ratio
        # Négatif binomial : r=2 = forte variance (longue queue)
        r = 2
        p_nb = r / (r + mu)
        counts = rng.negative_binomial(r, p_nb)
    else:
        # Mu faible, la plupart à 0
        mu = 0.05 + 0.0001 * n_tweets
        r = 0.5  # encore plus dispersé → beaucoup de 0
        p_nb = r / (r + mu)
        counts = rng.negative_binomial(r, p_nb)

    return counts


def main():
    create_database_schema()
    populate_postgres_users()
    # Il y a beaucoup de posts (plusieurs millions) on augmente donc la taille du batch size
    populate_postgres_posts(batch_size=5000)


if __name__ == "__main__":
    main()
