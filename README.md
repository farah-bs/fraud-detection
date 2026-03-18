# Détection de Fraude par GNN

Pipeline de détection de comptes frauduleux pour application mobile : bases de données hybrides (PostgreSQL + Neo4j) + Graph Neural Network (PyTorch).

## Bases de données

**PostgreSQL** : table `users`

| Colonne | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Identifiant |
| `username` | VARCHAR(64) | Nom d'utilisateur |
| `email` | VARCHAR(128) | Email |
| `created_at` | TIMESTAMP | Date de création |
| `is_verified` | BOOLEAN | Compte vérifié |
| `report_count` | INTEGER | Signalements reçus |
| `is_fraud` | BOOLEAN | Label ground-truth |

**Neo4j** : nœuds `User`, relations `KNOWS` et `SENT_MSG { count }`

## Lancement

```bash
# 1. Configurer l'environnement
cp .env.example .env

# 2. Démarrer les bases de données
docker compose up postgres neo4j -d

# 3. Simuler les données + entraîner le modèle
docker compose --profile train up train --build

# 4. Lancer l'API
docker compose up api --build -d
```

## API

| Route | Description |
|---|---|
| `GET /health` | Statut |
| `GET /score/{user_id}` | Score de fraude [0,1] |
| `POST /score/batch` | Scores en masse |
| `GET /top-suspicious` | Top N comptes suspects |

Documentation interactive : `http://localhost:8000/docs`

## Lancer le projet en local
**Préambule** :
Pour pouvoir lancer le projet en local, il est nécessaire d'avoir une instance de PostgreSQL et de Neo4j en fonctionnement.
PostgreSQL étant relativement connu et facile à mettre en place, nous allons nous concentrer sur Neo4j.
Il est préférable d'utiliser **Docker** pour lancer Neo4j en local, car cela simplifie grandement le processus d'installation et de configuration.

### Télécharger le dataset
- Le lien du dataset est : https://infolab.tamu.edu/data/
- Il faut aller dans la section Social Honeypot Dataset et mettre les fichiers dans social_honeypot_dataset

### Lancer Neo4j en local sur un docker

- Sur votre disque (pour ma part, je l'ai fait dans mon dossier Users windows) créer un **dossier neo4j**. Créer ensuite un dossier data et import (2 volumes qui seront utilisés par le container).
- Créer un fichier `docker-compose.yml` dans ce dossier avec le contenu suivant :

⚠️ Changer le mot de passe pour neo4j et les chemins vers les volumes selon votre système d'exploitation (Windows, Linux, MacOS) et votre organisation de fichiers.

```yaml
services:
  neo4j:
    image: neo4j:2026.02.2
    restart: always
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/mot-de-passe-neo4j
    volumes:
      # Attention selon votre système d'exploitation, le chemin vers les volumes peut différer
      - chemin\complet\vers\neo4j\data:/data
      - chemin\complet\vers\neo4j\import:/var/lib/neo4j/import
```

- Ouvrir un terminal, se placer dans le dossier contenant le `docker-compose.yml` et lancer la commande suivante :

```bash
docker compose up -d
```

### Modifier le fichier .env

Dans le fichier `.env` du projet, modifier les variables d'environnement suivantes pour correspondre à votre configuration locale :

```env
DATASET_DIR=chemin\vers\le\dossier\du\dataset\fraud-detection\data\social_honeypot_dataset
IMPORT_DIR=chemin/vers/le/dossier/neo4j/import # Créé précédemment pour le volume d'import de Neo4j
```

### Peupler les bases de données
- Lancer le fichier [populate_database_postgresql.py](data/populate_database_postgresql.py) pour insérer les données du dataset dans PostgreSQL.
- Lancer le fichier [populate_database_neo4j.py](data/populate_database_neo4j.py) pour créer les relations en utilisateurs et insérer les données du dataset dans neo4j.

Plus ou moins, Farah en avait deja dev une bonne partie, je suis en train de les adapter pour le moment je peux te dire qu'il faut :
- Route pour le batch de score de fraude des utilisateurs :
  - Méthode : POST 
  - Route : "/fraud/score/batch"
  - Paramètre : Aucun
  - Body : Aucun
  - Réponse :
    ```[
        {
            "user_id":        int,
            "fraud_score":    float,
            "is_suspicious":  bool,
            "threshold":      float,
            "message":        string,
        },
        {
            ...
        }
    ]
    ```
    user_id : id de l'utilisateur
    fraud_score: float entre 0 et 1 qui annonce la probabilité d'utilisateur frauduleux
    is_suspicious: booléen basé sur le seuil d'alerte
    threshold: seuil d'alerte (entre 0 et 1)
    message: Si suspect : "Compte suspect : vérification supplémentaire recommandée." sinon "Compte normal."
- Appeler la route de ntore API en POST "/moderation/check" avec pour paramètre :
  - Méthode : POST 
  - Route : "/moderation/check"
  - Paramètre : Aucun
  - Body : 
    ```{
        "content_post": str,
        "post_id": int,
    }
    ```
  - Réponse :
    ```{
         "content_post":   string,
         "post_id":        int,
         "scores":         dict[str, float],
         "verdict":        str,
    }
    ```
    content_post: le contenu du post à modérer
    post_id: l'id du post à modérer
    scores: un dictionnaire avec les scores de chaque catégorie de modération ("toxic": float, "severe_toxic": float, "obscene": float, "threat": float, "insult": float, "identity_hate": float, "spam": float, "safe": float)
    verdict: (toxic, severe_toxic, obscene, threat, insult, identity_hate, spam, safe) un de ces mots selon le seuil de modération (paramétré à 0.5 par défaut)