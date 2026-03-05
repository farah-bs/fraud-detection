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
