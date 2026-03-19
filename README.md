# AI Moderation and Fraud Detection

This repository contains two ML systems that can run together behind one FastAPI service:

1. A graph-based fraud detector for user accounts.
2. A tweet content moderation model for multi-label safety classification.

Both models are implemented in PyTorch and exposed through a unified API in [api/inference.py](api/inference.py).

## Project Overview

### Model 1: GNN Fraud Detector

The fraud model is defined in [model/gnn_model.py](model/gnn_model.py) and trained with [model/train_gnn.py](model/train_gnn.py).

- Architecture: GraphSAGE with batch normalization, ReLU, and dropout.
- Input: user-node feature matrix plus graph edges.
- Output: one fraud logit per user node (then sigmoid probability).
- Task: binary classification (fraud vs normal account).
- Main artifact: `model/gnn_fraud_detector.pt`.

Node feature vector (8 features) comes from both databases:

- Graph features from Neo4j: in/out degree, sent/received messages, clustering coefficient.
- Profile features from PostgreSQL: verification status, account age, report count.

### Model 2: Content Moderation Model

The moderation model is defined in [model/content_moderation_model.py](model/content_moderation_model.py) and trained with [model/train_content_moderation.py](model/train_content_moderation.py).

- Architecture: `cardiffnlp/twitter-roberta-base` encoder + custom MLP head.
- Head: Linear(768 -> 256) + GELU + Dropout + Linear(256 -> 8).
- Task: multi-label classification with `BCEWithLogitsLoss`.
- Output labels (8): toxic, severe_toxic, obscene, threat, insult, identity_hate, spam, safe.
- Main artifacts:
  - `checkpoints/content_moderation_best.pt`
  - `checkpoints/content_moderation_last.pt`

## Data Pipelines

### Pipeline A: Fraud Graph Data Pipeline

This pipeline creates and transforms relational + graph data into training tensors for the GNN.

1. Generate synthetic users and fraud patterns with [data/simulate_data.py](data/simulate_data.py).
2. Store tabular user metadata in PostgreSQL (`users` table).
3. Store network structure in Neo4j (`User` nodes, `KNOWS` / `SENT_MSG` edges).
4. Extract graph + profile features with [data/graph_features.py](data/graph_features.py).
5. Normalize features and build `edge_index` for PyTorch Geometric.
6. Create train/val/test masks and train GraphSAGE in [model/train_gnn.py](model/train_gnn.py).

Output of this pipeline:

- Trained GNN checkpoint.
- Cached graph tensors and user ID mapping inside the checkpoint for inference.

### Pipeline B: Content Moderation Data Pipeline

This pipeline starts from raw tweet files and builds an 8-label training dataset.

1. Read honeypot tweet dumps in [data/preprocess_tweets.py](data/preprocess_tweets.py).
2. Clean text (HTML removal, URL placeholder, mention anonymization, whitespace normalization).
3. Deduplicate and split into train/val/test CSV files.
4. Enrich data with pseudo-labels using `unitary/toxic-bert` in [data/pseudo_label.py](data/pseudo_label.py).
5. Create final hard labels:
   - Toxic categories from thresholded toxic-bert outputs.
   - `label_spam` from original source class.
   - `label_safe` when non-spam and no toxic class is active.
6. Train TwitterRoBERTa-based multi-label classifier in [model/train_content_moderation.py](model/train_content_moderation.py).

Output of this pipeline:

- `data/processed/tweets_*_labeled.csv`
- Moderation checkpoints in `checkpoints/`
- Training logs in `runs/content_moderation/`

## Unified API

API implementation: [api/inference.py](api/inference.py)

- System:
  - `GET /health`
- Fraud endpoints:
  - `GET /fraud/score/{user_id}`
  - `POST /fraud/score/batch`
  - `GET /fraud/top-suspicious`
- Moderation endpoint:
  - `POST /moderation/check`

Swagger docs are available at `http://localhost:8000/docs`.

## Quick Start

### Option 1: Docker (fraud pipeline + API)

```bash
cp .env.example .env
docker compose up postgres neo4j -d
docker compose --profile train up train --build
docker compose up api --build -d
```

### Option 2: Local training (content moderation)

```bash
pip install -r requirements.txt

# Build moderation dataset
python data/preprocess_tweets.py
python data/pseudo_label.py

# Train moderation model
python model/train_content_moderation.py
```

## Repository Structure

- [api/](api/): inference service.
- [data/](data/): simulation, preprocessing, pseudo-labeling, graph feature extraction.
- [model/](model/): model definitions and training scripts.
- [checkpoints/](checkpoints/): saved moderation checkpoints.
- [evaluation_results/](evaluation_results/): metrics and confusion matrices.
