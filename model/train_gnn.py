"""
Pipeline d'entraînement du GNN de détection de fraude
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Ajoute la racine du projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import MODEL_CONFIG, MODEL_PATH
from model.graph_features import extract_features_train_database
from model.gnn_model import FraudGNN


# ── Construction du Data object PyG ──────────────────────────────────────────

def build_graph(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
) -> Data:
    x          = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index,    dtype=torch.long)
    y          = torch.tensor(labels,        dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


# ── Masques train / val / test ────────────────────────────────────────────────

def make_masks(
    n: int,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = np.arange(n)

    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, stratify=labels, random_state=42
    )
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_idx, test_idx  = train_test_split(
        temp_idx,
        train_size=val_ratio_adjusted,
        stratify=labels[temp_idx],
        random_state=42,
    )

    def to_mask(idx):
        m = torch.zeros(n, dtype=torch.bool)
        m[idx] = True
        return m

    return to_mask(train_idx), to_mask(val_idx), to_mask(test_idx)


# ── Boucle d'entraînement ─────────────────────────────────────────────────────

def train_epoch(
    model: FraudGNN,
    data: Data,
    optimizer,
    pos_weight: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)
    loss   = F.binary_cross_entropy_with_logits(
        logits[mask], data.y[mask], pos_weight=pos_weight
    )
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model: FraudGNN,
    data: Data,
    mask: torch.Tensor,
) -> dict:
    model.eval()
    logits = model(data.x, data.edge_index)
    proba  = torch.sigmoid(logits[mask]).cpu().numpy()
    y_true = data.y[mask].cpu().numpy().astype(int)
    y_pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else 0.0
    report = classification_report(y_true, y_pred, target_names=["Normal", "Fraud"],
                                   zero_division=0, output_dict=True)
    return {"auc": auc, "report": report}


# ── Point d'entrée ────────────────────────────────────────────────────────────

def main():
    cfg = MODEL_CONFIG

    my_file = Path("graph_features.pt")

    if not my_file.is_file():
        # 1. Extraction des features
        print("=== Extraction des features ===")
        node_features, edge_index, labels, user_ids = extract_features_train_database()

        # 2. Graphe PyG
        data = build_graph(node_features, edge_index, labels)

        tensor_list = [data.x, data.edge_index, data.y, labels, user_ids]

        torch.save(tensor_list, "graph_features.pt")
    else:
        tensor_list = torch.load("graph_features.pt", weights_only=False)

        data_x, data_edge_index, data_y, labels, user_ids = tensor_list

        data = Data(x=data_x, edge_index=data_edge_index, y=data_y)

    # 3. Masques
    train_mask, val_mask, test_mask = make_masks(len(user_ids), labels)
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    # 4. Pondération des classes (fraude minoritaire)
    n_pos  = labels[train_mask.numpy()].sum()
    n_neg  = (labels[train_mask.numpy()] == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float)
    print(f"\nClass weight pos={pos_weight.item():.2f} "
          f"(train : {n_pos} fraude / {n_neg} normal)")

    # 5. Modèle + optimiseur
    model = FraudGNN(
        in_channels     = cfg["node_feature_dim"],
        hidden_channels = cfg["hidden_dim"],
        num_layers      = cfg["num_layers"],
        dropout         = cfg["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    # 6. Entraînement
    print("\n=== Entraînement ===")
    best_val_auc   = 0.0
    best_state     = None
    patience_count = 0
    EARLY_STOP     = 20

    for epoch in range(1, cfg["epochs"] + 1):
        loss = train_epoch(model, data, optimizer, pos_weight, train_mask)
        val_metrics = evaluate(model, data, val_mask)
        scheduler.step(val_metrics["auc"])

        if val_metrics["auc"] > best_val_auc:
            best_val_auc   = val_metrics["auc"]
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            f1_fraud = val_metrics["report"]["Fraud"]["f1-score"]
            print(f"Epoch {epoch:3d} | loss={loss:.4f} | "
                  f"val_AUC={val_metrics['auc']:.4f} | "
                  f"val_F1_fraud={f1_fraud:.4f}")

        if patience_count >= EARLY_STOP:
            print(f"\nEarly stopping à l'époque {epoch}.")
            break

    # 7. Évaluation finale sur le test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, data, test_mask)
    print("\n=== Résultats Test ===")
    print(f"AUC : {test_metrics['auc']:.4f}")
    print(classification_report(
        data.y[test_mask].cpu().numpy().astype(int),
        (torch.sigmoid(model(data.x, data.edge_index)[test_mask]).detach().numpy() >= 0.5).astype(int),
        target_names=["Normal", "Fraud"],
        zero_division=0,
    ))

    # 8. Sauvegarde
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "model_config":     cfg,
        "user_ids":         user_ids,
        # "node_features":    node_features,
        # "edge_index":       edge_index,
        "labels":           labels,
        "val_auc":          best_val_auc,
    }, MODEL_PATH)
    print(f"\nModèle sauvegardé dans : {MODEL_PATH}")


if __name__ == "__main__":
    main()
