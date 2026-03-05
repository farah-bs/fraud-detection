"""
Modèle GNN pour la détection de fraude.

Architecture :
  - GraphSAGE : agrège les voisins via mean-pooling, ce qui est robuste
    même avec des graphes de taille variable et des degrés hétérogènes.
  - La couche finale produit un score scalaire (logit) par nœud,
    passé dans une sigmoïde pour obtenir un "score de confiance".

Entrée  : (x, edge_index)
           x          — tensor (N, node_feature_dim)
           edge_index — tensor (2, E)  LongTensor
Sortie  : logits      — tensor (N,)   avant sigmoïde
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()  # Batch-norm après chaque couche cachée

        # Couche d'entrée
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Couches cachées intermédiaires
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Couche de sortie (→ 1 logit par nœud)
        self.convs.append(SAGEConv(hidden_channels, 1))

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Dernière couche : pas de ReLU ni dropout
        x = self.convs[-1](x, edge_index)
        return x.squeeze(-1)   # (N,)

    # ── helpers ───────────────────────────────────────────────────────────────

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités de fraude (sigmoïde appliquée)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return torch.sigmoid(logits)

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        threshold: float = 0.6,
    ) -> torch.Tensor:
        """Retourne 1 (fraude) ou 0 (normal) selon un seuil."""
        proba = self.predict_proba(x, edge_index)
        return (proba >= threshold).long()
