import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv, GraphConv, TAGConv
    from torch_geometric.data import Data
    _HAS_TORCH_GEOMETRIC = True
except Exception:  # pragma: no cover - optional dependency
    GCNConv = None
    GraphConv = None
    TAGConv = None
    Data = None
    _HAS_TORCH_GEOMETRIC = False
    import logging
    logging.getLogger(__name__).warning("torch_geometric not available — GNN modules will use fallbacks or be limited")
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SpatioTemporalGCN(nn.Module):

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, num_layers: int = 3,
                 dropout: float = 0.1, output_dim: int = 1):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gcn_layers = nn.ModuleList([
            GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.input_proj(x)
        x = F.relu(x)

        attention_weights = x.clone()

        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        anomaly_scores = self.anomaly_head(x)

        return anomaly_scores, attention_weights

class SpatioTemporalGraphAttention(nn.Module):

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, num_heads: int = 8,
                 num_layers: int = 3, dropout: float = 0.1, output_dim: int = 1):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim * num_heads)

        try:
            from torch_geometric.nn import GATConv
            self.gat_layers = nn.ModuleList([
                GATConv(hidden_dim * num_heads if i == 0 else hidden_dim * num_heads,
                       hidden_dim, heads=num_heads, dropout=dropout)
                for i in range(num_layers)
            ])
        except ImportError:
            logger.warning("torch_geometric.nn.GATConv not available, using GraphConv")
            self.gat_layers = nn.ModuleList([
                GraphConv(hidden_dim * num_heads, hidden_dim)
                for i in range(num_layers)
            ])

        self.dropout = nn.Dropout(dropout)

        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

        self.attention_head = nn.Linear(hidden_dim * num_heads, num_heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.input_proj(x)
        x = F.relu(x)

        attention_weights = None
        for gat in self.gat_layers:
            out = gat(x, edge_index)
            x = out if not isinstance(out, tuple) else out[0]
            x = F.relu(x)
            x = self.dropout(x)

        attention_weights = torch.softmax(self.attention_head(x), dim=1)

        anomaly_scores = self.anomaly_head(x)

        return anomaly_scores, attention_weights

class TrajectoryFeatureExtractor(nn.Module):

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 6):

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:

        num_nodes, seq_len, _ = positions.shape
        flat_pos = positions.view(-1, 2)

        features = self.mlp(flat_pos)
        features = features.view(num_nodes, seq_len, -1)

        features = features.mean(dim=1)

        return features

def build_interaction_graph(positions: torch.Tensor, interaction_dist: float = 50.0) -> torch.Tensor:

    num_nodes = positions.shape[0]
    edge_list = []

    dists = torch.cdist(positions, positions)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if dists[i, j] < interaction_dist:
                edge_list.append([i, j])
                edge_list.append([j, i])

    if len(edge_list) == 0:

        edge_list = [[i, i] for i in range(num_nodes)]

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

if __name__ == "__main__":

    import torch

    num_nodes = 10
    input_dim = 6

    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 30))

    model = SpatioTemporalGCN(input_dim=input_dim, hidden_dim=128, num_layers=3)

    anomaly_scores, attention = model(x, edge_index)
    print(f"Anomaly scores: {anomaly_scores.shape}")
    print(f"Attention weights: {attention.shape}")
