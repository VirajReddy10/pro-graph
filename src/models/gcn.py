"""
GCN-based encoder for link prediction on the Yeast PPI network.

Architecture: two GCN layers produce a learned embedding per protein.
Link prediction score between two proteins is the dot product of their
embeddings (standard Graph Autoencoder decoder).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(torch.nn.Module):
    """Two-layer GCN that maps node features -> learned node embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def decode(z, edge_label_index):
    """
    Score each candidate edge by the dot product of its two node embeddings.

    Args:
        z: [num_nodes, embedding_dim] learned node embeddings
        edge_label_index: [2, num_candidate_edges] pairs of node indices to score

    Returns:
        [num_candidate_edges] raw scores (logits, not yet passed through sigmoid)
    """
    src, dst = edge_label_index
    return (z[src] * z[dst]).sum(dim=-1)