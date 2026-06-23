"""
GAT-based encoder for link prediction on the Yeast PPI network.

Same overall structure as the GCN encoder (two-layer, dot-product
decoder), but uses Graph Attention layers instead of Graph Convolution
layers, letting the model learn to weight neighbors differently rather
than treating them uniformly.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(torch.nn.Module):
    """Two-layer GAT that maps node features -> learned node embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        # First layer: multiple attention "heads" run in parallel, each
        # learning a different weighting scheme, then concatenated.
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        # Second layer: single head, no concatenation, producing the
        # final embedding dimension directly.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x