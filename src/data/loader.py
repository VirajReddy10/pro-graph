"""
Loader for the Yeast protein-protein interaction (PPI) network.

Source: Zhang & Chen, "Link Prediction Based on Graph Neural Networks"
(NeurIPS 2018), benchmark dataset originally from the DIP database.
2,375 proteins (nodes), 11,693 known interactions (edges).
"""

import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import Data

YEAST_MAT_PATH = "data/raw/yeast.mat"


def load_yeast_graph() -> Data:
    """
    Load the Yeast PPI network as a PyTorch Geometric Data object.

    Returns a Data object with:
        - num_nodes: 2375
        - edge_index: [2, num_edges] tensor of node index pairs (undirected,
          so each edge appears twice: (i,j) and (j,i))
        - x: placeholder node features (we'll replace these later;
          GCN/GAT need *some* initial feature, even if just identity-like)
    """
    mat = sio.loadmat(YEAST_MAT_PATH)
    adjacency = mat["net"].tocoo()

    # edge_index: PyG expects shape [2, num_edges], dtype long
    edge_index = torch.tensor(
        np.vstack([adjacency.row, adjacency.col]),
        dtype=torch.long,
    )

    num_nodes = adjacency.shape[0]

    # Placeholder node features: simple node degree as a 1-dim feature.
    # This gives the model *something* to work with structurally, without
    # external biological data. We'll discuss richer features later.
    degree = torch.zeros(num_nodes, dtype=torch.float)
    for node in adjacency.row:
        degree[node] += 1
    x = degree.unsqueeze(1)  # shape [num_nodes, 1]

    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    return data