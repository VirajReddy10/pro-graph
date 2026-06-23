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

    # Normalize: raw degree counts (1 to 118, heavily right-skewed) cause
    # exploding magnitudes through GCN's message passing, which saturates
    # the sigmoid decoder and breaks gradient flow during training.
    degree_normalized = (degree - degree.mean()) / degree.std()
    x = degree_normalized.unsqueeze(1)  # shape [num_nodes, 1]

    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    return data

def load_yeast_graph_structural() -> Data:
    """
    Load the Yeast PPI network with richer structural features
    (degree, clustering coefficient, PageRank, betweenness centrality)
    instead of just degree.
    """
    mat = sio.loadmat(YEAST_MAT_PATH)
    adjacency = mat["net"].tocoo()

    edge_index = torch.tensor(
        np.vstack([adjacency.row, adjacency.col]),
        dtype=torch.long,
    )
    num_nodes = adjacency.shape[0]

    x = compute_structural_features(adjacency)

    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    return data


import networkx as nx


def compute_structural_features(adjacency) -> torch.Tensor:
    """
    Compute richer structural node features beyond simple degree:
    clustering coefficient, PageRank, and betweenness centrality.

    Each feature is z-score normalized independently before being
    combined into a single [num_nodes, 4] feature matrix (degree +
    the three new features).
    """
    graph = nx.from_scipy_sparse_array(adjacency)

    num_nodes = adjacency.shape[0]

    degree = torch.tensor([d for _, d in graph.degree()], dtype=torch.float)
    clustering = torch.tensor([nx.clustering(graph, n) for n in range(num_nodes)], dtype=torch.float)
    pagerank_dict = nx.pagerank(graph)
    pagerank = torch.tensor([pagerank_dict[n] for n in range(num_nodes)], dtype=torch.float)
    betweenness_dict = nx.betweenness_centrality(graph)
    betweenness = torch.tensor([betweenness_dict[n] for n in range(num_nodes)], dtype=torch.float)

    def normalize(t):
        return (t - t.mean()) / t.std()

    features = torch.stack([
        normalize(degree),
        normalize(clustering),
        normalize(pagerank),
        normalize(betweenness),
    ], dim=1)

    return features


from torch_geometric.transforms import RandomLinkSplit


def split_graph_for_link_prediction(data: Data, seed: int = 42):
    """
    Split the graph into train/val/test sets for link prediction.

    Follows the standard SEAL-paper convention: 85% train, 5% val,
    10% test. Validation and test sets include both real ("positive")
    held-out edges and randomly sampled non-existent ("negative") edges,
    used to evaluate how well the model distinguishes real interactions
    from random protein pairs.
    """
    torch.manual_seed(seed)
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=False,
    )
    train_data, val_data, test_data = transform(data)
    return train_data, val_data, test_data