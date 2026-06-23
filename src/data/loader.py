"""
Loader for the Yeast protein-protein interaction (PPI) network.

Source: Zhang & Chen, "Link Prediction Based on Graph Neural Networks"
(NeurIPS 2018), benchmark dataset originally from the DIP database.
2,375 proteins (nodes), 11,693 known interactions (edges).
"""

from pathlib import Path

import networkx as nx
import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

YEAST_MAT_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "yeast.mat")


def load_yeast_graph() -> Data:
    """
    Load the Yeast PPI network as a PyTorch Geometric Data object, using
    normalized protein degree as the only node feature.
    """
    mat = sio.loadmat(YEAST_MAT_PATH)
    adjacency = mat["net"].tocoo()

    edge_index = torch.tensor(
        np.vstack([adjacency.row, adjacency.col]),
        dtype=torch.long,
    )
    num_nodes = adjacency.shape[0]

    degree = torch.zeros(num_nodes, dtype=torch.float)
    for node in adjacency.row:
        degree[node] += 1

    degree_normalized = (degree - degree.mean()) / degree.std()
    x = degree_normalized.unsqueeze(1)

    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    return data


def compute_structural_features(adjacency) -> torch.Tensor:
    """
    Compute richer structural node features beyond simple degree:
    clustering coefficient, PageRank, and betweenness centrality.
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


def load_yeast_graph_structural() -> Data:
    """
    Load the Yeast PPI network with richer structural features
    (degree, clustering coefficient, PageRank, betweenness centrality).
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


def split_graph_for_link_prediction(data: Data, seed: int = 42):
    """
    Split the graph into train/val/test sets for link prediction.
    Follows the standard SEAL-paper convention: 85% train, 5% val, 10% test.
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

