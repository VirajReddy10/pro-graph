"""
Run GCN and GAT training across multiple seeds to get a statistically
meaningful comparison (mean +/- std test AUC), rather than relying on
a single run, given observed run-to-run variance from non-deterministic
negative sampling and dropout.

Usage:
    python scripts/run_multi_seed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling

from data.loader import load_yeast_graph_structural, split_graph_for_link_prediction
from models.gcn import GCNEncoder, decode
from models.gat import GATEncoder

HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32
HEADS = 4
LEARNING_RATE = 0.01
SEEDS = [42, 43, 44, 45, 46]


def train_one_epoch(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    z = model(train_data.x, train_data.edge_index)
    pos_edge_index = train_data.edge_label_index
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=pos_edge_index.size(1),
    )
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_edge_index.size(1)),
        torch.zeros(neg_edge_index.size(1)),
    ])
    scores = decode(z, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(scores, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    z = model(data.x, data.edge_index)
    scores = decode(z, data.edge_label_index)
    probs = torch.sigmoid(scores)
    return roc_auc_score(data.edge_label.cpu().numpy(), probs.cpu().numpy())


def run_one(model_name: str, seed: int, num_epochs: int) -> float:
    torch.manual_seed(seed)
    data = load_yeast_graph_structural()
    train_data, val_data, test_data = split_graph_for_link_prediction(data, seed=seed)

    if model_name == "GCN":
        model = GCNEncoder(in_channels=4, hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS)
    else:
        model = GATEncoder(in_channels=4, hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS, heads=HEADS)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_auc = 0.0
    best_state = None
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, optimizer, train_data)
        if epoch % 10 == 0:
            val_auc = evaluate(model, val_data)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return evaluate(model, test_data)


def main():
    results = {"GCN": [], "GAT": []}

    for seed in SEEDS:
        gcn_auc = run_one("GCN", seed, num_epochs=100)
        gat_auc = run_one("GAT", seed, num_epochs=300)
        results["GCN"].append(gcn_auc)
        results["GAT"].append(gat_auc)
        print(f"Seed {seed}: GCN={gcn_auc:.4f}  GAT={gat_auc:.4f}")

    print()
    for model_name, aucs in results.items():
        aucs = np.array(aucs)
        print(f"{model_name}: mean={aucs.mean():.4f}  std={aucs.std():.4f}  (runs: {[round(a, 4) for a in aucs]})")


if __name__ == "__main__":
    main()
