"""
Train a GCN for link prediction on the Yeast PPI network.

Usage:
    python scripts/train_gcn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling

from data.loader import load_yeast_graph, split_graph_for_link_prediction
from models.gcn import GCNEncoder, decode

HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
SEED = 42


def train_one_epoch(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()

    z = model(train_data.x, train_data.edge_index)

    # Positive edges: the held-out supervision edges from the split
    pos_edge_index = train_data.edge_label_index

    # Negative edges: resample fresh each epoch (standard practice),
    # same count as positives, avoiding any edge that's actually in
    # the full message-passing graph (so we don't accidentally sample
    # a "negative" that's actually a real, known interaction)
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
    """Compute AUC-ROC on a val/test split (which already has fixed pos+neg edges)."""
    model.eval()
    z = model(data.x, data.edge_index)
    scores = decode(z, data.edge_label_index)
    probs = torch.sigmoid(scores)
    auc = roc_auc_score(data.edge_label.cpu().numpy(), probs.cpu().numpy())
    return auc


def main():
    torch.manual_seed(SEED)

    print("Loading Yeast PPI graph...")
    data = load_yeast_graph()
    train_data, val_data, test_data = split_graph_for_link_prediction(data, seed=SEED)

    print(f"  Train supervision edges: {train_data.edge_label_index.size(1)}")
    print(f"  Val edges (pos+neg): {val_data.edge_label_index.size(1)}")
    print(f"  Test edges (pos+neg): {test_data.edge_label_index.size(1)}")

    model = GCNEncoder(in_channels=1, hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print()
    print("Training...")
    best_val_auc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_one_epoch(model, optimizer, train_data)

        if epoch % 10 == 0 or epoch == 1:
            val_auc = evaluate(model, val_data)
            print(f"  Epoch {epoch:3d} | loss: {loss:.4f} | val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print()
    print(f"Best val AUC: {best_val_auc:.4f}")

    model.load_state_dict(best_state)
    test_auc = evaluate(model, test_data)
    print(f"Test AUC (best model): {test_auc:.4f}")

    torch.save(model.state_dict(), "models/gcn_yeast.pt")
    print("Model saved to models/gcn_yeast.pt")


if __name__ == "__main__":
    main()