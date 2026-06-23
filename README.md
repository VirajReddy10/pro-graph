# Pro Graph

A Graph Neural Network for predicting protein-protein interactions (PPI), comparing GCN and GAT architectures with a statistically rigorous multi-seed evaluation.

## Status

✅ **Complete** — Dataset loaded and verified, GCN and GAT models trained, link prediction evaluated, multi-seed comparison performed.

## The Task

Given a network of known protein-protein interactions, can a model predict whether two proteins are likely to interact, even for pairs it has never seen? This is the **link prediction** task: hide some known interactions, train a model on the rest, then test whether it correctly distinguishes real hidden interactions from random, non-interacting protein pairs.

## Results

### Multi-Seed Comparison (5 seeds, test AUC)

| Model | Mean Test AUC | Std Dev |
|---|---|---|
| GCN | 0.9132 | 0.0083 |
| GAT | 0.9131 | 0.0031 |

**Finding**: GCN and GAT perform statistically equivalently on this graph (difference of 0.0001, well within noise). GAT shows ~2.7x lower run-to-run variance, suggesting more consistent behavior even without an accuracy advantage.

See [`notebooks/01_results_comparison.ipynb`](notebooks/01_results_comparison.ipynb) for the full analysis, including graph visualization and the feature-engineering investigation.

### A Methodological Lesson (worth being upfront about)

Early single-run comparisons gave **contradictory** results — one run suggested GAT was ahead, a later run suggested GCN was ahead. This is because both negative edge sampling and dropout are non-deterministic between runs. Only after training each model across 5 different seeds and comparing mean ± standard deviation did a trustworthy picture emerge: the two models are essentially tied in accuracy, with GAT being modestly more consistent. This project intentionally documents that mistake and correction rather than hiding it, since recognizing and fixing premature conclusions from single stochastic runs is a real, valuable engineering skill.

## Dataset

**Yeast protein-protein interaction network** (Zhang & Chen, "Link Prediction Based on Graph Neural Networks," NeurIPS 2018), originally sourced from the DIP database (von Mering et al., 2002).

- 2,375 proteins (nodes)
- 11,693 known interactions (edges)
- Standard 85/5/10 train/val/test split (SEAL paper convention)

**Important limitation**: this benchmark distribution is fully anonymized — no protein identifiers (UniProt IDs, gene names) survive. This means biologically-informed embeddings (e.g., pretrained ProtBERT sequence embeddings) **cannot legitimately be attached to specific nodes** in this graph; there's no way to know which node is which real protein. This was investigated and confirmed before any feature engineering began, rather than fabricating a mapping.

## Node Features

Since real biological embeddings weren't usable on this anonymized graph, node features instead use **structural graph properties**, computed directly from the network topology via `networkx`:

- **Degree** — number of known interactions
- **Clustering coefficient** — how interconnected a protein's neighbors are with each other
- **PageRank** — overall structural "importance" in the network
- **Betweenness centrality** — how often a protein lies on shortest paths between other protein pairs (a "bridging" role)

All four features are z-score normalized. Adding these three features beyond simple degree improved both models substantially (GCN: 0.890 → 0.913 mean; GAT: 0.887 → 0.913 mean, both single-feature numbers from early single-run testing).

## Models

**Graph Convolutional Network (GCN)**: 2-layer `GCNConv`, uniform neighbor aggregation.

**Graph Attention Network (GAT)**: 2-layer `GATConv`, 4 attention heads in the first layer, learns to weight neighbors differently rather than averaging uniformly.

Both use a simple dot-product decoder: link prediction score = dot product of the two candidate proteins' learned embeddings.

## Quickstart

```bash
git clone https://github.com/VirajReddy10/pro-graph.git
cd pro-graph
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the dataset:

```bash
mkdir -p data/raw
curl -sL -o data/raw/yeast.mat "https://raw.githubusercontent.com/muhanzhang/SEAL/master/MATLAB/data/Yeast.mat"
```

Train individual models:

```bash
python scripts/train_gcn.py
python scripts/train_gat.py
```

Run the full multi-seed comparison:

```bash
python scripts/run_multi_seed.py
```

View results:

```bash
jupyter notebook notebooks/01_results_comparison.ipynb
```

## Project Structure
pro-graph/

├── data/

│   └── raw/

│       └── yeast.mat            # (gitignored; downloaded via curl)

├── models/                       # (gitignored; trained model weights)

├── notebooks/

│   └── 01_results_comparison.ipynb

├── src/

│   ├── data/

│   │   └── loader.py             # Graph loading, feature engineering, train/val/test split

│   └── models/

│       ├── gcn.py                # GCN encoder + dot-product decoder

│       └── gat.py                # GAT encoder

├── scripts/

│   ├── train_gcn.py

│   ├── train_gat.py

│   └── run_multi_seed.py         # Multi-seed statistical comparison

└── requirements.txt

## Key Engineering Decisions

**Feature normalization caught early**: raw protein degree (range 1-118, heavily skewed) caused exploding edge prediction scores (some over 3000) that would have saturated the sigmoid decoder and broken gradient flow. Caught via a sanity-check forward pass before any training was attempted, fixed via z-score normalization.

**Multi-seed evaluation over single-run claims**: rather than reporting whichever number came out of one training run, both models were evaluated across 5 seeds to get statistically meaningful mean ± std comparisons, after single-run results proved contradictory between attempts.

**Honest scope limitation on biological embeddings**: investigated whether pretrained ProtBERT embeddings could be used as richer node features, discovered the benchmark dataset is fully anonymized with no recoverable protein identity, and pivoted to structural graph features instead rather than fabricating a protein-ID mapping.

## Future Directions

- [ ] Switch to a protein-ID-linked dataset (e.g., STRING database) to enable real ProtBERT embeddings as node features
- [ ] Try GraphSAGE or other architectures for comparison
- [ ] Hyperparameter tuning (hidden dimensions, number of attention heads, learning rate)
- [ ] Visualize learned embeddings (t-SNE/UMAP) to see if biologically meaningful clusters emerge from purely structural training

## License

MIT
