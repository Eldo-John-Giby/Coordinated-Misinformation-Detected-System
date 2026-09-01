# AI/ML Tech Stack — Coordinated Misinformation Network Detection System

---

## 🧠 Models Used

### 1. Graph Neural Networks (GNNs)

#### a) Graph Convolutional Network (GCN)
- **File:** `src/models/gcn.py`
- **Framework:** PyTorch Geometric (`torch_geometric`)
- **Layer type:** `GCNConv`
- **Architecture:**
  - Input → GCNConv → BatchNorm → ReLU → Dropout
  - Repeat for `num_layers` (default: 2)
  - Final GCNConv output layer (no activation)
- **Default config:** hidden_channels=64, dropout=0.5, lr=0.001, epochs=100, patience=10
- **Purpose:** Node-level classification (IO vs Control account detection) on the coordination graph
- **Techniques:** Early stopping, class-weighted cross-entropy loss, Adam optimizer with weight decay

#### b) GraphSAGE
- **File:** `src/models/graphsage.py`
- **Framework:** PyTorch Geometric (`torch_geometric`)
- **Layer type:** `SAGEConv`
- **Architecture:**
  - Input → SAGEConv → BatchNorm → ReLU → Dropout
  - Repeat for `num_layers` (default: 2)
  - Final SAGEConv output layer (no activation)
- **Default config:** hidden_channels=64, dropout=0.5, lr=0.001, epochs=100, aggregator="mean"
- **Purpose:** Node-level classification; uses neighbor sampling/aggregation for scalability
- **Techniques:** Same training regime as GCN (early stopping, class-weighted loss)

---

### 2. Traditional / Baseline ML Models

#### a) Random Forest Classifier
- **File:** `src/models/baseline.py`
- **Library:** `scikit-learn.ensemble.RandomForestClassifier`
- **Config:** n_estimators=100, max_depth=20, class_weight="balanced"
- **Purpose:** Account-level classification baseline for comparison with GNN approaches

#### b) Logistic Regression
- **File:** `src/models/baseline.py`
- **Library:** `scikit-learn.linear_model.LogisticRegression`
- **Config:** max_iter=1000, class_weight="balanced"
- **Purpose:** Linear baseline; features are StandardScaler-normalized before training

#### c) Threshold-Based Baseline
- **File:** `src/models/baseline.py`
- **Method:** Simple score thresholding with auto-optimization (sweeping thresholds 0.1–0.95)
- **Purpose:** Naive baseline using coordination scores directly

---

### 3. NLP / Text Embedding Model

#### Sentence Transformers (all-MiniLM-L6-v2)
- **File:** `src/nlp/embeddings.py`
- **Library:** `sentence-transformers`
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding dimension:** 384
- **Batch size:** 256
- **Device:** Auto-detects CUDA/CPU
- **Purpose:** Produces L2-normalized 384-dim embeddings for each post's text content; used for semantic similarity computation between posts and accounts

---

## 🔍 Similarity Search

| Method | Library | Description |
|--------|---------|-------------|
| **FAISS** (primary) | `faiss-cpu` | Fast approximate nearest-neighbor search using inner product (cosine sim on normalized vectors). Index types: Flat, IVF |
| **sklearn NearestNeighbors** (fallback) | `scikit-learn` | Brute-force cosine NN search; used if FAISS is unavailable |

- **File:** `src/nlp/similarity.py`
- **Default k:** 20 neighbors
- **Threshold:** 0.7 cosine similarity

---

## 📊 Feature Engineering Pipeline

### Node (Account) Features
| Feature | Description |
|---------|-------------|
| `post_count` | Number of posts by account |
| `avg_follower_count` | Average follower count |
| `avg_following_count` | Average following count |
| `repost_ratio` | Ratio of reposts to total posts |
| `avg_hashtag_count` | Average hashtags per post |
| `avg_url_count` | Average URLs per post |
| `avg_mention_count` | Average mentions per post |

### Edge (Coordination) Features
| Feature | Weight | Description |
|---------|--------|-------------|
| `semantic_similarity` | 0.35 | Cosine similarity of account embeddings (max-mean of top-5 pairwise) |
| `temporal_score` | 0.25 | Exponential decay of mean minimum time gap between posts (τ=3600s) |
| `shared_url_score` | 0.15 | Jaccard-like URL overlap between accounts |
| `shared_hashtag_score` | 0.10 | Common hashtags / total unique hashtags |
| `shared_mention_score` | 0.05 | Common mentions / total unique mentions |
| `repost_score` | 0.10 | Direct repost + shared repost target similarity |

**Combined coordination score** = weighted sum of the above 6 features.

### Temporal Features
- **Exponential time decay scoring** (τ = 1 hour)
- **Burst detection** (window = 5 min, min 3 posts)
- **Synchronization score** (fraction of posts within burst window)

### Semantic Features
- **Max-mean:** Average of top-5 pairwise post similarities
- **Centroid similarity:** Cosine similarity between account centroid embeddings
- **Max pairwise:** Maximum pairwise similarity between any two posts

---

## 🕸️ Graph Construction

- **File:** `src/graph/builder.py`
- **Libraries:** `networkx`, `torch_geometric`
- **Graph type:** Undirected weighted graph
- **Nodes:** Accounts (features = node feature vector)
- **Edges:** Accounts with coordination_score ≥ threshold (default 0.3)
- **Max edges per node:** 50 (scalability cap)
- **PyG Data object** built for GNN training with node features and labels

---

## 🧩 Community Detection

| Method | Library | Description |
|--------|---------|-------------|
| **Louvain** (primary) | `python-louvain` | Resolution-based modularity optimization |
| **Greedy Modularity** (fallback) | `networkx` | If python-louvain not installed |
| **Label Propagation** | `networkx` | Semi-synchronous label propagation |

- **File:** `src/graph/community.py`
- **Scoring:** Suspiciousness = 0.3×density + 0.3×avg_coordination + 0.2×size_factor + 0.2×io_ratio

---

## 📈 Evaluation & Metrics

| Metric | Usage |
|--------|-------|
| Accuracy | All models |
| Precision (weighted) | All models |
| Recall (weighted) | All models |
| F1 Score (weighted) | All models + early stopping |
| ROC-AUC | GNN and baseline models |
| PR-AUC (Average Precision) | GNN and baseline models |
| Adjusted Rand Index | Community detection |
| Normalized Mutual Information | Community detection |

- **File:** `src/evaluation/metrics.py`
- **Experiments:** Ablation study (per-feature-category removal) + model comparison

---

## 📦 Full Python Dependency Stack

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework (GCN, GraphSAGE) |
| `torch-geometric` | ≥2.3.0 | Graph neural network layers & data utilities |
| `scikit-learn` | ≥1.1.0 | Random Forest, Logistic Regression, preprocessing, metrics |
| `sentence-transformers` | ≥2.2.0 | Text embedding model (MiniLM-L6-v2) |
| `faiss-cpu` | ≥1.7.0 | Fast nearest-neighbor similarity search |
| `networkx` | ≥2.8.0 | Graph construction, community detection, statistics |
| `numpy` | ≥1.23.0 | Numerical operations |
| `pandas` | ≥1.5.0 | Data manipulation |
| `matplotlib` | ≥3.6.0 | Static visualization |
| `seaborn` | ≥0.12.0 | Statistical visualization |
| `plotly` | ≥5.10.0 | Interactive visualization |
| `streamlit` | ≥1.20.0 | Web dashboard/app |
| `PyYAML` | ≥6.0 | Configuration parsing |
| `tqdm` | ≥4.64.0 | Progress bars |

---

## 🔄 Pipeline Architecture (End-to-End)

```
┌─────────────────────────────────────────────────────────┐
│                   DATA LOADING                          │
│  Zenodo IO dataset (Honduras, UAE, etc.) → DataFrame   │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  PREPROCESSING                          │
│  Text normalization, URL/mention extraction,            │
│  timestamp parsing, missing value handling               │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│              NLP EMBEDDING (MiniLM-L6-v2)              │
│  384-dim Sentence Transformer embeddings per post       │
│  → Aggregated to account-level embeddings               │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│          SIMILARITY SEARCH (FAISS / sklearn)            │
│  K-NN search for semantically similar account pairs     │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING                        │
│  Semantic + Temporal + URL/Hashtag/Mention/Repost       │
│  → Weighted coordination score per edge                 │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│              GRAPH CONSTRUCTION                         │
│  NetworkX graph → PyG Data for GNNs                     │
│  Nodes=accounts, Edges=coordination links               │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│              COMMUNITY DETECTION                        │
│  Louvain / Label Propagation → Suspicious group scoring │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│              MODEL TRAINING & EVALUATION                │
│  Baselines: Threshold, Random Forest, Logistic Reg      │
│  GNNs: GCN, GraphSAGE                                   │
│  + Ablation study + Model comparison                    │
└────────────────────┬────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│         EXPLAINABILITY & VISUALIZATION                  │
│  Feature importance, group explanations, network plots  │
│  Streamlit dashboard                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Key File Reference

| Module | Key Files |
|--------|-----------|
| **Data** | `src/data/loader.py`, `src/data/preprocessing.py`, `src/data/convert_zenodo.py` |
| **NLP** | `src/nlp/embeddings.py`, `src/nlp/similarity.py` |
| **Features** | `src/features/semantic.py`, `src/features/temporal.py`, `src/features/edge_features.py`, `src/features/network.py` |
| **Graph** | `src/graph/builder.py`, `src/graph/community.py`, `src/graph/visualization.py` |
| **Models** | `src/models/baseline.py`, `src/models/gcn.py`, `src/models/graphsage.py` |
| **Evaluation** | `src/evaluation/metrics.py`, `src/evaluation/experiments.py` |
| **Pipeline** | `src/pipeline.py` (orchestrator) |
| **Config** | `configs/config.yaml` |
| **App** | `app/streamlit_app.py` |
