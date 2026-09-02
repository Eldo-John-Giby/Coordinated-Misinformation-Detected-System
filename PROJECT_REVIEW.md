# Coordinated Misinformation Network Detection System
## AI/ML Class Project Review

---

## 1. Project Overview

**Problem:** Detect coordinated networks of social media accounts engaged in state-sponsored Information Operations (IOs). These accounts mimic legitimate behavior and manipulate public discourse in coordinated fashion.

**Approach:** Build an end-to-end pipeline that uses NLP embeddings, temporal analysis, content-sharing patterns, and graph neural networks to classify accounts as IO (malicious) or Control (legitimate) and identify coordinated groups.

**Dataset:** "Labeled Datasets for Research on Information Operations" from Zenodo (https://zenodo.org/records/14141550), containing 26 campaigns from 16 state actors across 13M+ posts from 303K accounts.

---

## 2. Dataset Columns (19 columns per record)

| Column | Type | Description |
|--------|------|-------------|
| `postid` | string | Unique post identifier |
| `post_text` | string | Text content (PII hashed) |
| `application_name` | string | Hashed application name |
| `post_language` | string | Language of the post |
| `in_reply_to_postid` | string | ID of post being replied to |
| `in_reply_to_accountid` | string | Account being replied to |
| `post_time` | datetime | Timestamp of the post |
| `accountid` | string | Anonymized account identifier |
| `account_profile_description` | string | Profile bio (PII hashed) |
| `follower_count` | int | Number of followers |
| `following_count` | int | Number of accounts followed |
| `account_creation_date` | date | Account creation date |
| `is_repost` | bool | Whether this is a repost |
| `reposted_accountid` | string | Original account of repost |
| `reposted_postid` | string | Original post of repost |
| `hashtags` | string | Comma-separated hashtags |
| `urls` | string | Comma-separated hashed URLs |
| `account_mentions` | string | Comma-separated mentioned accounts |
| `is_control` | bool | **True** = control (organic), **False** = IO account |

**Ground Truth Label:** `is_control` — aggregated to account level. An account is IO if ANY of its posts are labeled IO.

---

## 3. Dataset Used

- **Campaign:** Honduras (default)
- **Total Posts Loaded:** 1,262,830
- **Unique Accounts:** 224,685
- **Graph Statistics:**
  - 224,684 nodes (accounts)
  - 987,981 edges (coordination links)
  - Graph density: 3.9 × 10⁻⁵
  - Average degree: 8.79
  - Median degree: 2.0
  - Connected components: 104,836
  - Average clustering coefficient: 0.312

---

## 4. Full Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: DATA LOADING                                      │
│  Zenodo IO dataset (Honduras) → Pandas DataFrame            │
│  Files: src/data/loader.py                                  │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: PREPROCESSING                                     │
│  - Remove malformed records (null postid/accountid)         │
│  - Parse timestamps, handle missing values                  │
│  - Normalize text (lowercase, remove URLs, strip whitespace)│
│  - Extract URLs, hashtags, mentions into lists              │
│  - Identify repost relationships                            │
│  - Build account-to-node-ID mappings                        │
│  - Compute per-account ground truth labels                  │
│  Files: src/data/preprocessing.py                           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: NLP EMBEDDING COMPUTATION                         │
│  Model: sentence-transformers/all-MiniLM-L6-v2             │
│  - 384-dimensional embeddings per post                      │
│  - L2-normalized for cosine similarity                      │
│  - Batch size: 256, device: auto (CUDA/CPU)                 │
│  - Aggregated to account-level by averaging                 │
│  Files: src/nlp/embeddings.py                               │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: SIMILARITY SEARCH                                 │
│  Method: FAISS (primary) or sklearn NearestNeighbors        │
│  - K-NN search: k=20 neighbors                              │
│  - Cosine similarity threshold: 0.7                         │
│  - Finds semantically similar account pairs                 │
│  Files: src/nlp/similarity.py                               │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: EDGE FEATURE COMPUTATION                          │
│  For each account pair, compute:                            │
│  - Semantic similarity (cosine of embeddings)               │
│  - Temporal coordination (exponential time decay)           │
│  - Shared URL score (Jaccard-like)                          │
│  - Shared hashtag score (common/unique)                     │
│  - Shared mention score                                     │
│  - Repost score (direct + shared targets)                   │
│  - Combined coordination score (weighted sum)               │
│  Files: src/features/edge_features.py                       │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6: GRAPH CONSTRUCTION                                │
│  - Nodes = social media accounts                            │
│  - Edges = coordination links (score ≥ 0.3 threshold)       │
│  - Node features: 7 behavioral features per account         │
│  - Edge features: 7 coordination evidence features          │
│  - Max edges per node: 50 (scalability cap)                 │
│  - Convert to PyG Data object for GNN training              │
│  Files: src/graph/builder.py                                │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 7: COMMUNITY DETECTION                               │
│  Method: Louvain (primary) / Greedy Modularity (fallback)   │
│  - Scores communities by suspiciousness                     │
│  - Generates interpretable evidence per group               │
│  Files: src/graph/community.py                              │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 8-9: MODEL TRAINING                                  │
│  Baselines: Threshold, Random Forest, Logistic Regression   │
│  GNNs: GCN, GraphSAGE                                       │
│  Files: src/models/baseline.py, gcn.py, graphsage.py       │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 10-13: EVALUATION, ABLATION, EXPLAINABILITY, VIZ     │
│  - Model comparison across all metrics                      │
│  - Feature ablation study                                   │
│  - Feature importance analysis                              │
│  - Group explanations with evidence                         │
│  - Visualization: network graphs, ROC/PR curves, etc.       │
│  Files: src/evaluation/experiments.py, metrics.py           │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Models Used

### 5.1 Graph Neural Networks (GNNs)

#### GCN (Graph Convolutional Network)
- **File:** `src/models/gcn.py`
- **Framework:** PyTorch Geometric
- **Layer:** `GCNConv`
- **Architecture:**
  ```
  Input (7 features) → GCNConv(7, 64) → BatchNorm → ReLU → Dropout(0.5)
                     → GCNConv(64, 64) → BatchNorm → ReLU → Dropout(0.5)
                     → GCNConv(64, 2) → Output logits
  ```
- **Config:** hidden_channels=64, num_layers=2, dropout=0.5
- **Purpose:** Node classification — classify each account as IO (1) or Control (0)

#### GraphSAGE
- **File:** `src/models/graphsage.py`
- **Framework:** PyTorch Geometric
- **Layer:** `SAGEConv` (mean aggregation)
- **Architecture:** Same structure as GCN but uses neighborhood sampling/aggregation
- **Config:** hidden_channels=64, num_layers=2, dropout=0.5, aggregator="mean"
- **Purpose:** Scalable node classification; learns from local neighborhoods

### 5.2 Baseline ML Models

#### Random Forest
- **File:** `src/models/baseline.py`
- **Library:** scikit-learn
- **Config:** n_estimators=100, max_depth=20, class_weight="balanced"
- **Features:** 7 account-level behavioral features (post_count, avg_follower_count, etc.)

#### Logistic Regression
- **File:** `src/models/baseline.py`
- **Library:** scikit-learn
- **Config:** max_iter=1000, class_weight="balanced"
- **Preprocessing:** StandardScaler applied to features

#### Threshold Baseline
- Simple score thresholding with auto-optimization (sweeping 0.1–0.95)

### 5.3 NLP Embedding Model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Embedding dimension:** 384
- **Purpose:** Produces L2-normalized embeddings for each post's text; used for semantic similarity computation

---

## 6. Feature Engineering

### 6.1 Node (Account) Features — 7 dimensions used as GNN input

| Feature | Description |
|---------|-------------|
| `post_count` | Number of posts by account |
| `avg_follower_count` | Average follower count |
| `avg_following_count` | Average following count |
| `repost_ratio` | Ratio of reposts to total posts |
| `avg_hashtag_count` | Average hashtags per post |
| `avg_url_count` | Average URLs per post |
| `avg_mention_count` | Average mentions per post |

Additional derived features computed but not all used as GNN input: `posts_per_day`, `follower_following_ratio`, `reply_ratio`, `unique_hashtag_ratio`, `unique_url_ratio`.

### 6.2 Edge (Coordination) Features — Weighted combination

| Feature | Weight | Description |
|---------|--------|-------------|
| `semantic_similarity` | 0.35 | Cosine similarity of account embeddings (max-mean of top-5 pairwise) |
| `temporal_score` | 0.25 | Exponential decay of mean minimum time gap (τ=3600s) |
| `shared_url_score` | 0.15 | Jaccard-like URL overlap: \|intersection\| / max(\|A\|, \|B\|) |
| `shared_hashtag_score` | 0.10 | Common hashtags / total unique hashtags |
| `shared_mention_score` | 0.05 | Common mentions / total unique mentions |
| `repost_score` | 0.10 | Direct repost + shared repost target similarity |

**Combined coordination score** = Σ(wᵢ × featureᵢ)

Edge threshold: 0.3 — only pairs with coordination_score ≥ 0.3 get an edge in the graph.

### 6.3 Temporal Features

- **Exponential time decay:** `temporal_score = exp(-mean_min_delta / τ)` where τ = 3600 seconds
- **Burst detection:** Window = 5 minutes, minimum 3 posts to count as burst
- **Synchronization score:** Fraction of posts within burst window

### 6.4 Semantic Features

- **Max-mean:** Average of top-5 pairwise post similarities between accounts
- **Centroid similarity:** Cosine similarity between account centroid embeddings
- **Max pairwise:** Maximum pairwise similarity between any two posts

---

## 7. Training / Testing Split

### Strategy: Account-Level Split (70 / 15 / 15)

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model training |
| Validation | 15% | Early stopping, hyperparameter tuning |
| Test | 15% | Final evaluation |

**Key details:**
- Split is at the **account level**, NOT the post level
- All posts from one account stay in the same split
- Prevents information leakage where model learns account-specific patterns
- Stratified split maintains class balance across splits
- GNN models use random node masks on the PyG Data object (70/15/15)

**Why account-level split?**
> Random post-level splitting would cause posts from the same IO account to appear in both train and test, allowing the model to memorize account-specific patterns rather than learning generalizable coordination signals.

---

## 8. Training Details

### GNN Training (GCN & GraphSAGE)
- **Optimizer:** Adam (lr=0.001, weight_decay=5e-4)
- **Loss:** CrossEntropyLoss with class weights (to handle imbalance)
- **Class weights:** Inverse frequency normalization
- **Early stopping:** Patience = 10 epochs, monitored on validation loss
- **Max epochs:** 100
- **Batch training:** Full-batch (entire graph forward pass per epoch)
- **Best model:** Restored from checkpoint with lowest validation loss

### Baseline Training
- **Random Forest:** 100 trees, max_depth=20, balanced class weights
- **Logistic Regression:** StandardScaler normalization, max_iter=1000, balanced weights
- **Train/test:** 80/20 split within the account-level train set

---

## 9. Evaluation Metrics

### Classification Metrics
| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision (weighted) | Weighted by class frequency |
| Recall (weighted) | Weighted by class frequency |
| F1 Score (weighted) | Harmonic mean of precision/recall |
| ROC-AUC | Area under ROC curve |
| PR-AUC | Area under Precision-Recall curve |

### Community Detection Metrics
| Metric | Description |
|--------|-------------|
| Adjusted Rand Index (ARI) | Agreement between predicted and true labels |
| Normalized Mutual Information (NMI) | Information-theoretic similarity |

---

## 10. Results

### 10.1 Model Comparison (Honduras Campaign)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Threshold Baseline | 0.9917 | 0.0000 | 0.0000 | 0.0000 | 0.9307 | 0.1546 |
| **Random Forest** | **0.9992** | **0.9517** | **0.9517** | **0.9517** | **0.9973** | **0.9890** |
| Logistic Regression | 0.9720 | 0.2283 | 0.9946 | 0.3714 | 0.9988 | 0.9149 |
| GCN | 0.9463 | 0.9929 | 0.9463 | 0.9662 | 0.9955 | 0.5889 |
| GraphSAGE | 0.9176 | 0.9923 | 0.9176 | 0.9502 | 0.9873 | 0.6800 |

### 10.2 Key Observations
- **Random Forest** achieves the best overall performance (F1=0.9517, ROC-AUC=0.9973)
- **GCN** achieves the highest precision (0.9929) but lower recall (0.9463)
- **GraphSAGE** has highest precision among GNNs (0.9923) but lowest recall (0.9176)
- **Logistic Regression** has high recall (0.9946) but very low precision (0.2283), indicating many false positives
- **Threshold baseline** fails at classification (F1=0) but has reasonable ROC-AUC (0.9307)

### 10.3 Ablation Study

| Ablation | Accuracy | F1 |
|----------|----------|----|
| Full model | 0.9992 | 0.9992 |
| No semantic | 0.9992 | 0.9992 |
| No temporal | 0.9992 | 0.9992 |
| No URL | 0.9992 | 0.9992 |
| No hashtag | 0.9992 | 0.9992 |
| No mention | 0.9992 | 0.9992 |
| No repost | 0.9992 | 0.9992 |

> **Note:** The ablation study shows identical performance across all conditions, likely because the baseline model relies primarily on account-level behavioral features (post_count, follower counts, etc.) rather than the coordination edge features. This is a known limitation — the ablation applies to edge features but the Random Forest uses node-level features directly.

---

## 11. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data | pandas, numpy |
| ML | scikit-learn |
| NLP | sentence-transformers (all-MiniLM-L6-v2) |
| Similarity Search | FAISS (primary), sklearn NearestNeighbors (fallback) |
| Graph Construction | NetworkX |
| GNN | PyTorch ≥ 2.0, PyTorch Geometric ≥ 2.3 |
| Community Detection | python-louvain |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Configuration | PyYAML |

---

## 12. Project File Structure

```
project/
├── data/
│   ├── raw/                         # Downloaded CSV files
│   └── processed/                   # Processed data
├── src/
│   ├── data/
│   │   ├── loader.py               # Dataset loading and discovery
│   │   ├── preprocessing.py        # Data cleaning and normalization
│   │   └── convert_zenodo.py       # Zenodo JSONL → CSV converter
│   ├── nlp/
│   │   ├── embeddings.py           # Sentence Transformer embeddings
│   │   └── similarity.py           # FAISS/sklearn similarity search
│   ├── features/
│   │   ├── semantic.py             # Semantic coordination features
│   │   ├── temporal.py             # Temporal coordination features
│   │   ├── network.py              # Account-level network features
│   │   └── edge_features.py        # Pairwise edge features
│   ├── graph/
│   │   ├── builder.py              # NetworkX + PyG graph construction
│   │   ├── community.py            # Louvain community detection
│   │   └── visualization.py        # Plotting and visualization
│   ├── models/
│   │   ├── baseline.py             # RF, LR, threshold baselines
│   │   ├── gcn.py                  # Graph Convolutional Network
│   │   └── graphsage.py            # GraphSAGE
│   ├── evaluation/
│   │   ├── metrics.py              # Classification + community metrics
│   │   └── experiments.py          # Experiment runner + ablation
│   └── pipeline.py                 # End-to-end orchestration
├── app/
│   └── streamlit_app.py            # Interactive dashboard
├── configs/
│   └── config.yaml                 # All hyperparameters and settings
├── results/
│   ├── pipeline_results.json       # Full results output
│   ├── model_comparison.csv        # Model comparison table
│   └── ablation_results.csv        # Ablation study results
├── requirements.txt
├── train.py                        # Main entry point
└── README.md
```

---

## 13. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run with sample data (no dataset download needed)
python train.py --sample

# Run with real dataset
python train.py --config configs/config.yaml

# Run specific campaign
python train.py --campaign Honduras

# Launch interactive dashboard
streamlit run app/streamlit_app.py
```

---

## 14. Limitations

1. **Dataset restriction:** Requires academic access from Zenodo
2. **Anonymization:** URLs, mentions, and account IDs are hashed, limiting analysis depth
3. **Platform specificity:** Single platform (Twitter/X)
4. **Class imbalance:** IO accounts are outnumbered by control accounts
5. **Generalizability:** Models trained on known campaigns may not generalize to novel IO tactics
6. **Ablation limitation:** Feature ablation applies to edge features but baselines use node features directly, limiting ablation insights
