# Coordinated Misinformation Network Detection System
## Using Graph Learning and NLP

> **⚠️ Disclaimer:** This system detects **coordination patterns** between social media accounts. It does **NOT** automatically establish that an account is malicious, that content is false, or that an Information Operation (IO) is occurring. Coordination evidence is one signal among many that may warrant further investigation.

---

## 1. Problem Statement

State-sponsored Information Operations (IOs) involve coordinated groups of social media accounts that manipulate public discourse. Detecting these coordinated networks is critical for platform integrity, but challenging because:

- IO accounts mimic legitimate behavior
- Coordination can be subtle and distributed
- Existing datasets lack comprehensive control data
- Detection must generalize across campaigns and countries

## 2. Motivation

Previous work has focused on:
- **Post-level classification**: Identifying individual false/misleading posts
- **Account-level classification**: Detecting individual inauthentic accounts
- **Network analysis**: Studying interaction patterns of known IO campaigns

**What's missing:** A systematic approach that combines **semantic similarity**, **temporal coordination**, **shared content**, and **graph structure** to detect coordinated account networks—while being rigorously evaluated against baselines and ablated.

## 3. Research Gap

1. Most detection methods treat accounts independently, ignoring coordination signals
2. Limited systematic comparison between traditional ML and graph neural networks for this task
3. Lack of interpretable explanations for why groups are flagged
4. No comprehensive ablation study across coordination feature categories

## 4. Dataset

### Source
**"Labeled Datasets for Research on Information Operations"**
- Zenodo: https://zenodo.org/records/14141550
- Paper: Seçkin et al., "Labeled Datasets for Research on Information Operations" (ICWSM 2025)

### Dataset Structure
The dataset contains 19 columns per record:

| Column | Description |
|--------|-------------|
| `postid` | Unique post identifier |
| `post_text` | Text content (PII hashed) |
| `application_name` | Hashed application name |
| `post_language` | Language of the post |
| `in_reply_to_postid` | ID of post being replied to |
| `in_reply_to_accountid` | Account being replied to |
| `post_time` | Timestamp of the post |
| `accountid` | Anonymized account identifier |
| `account_profile_description` | Profile bio (PII hashed) |
| `follower_count` | Number of followers |
| `following_count` | Number of accounts followed |
| `account_creation_date` | Account creation date |
| `is_repost` | Boolean: is this a repost? |
| `reposted_accountid` | Original account of repost |
| `reposted_postid` | Original post of repost |
| `hashtags` | Comma-separated hashtags |
| `urls` | Comma-separated hashed URLs |
| `account_mentions` | Comma-separated mentioned accounts |
| `is_control` | **True** = control (organic), **False** = IO account |

### Dataset Statistics
- **26 campaigns** from **16 state actors**
- **13M+ posts** from **303k accounts**
- **703 files** (50k posts each)
- Countries: Armenia, Bangladesh, Catalonia, China, Cuba, Ecuador, Egypt, Ghana, Iran, Nigeria, Qatar, Russia, Spain, Ukraine, Venezuela

### Download Instructions

1. Visit https://zenodo.org/records/14141550
2. Request access (requires academic affiliation)
3. Download CSV files for the campaign(s) of interest
4. Place files in `data/raw/` directory
5. Structure:
   ```
   data/raw/
   ├── Armenia/
   │   ├── file1.csv
   │   ├── file2.csv
   │   └── ...
   ├── China_1/
   │   └── ...
   └── ...
   ```

### Quick Start (Sample Data)
For testing without the full dataset:
```bash
python train.py --sample
```
This creates a small synthetic dataset and runs the pipeline.

## 5. Feature Engineering

### 5.1 Semantic Features
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
- **Account similarity**: Average of top-5 pairwise post similarities
- **Centroid similarity**: Similarity between account centroid embeddings
- **Efficient search**: FAISS or sklearn NearestNeighbors

### 5.2 Temporal Features
- **Temporal proximity**: `temporal_score = exp(-Δt / τ)` where `τ = 3600s`
- **Synchronization ratio**: Fraction of posts within burst window (5 min)
- **Burst detection**: Identifies clusters of posts in short time windows

### 5.3 Shared Content Features
- **Shared URL score**: `|common URLs| / max(|URLs_A|, |URLs_B|)`
- **Shared hashtag score**: `|common hashtags| / |unique hashtags|`
- **Shared mention score**: `|common mentions| / |unique mentions|`
- **Repost score**: Direct repost + shared repost targets

### 5.4 Account-Level Features
- Post count, posting frequency
- Follower/following ratios
- Repost ratio
- Hashtag/URL/mention diversity
- Posting duration

## 6. Graph Construction

### Node Definition
**NODE = social media account**

### Edge Definition
**EDGE = evidence of coordination between two accounts**

Edges are created **only** when the coordination score exceeds a threshold. No edges are created for accounts that simply exist.

### Coordination Score
The weighted combination of coordination evidence:

```
coordination_score = w_sem × semantic_score
                   + w_temp × temporal_score
                   + w_url × url_score
                   + w_ht  × hashtag_score
                   + w_ment × mention_score
                   + w_rep × repost_score
```

**Default weights:**
| Feature | Weight |
|---------|--------|
| Semantic | 0.35 |
| Temporal | 0.25 |
| URL | 0.15 |
| Hashtag | 0.10 |
| Repost | 0.10 |
| Mention | 0.05 |

Weights are configurable and can be learned via logistic regression.

### Edge Features
Each edge carries:
- `coordination_score` (combined)
- `semantic_similarity`
- `temporal_score`
- `shared_url_score`
- `shared_hashtag_score`
- `shared_mention_score`
- `repost_score`

## 7. Baselines

| # | Baseline | Description |
|---|----------|-------------|
| 1 | **Threshold** | Classification based on coordination score threshold |
| 2 | **Louvain** | Community detection on coordination graph |
| 3 | **Random Forest** | ML on account-level behavioral features |
| 4 | **Logistic Regression** | Linear model on account features |

## 8. GNN Architecture

### Task Formulation
**Node Classification**: Classify each account as IO (1) or Control (0)

**Rationale**: The dataset provides `is_control` labels at the post level, which we aggregate to the account level. An account is labeled IO if any of its posts are from an IO campaign.

### GCN (Graph Convolutional Network)
```
Input Features (7) → GCNConv(7, 64) → BN → ReLU → Dropout
                   → GCNConv(64, 64) → BN → ReLU → Dropout
                   → GCNConv(64, 2) → Output
```

### GraphSAGE
Same architecture but using `SAGEConv` with mean aggregation, which performs neighborhood sampling.

## 9. Data Splitting

### Strategy: Account-Level Split
- **70% train / 15% val / 15% test** at the account level
- Posts from the same account never appear in both train and test
- **Rationale**: Prevents information leakage where the model learns to recognize specific accounts rather than coordination patterns

### Why Not Random Split?
Random post-level splitting would cause posts from the same IO account to appear in both train and test, allowing the model to memorize account-specific patterns rather than learning generalizable coordination signals.

### Why Not Temporal Split?
While temporally defensible, the dataset spans 6-12 years per campaign. A strict temporal cutoff may not be meaningful when the goal is detecting coordinated behavior within overlapping time windows.

## 10. Evaluation

### Account-Level Classification Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- ROC-AUC
- PR-AUC

### Community Detection Metrics
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Community Purity

### Model Comparison
All methods are evaluated on the same test split:

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| Threshold | - | - | - |
| Louvain | - | - | - |
| Random Forest | - | - | - |
| Logistic Regression | - | - | - |
| GCN | - | - | - |
| GraphSAGE | - | - | - |

*(Results will be populated after running experiments)*

## 11. Explainability

The system provides **interpretable evidence** for every flagged group:

```
Group ID: 17
Accounts: A123, A456, A789, A912
Coordination Score: 0.91

Evidence:
- Accounts A123 and A456 posted semantically similar content (similarity: 0.94)
  within 18 seconds of each other
- Shared 4 URLs and 6 hashtags
- A789 reposted content from A456

Explanation Methods:
- Feature importance (from Random Forest)
- Permutation importance
- Edge-level coordination analysis
```

### GNN Explainability
- Feature ablation study
- Permutation importance on node features
- GNNExplainer (when installed)

## 12. Ablation Study

Feature categories are removed one at a time:

| Ablation | F1 | ROC-AUC | Δ F1 |
|----------|-----|---------|------|
| Full model | - | - | - |
| No semantic | - | - | - |
| No temporal | - | - | - |
| No URL | - | - | - |
| No hashtag | - | - | - |
| No mention | - | - | - |
| No repost | - | - | - |

## 13. Visualizations

Output to `results/figures/`:
- `network_graph.png` — Coordination network (nodes=accounts, edges=coordination)
- `communities.png` — Detected communities
- `coordination_distribution.png` — Distribution of coordination scores
- `model_comparison.png` — Bar chart of model performance
- `confusion_matrix.png` — Confusion matrix for best model
- `roc_curve.png` — ROC curves for all models
- `pr_curve.png` — Precision-Recall curves
- `ablation_study.png` — Feature ablation results
- `group_explanation_*.png` — Explanation dashboards

## 14. Limitations

1. **Dataset restriction**: Requires academic access; data is not publicly downloadable
2. **Anonymization**: URLs, mentions, and account IDs are hashed, limiting some analyses
3. **Platform specificity**: Dataset is from a single platform (likely Twitter/X)
4. **Class imbalance**: IO accounts are typically outnumbered by control accounts
5. **Temporal scope**: Campaigns span different time periods, making cross-campaign comparison challenging
6. **Coordination threshold**: The edge creation threshold is somewhat arbitrary
7. **Generalizability**: Models trained on known campaigns may not generalize to novel IO tactics

## 15. Ethical Considerations

1. **This system does not accuse accounts of being malicious** — it identifies coordination patterns
2. **Coordination ≠ Malice**: Legitimate organizations may coordinate without malicious intent
3. **False positives**: The system will flag some legitimate coordinated activity
4. **Privacy**: All data is anonymized per the dataset's privacy policy
5. **Dual use**: The detection methods could potentially be used to evade detection
6. **Human oversight**: All flagged groups should be reviewed by trained analysts
7. **Context matters**: Coordination signals must be interpreted alongside other evidence

## 16. How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run with Sample Data
```bash
python train.py --sample
```

### Run with Real Dataset
1. Download data to `data/raw/`
2. Edit `configs/config.yaml` to set the campaign name
3. Run:
```bash
python train.py --config configs/config.yaml
```

### Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## 17. Reproducing Experiments

### Full Experiment Suite
```bash
python train.py --config configs/config.yaml
```

### Specific Campaign
```bash
python train.py --campaign Armenia
python train.py --campaign Cuba
```

### Ablation Study
Set `experiments.ablation: true` in config and run the pipeline.

### Cross-Campaign Evaluation
Set `experiments.cross_campaign: true` (requires loading multiple campaigns).

## 18. Project Structure

```
project/
├── data/
│   ├── raw/                    # Downloaded CSV files
│   └── processed/              # Processed data
├── src/
│   ├── data/
│   │   ├── loader.py          # Dataset loading and discovery
│   │   └── preprocessing.py   # Data cleaning and normalization
│   ├── nlp/
│   │   ├── embeddings.py      # Sentence Transformer embeddings
│   │   └── similarity.py      # FAISS/sklearn similarity search
│   ├── features/
│   │   ├── semantic.py        # Semantic coordination features
│   │   ├── temporal.py        # Temporal coordination features
│   │   ├── network.py         # Account-level network features
│   │   └── edge_features.py   # Pairwise edge features
│   ├── graph/
│   │   ├── builder.py         # NetworkX + PyG graph construction
│   │   ├── community.py       # Louvain community detection
│   │   └── visualization.py   # Plotting and visualization
│   ├── models/
│   │   ├── baseline.py        # RF, LR, threshold baselines
│   │   ├── gcn.py             # Graph Convolutional Network
│   │   └── graphsage.py       # GraphSAGE
│   ├── evaluation/
│   │   ├── metrics.py         # Classification + community metrics
│   │   └── experiments.py     # Experiment runner + ablation
│   └── pipeline.py            # End-to-end orchestration
├── app/
│   └── streamlit_app.py       # Interactive dashboard
├── tests/
│   └── test_pipeline.py       # Unit tests
├── configs/
│   └── config.yaml            # Configuration file
├── results/
│   ├── figures/               # Generated plots
│   └── models/                # Saved model weights
├── requirements.txt
├── train.py                   # Main entry point
└── README.md
```

## 19. Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data | pandas, numpy |
| ML | scikit-learn |
| NLP | sentence-transformers |
| Similarity | FAISS, sklearn |
| Graph Analysis | NetworkX |
| GNN | PyTorch, PyTorch Geometric |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Config | PyYAML |

## 20. Citation

If you use this system or the dataset, please cite:

```bibtex
@inproceedings{seckin2025labeled,
  title={Labeled Datasets for Research on Information Operations},
  author={Seçkin, Özgür Can and Pote, Manita and Nwala, Alexander C. and
          Yin, Lake and Luceri, Luca and Flammini, Alessandro and Menczer, Filippo},
  booktitle={ICWSM},
  year={2025}
}
```

## 21. License

This project code is provided for research purposes.
The dataset is licensed under CC BY-NC-ND 4.0.
