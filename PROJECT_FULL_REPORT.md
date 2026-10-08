# Coordinated Misinformation Network Detection System
## Complete Project Documentation (everything about this repo, in one file)

> **One-line summary:** An end-to-end ML research pipeline that detects **coordinated networks of accounts** (state-sponsored Information Operations) in social media data by combining NLP semantic matching, temporal/content-sharing behavioral signals, graph construction, community detection, and Graph Neural Networks — with rigorous evaluation, ablation, explainability, and a Streamlit dashboard.

> **⚠️ Core disclaimer (from the README):** The system detects **coordination patterns**. It does **NOT** establish that an account is malicious, that content is false, or that an Information Operation is occurring. Coordination is one signal among many that warrants further human investigation.

---

## Table of Contents
1. [Purpose & Problem Statement](#1-purpose--problem-statement)
2. [Research Gap & Motivation](#2-research-gap--motivation)
3. [Datasets](#3-datasets)
4. [High-Level Architecture](#4-high-level-architecture)
5. [Module-by-Module Code Reference (every file)](#5-module-by-module-code-reference)
6. [Feature Engineering (all features & formulas)](#6-feature-engineering)
7. [Graph Construction](#7-graph-construction)
8. [Community Detection & Suspicious-Group Scoring](#8-community-detection)
9. [Models (GNNs + Baselines)](#9-models)
10. [Evaluation Protocol & Actual Results](#10-evaluation-protocol--actual-results)
11. [Explainability](#11-explainability)
12. [Paraphrase-Robustness Experiment](#12-paraphrase-robustness-experiment)
13. [Degradation Tracking & Data Provenance](#13-degradation-tracking--data-provenance)
14. [Configuration Reference (config.yaml)](#14-configuration-reference)
15. [Outputs & Artifacts (results/ directory)](#15-outputs--artifacts)
16. [Scripts & Tests](#16-scripts--tests)
17. [Streamlit Dashboard](#17-streamlit-dashboard)
18. [Technology Stack](#18-technology-stack)
19. [How to Run Everything](#19-how-to-run-everything)
20. [Development History: Code Review & Fix Pass](#20-development-history)
21. [Limitations](#21-limitations)
22. [Ethical Considerations](#22-ethical-considerations)
23. [Citation & License](#23-citation--license)

---

## 1. Purpose & Problem Statement

State-sponsored **Information Operations (IOs)** involve coordinated groups of social media accounts that manipulate public discourse (e.g., the Honduras election-influence campaign analyzed here). Detecting these coordinated networks is critical for platform integrity, but hard because:

- IO accounts **mimic legitimate behavior**
- Coordination can be **subtle and distributed**
- Existing datasets lack comprehensive **control data**
- Detection must **generalize** across campaigns and countries

**What this project builds:** a systematic approach that combines **semantic similarity** (do accounts post paraphrased versions of the same messaging?), **temporal coordination** (do they post within minutes of each other?), **shared content** (URLs/hashtags/mentions/reposts), and **graph structure** (does the coordination graph form dense suspicious communities?) — while being rigorously evaluated against baselines and ablated.

**Task formulation:** **Node classification** — each account is labeled IO (1) or Control (0). Ground truth comes from the dataset's `is_control` flag at post level, aggregated: an account is IO if **any** of its posts come from an IO campaign.

---

## 2. Research Gap & Motivation

Previous work focused on:
- **Post-level classification** (individual false/misleading posts)
- **Account-level classification** (individual inauthentic accounts)
- **Network analysis** of *known* IO campaigns

What was missing:
1. Most methods treat accounts independently, ignoring coordination signals
2. Limited systematic comparison between traditional ML and GNNs for this task
3. Lack of **interpretable explanations** for why groups are flagged
4. No comprehensive **ablation study** across coordination feature categories
5. (This project's added contribution) Most semantic matching is **bi-encoder-only**; this system adds a **cross-encoder rerank stage** that is robust to paraphrased/LLM-rewritten coordinated content

---

## 3. Datasets

### 3.1 Primary source (the one all reported results use)
- **"Twitter dataset about Information Operations in Honduras and UAE"**
- Zenodo record **13912659** — https://zenodo.org/records/13912659
- Paper: Cima et al., *"Coordinated Behavior in Information Operations on Twitter"*, IEEE Access 2024
- Two campaigns available: **Honduras** (default) and **UAE**; each has `bad` (IO) and `good` (control) anonymized JSONL files
- Downloaded/converted by `src/data/convert_zenodo.py` or `python train.py --convert`

### 3.2 Broader dataset referenced in README
- **"Labeled Datasets for Research on Information Operations"** — Zenodo record 14141550 (Seçkin et al., ICWSM 2025): 26 campaigns, 16 state actors, 13M+ posts, 303k accounts, 703 files. Requires academic access request.

### 3.3 The 19-column schema (converted pipeline format)

| Column | Description |
|---|---|
| `postid` | Unique post identifier |
| `post_text` | Text content (PII hashed) |
| `application_name` | Hashed application name |
| `post_language` | Language of the post |
| `in_reply_to_postid` / `in_reply_to_accountid` | Reply targets |
| `post_time` | Timestamp |
| `accountid` | Anonymized account identifier |
| `account_profile_description` | Profile bio |
| `follower_count` / `following_count` | Follower stats |
| `account_creation_date` | Account age |
| `is_repost` / `reposted_accountid` / `reposted_postid` | Repost links |
| `hashtags` / `urls` / `account_mentions` | Comma-separated lists (URLs/IDs hashed) |
| `is_control` | **True = control (organic), False = IO account** ← ground truth |

### 3.4 Data actually on disk & used in the final run
- `results/pipeline_results.json` says: `data_source: "real"`, **32,000 posts** (20,000 IO + 12,000 control) from **3,733 accounts** (Honduras election IO campaign)
- `data/pair_truth.csv` — 677 labeled post-pairs created by `scripts/fetch_zenodo_sample.py`: **positive pairs** = posts generated from the same "template shape" (coordinated messaging), **negative pairs** = random cross pairs. Used to measure retrieval recall of the semantic matcher.
- `data/sample_backup/` — backup of the synthetic sample generator's output (`SYNTHETIC_DATA.txt` provenance marker + `sample_campaign.csv`)
- Synthetic sample data can be generated for testing via `python train.py --sample` (50 accounts by default, written to `data/raw/sample_campaign.csv`); **strictly opt-in** for real runs (see §13)

---

## 4. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1  DATA LOADING          src/data/loader.py               │
│   Zenodo JSONL → CSV → pandas DataFrame (Honduras/UAE/sample)   │
│   Provenance check: real vs synthetic (never silently faked)    │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2  PREPROCESSING         src/data/preprocessing.py        │
│   Schema validation, drop malformed, timestamp parsing,         │
│   text normalization, URL/hashtag/mention extraction,           │
│   repost identification, account ID canonicalization,           │
│   per-account IO/Control labels                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3  BI-ENCODER EMBEDDING  src/nlp/embeddings.py            │
│   all-MiniLM-L6-v2 → 384-dim L2-normalized embedding per post   │
│   (batch 256, CUDA/CPU auto); stashed for node-feature PCA      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4  POST-LEVEL RETRIEVAL  src/nlp/similarity.py            │
│   FAISS (flat index) top-k=20 post pairs above floor 0.30       │
│   (sklearn NN fallback). Aggregate to account pairs             │
│   (top-3 post pairs kept per account pair)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4b  CROSS-ENCODER RERANK src/nlp/similarity.py            │
│   cross-encoder/nli-deberta-v3-base scores the ACTUAL matched   │
│   post texts (precision gate); capped at 20,000 pairs;          │
│   unscored pairs keep bi-encoder score (recorded degradation)   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5  EDGE FEATURES         src/features/*.py                │
│   semantic + temporal + URL + hashtag + mention + repost        │
│   → weighted coordination score C(A,B) per account pair         │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 6  GRAPH CONSTRUCTION    src/graph/builder.py             │
│   NetworkX undirected weighted graph (edges ≥ 0.3, cap 50/node) │
│   + PyG Data object (node features + labels + masks)            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 7  COMMUNITY DETECTION   src/graph/community.py           │
│   Louvain partition → suspiciousness scoring (pre-GNN pass)     │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 8-9  MODELS              src/models/*.py                  │
│   Baselines: Threshold / Random Forest / LR / Louvain-as-baseline│
│   GNNs: GCN + GraphSAGE — ALL on ONE canonical 70/15/15 split   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 9b  GNN → COMMUNITY INTEGRATION                           │
│   Trained GNN's per-node P(IO) re-scores communities → the GNN  │
│   provably changes which groups are flagged (diff is recorded)  │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 10-13  EVAL / ABLATION / EXPLAINABILITY / VIZ             │
│   Model comparison table, 7-condition ablation, GNNExplainer +  │
│   permutation importance, 6-signal group explanations, figures  │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 14  ARTIFACT PERSISTENCE + RESULTS JSON                   │
│   models/*.pt, graph.pkl, embeddings, pipeline_results.json     │
│   + full "degraded_components" transparency section             │
└─────────────────────────────────────────────────────────────────┘
```

**The coordination score at the heart of everything:**

```
C(A,B) = 0.35·Semantic + 0.25·Temporal + 0.15·URL
       + 0.10·Hashtag + 0.10·Repost + 0.05·Mention
```
All six components normalized to [0,1] before combination. Weights live in `configs/config.yaml` (`coordination.weights`) and are fixed by default; an optional **learned-weights mode** (`EdgeFeatureExtractor.learn_weights`, off by default) fits them by logistic regression against labels and renormalizes to sum 1.0 — falls back to fixed weights when labels are missing/degenerate.

---

## 5. Module-by-Module Code Reference

### 5.1 Entry point
**`train.py`** (~290 lines) — CLI entry point.
- Flags: `--config`, `--campaign`, `--sample`, `--convert` (download+convert Zenodo), `--convert-max-posts`, `--allow-synthetic-fallback` (strictly opt-in synthetic data), `--log-level`, `--experiment paraphrase_robustness`, `--n-pairs`
- Contains a **full default config** baked into `get_default_config()` as a fallback when no YAML is found
- Delegates everything to `CoordinationDetectionPipeline(config).run(use_sample=...)`

### 5.2 Data layer (`src/data/`)
| File | Purpose |
|---|---|
| `loader.py` | `DataLoader`: campaign file discovery (`data/raw/<Campaign>/*.csv`), CSV loading, schema validation, per-campaign / all-campaign loading, dataset info, processed save/load. `canonicalize_account_ids()` fixes the pandas float-coercion bug that once collapsed large integer IDs (this caused self-loops in the first real run). `create_sample_dataset()` generates synthetic test data + `SYNTHETIC_DATA.txt` marker. `SYNTHETIC_MARKER_FILE` constant. |
| `preprocessing.py` | `Preprocessor.process()` orchestrates: schema validation → remove malformed records (null postid/accountid) → timestamp parsing (fail-loud if drop rate too high) → text normalization (lowercase, strip URLs, keep mentions) → extract URLs/hashtags/mentions into lists → identify reposts → account mappings → per-account labels (`is_io = NOT is_control`). Every missing column/skip recorded as a degradation. |
| `convert_zenodo.py` | Downloads Zenodo record 13912659 JSONL (Honduras/UAE bad+good files), streams from URL, parses JSON arrays, converts epoch-ms timestamps, maps IO/control → the 19-column pipeline CSV schema. CLI: `python -m src.data.convert_zenodo --campaign Honduras`. |

### 5.3 NLP layer (`src/nlp/`)
| File | Purpose |
|---|---|
| `embeddings.py` | `EmbeddingModel` wraps `sentence-transformers/all-MiniLM-L6-v2` (384-dim). `encode_texts`/`encode_dataframe` batch-encode posts (batch 256, auto device), L2-normalized. `compute_account_embeddings` averages post embeddings per account. Save/load embeddings. |
| `similarity.py` | `SimilaritySearch`: FAISS flat inner-product index (cosine on normalized vectors) with sklearn `NearestNeighbors` fallback (degradation recorded if FAISS missing). `find_post_pairs` does **post-level** top-k=20 retrieval with a **retrieval floor 0.30** (recall-oriented; unrelated text rarely exceeds ~0.3 cosine with MiniLM). `find_account_similarities` for account-level mode. `CrossEncoderReranker`: loads `cross-encoder/nli-deberta-v3-base`, scores text pairs jointly (cross-attention), sigmoid-normalizes scores, `rerank()` only the top-k candidates (cost control), leaving the rest bi-encoder-only (NaN + degradation recorded). |

### 5.4 Feature layer (`src/features/`)
| File | Purpose |
|---|---|
| `semantic.py` | `SemanticFeatures`: `aggregate_post_pair_candidates` groups post pairs into account-pair candidates keeping top-N=3 matched post pairs each; `merge_semantic_scores` combines bi-encoder + cross-encoder scores (cross-encoder is the precision gate when available); max-mean / centroid / max-pairwise similarity helpers. |
| `temporal.py` | `TemporalFeatures`: temporal proximity `exp(-Δt_mean_min / τ)`, τ=3600s; burst detection (5-min window, ≥3 posts); synchronization score (fraction of posts in burst window); `compute_batch_temporal_features` over all account pairs. |
| `edge_features.py` | `EdgeFeatureExtractor`: shared-URL score `|A∩B| / max(|A|,|B|)`, shared-hashtag `|A∩B| / |A∪B|-style`, shared-mention score, repost score (direct repost + shared repost targets), `compute_coordination_score` (weighted sum, accepts config or learned weights), `learn_weights` (logistic regression on the 6 signals → normalized weights, returns None on degenerate labels), `compute_batch_edge_features` builds the full edge table with `coordination_score`, per-signal scores, `semantic_fallback` tags, and matched post-pair evidence. |
| `network.py` | `NetworkFeatures`: 7 core account features (`post_count`, `avg_follower_count`, `avg_following_count`, `repost_ratio`, `avg_hashtag_count`, `avg_url_count`, `avg_mention_count`) plus derived extras (`posts_per_day`, `follower_following_ratio`, `reply_ratio`, uniqueness ratios), network centrality, repost-network features, interaction network construction, `select_node_features()`. |

### 5.5 Graph layer (`src/graph/`)
| File | Purpose |
|---|---|
| `builder.py` | `GraphBuilder.build_networkx_graph`: undirected weighted graph; edge only when `coordination_score ≥ edge_threshold (0.3)`; drops `min_edge_score (0.1)`; prunes to ≤50 edges/node (keeps highest-scoring). `build_pyg_data` converts to `torch_geometric.data.Data` with node features, labels, edge weights. `get_graph_statistics` (nodes, edges, density, degree stats, components, clustering, edge-weight stats). Save/load via pickle. |
| `community.py` | `CommunityDetector`: Louvain (via `python-louvain`; greedy-modularity fallback = recorded degradation), label-propagation alternative. `score_communities`: suspiciousness = `0.3·density + 0.3·avg_coordination + 0.2·size_factor + 0.2·io_ratio_from_gnn` — when GNN per-node P(IO) scores are supplied they replace the label-derived io_ratio (this is how the GNN influences final flagging). `get_suspicious_groups(top_k=10, min_size=3)`. `generate_group_explanations`: per-group numeric 6-signal breakdown + matched post pairs as evidence. |
| `visualization.py` | `GraphVisualizer`: network plot (max 200 nodes), community plot, coordination-score distribution, confusion matrix, ROC curves, PR curves, model-comparison bars, ablation bars, per-group explanation dashboards. Saves PNGs at config DPI. |

### 5.6 Model layer (`src/models/`)
| File | Purpose |
|---|---|
| `gcn.py` | `GCN` (torch module): GCNConv(in→64) → BN → ReLU → Dropout(0.5) → GCNConv(64→64) → BN → ReLU → Dropout → GCNConv(64→2). `GCNModel` wrapper: build/train (Adam lr=0.001, weight_decay 5e-4, class-weighted cross-entropy, early stopping patience 10 on val loss, best-checkpoint restore), full-batch training, predict (logits + softmax probs), node embeddings, evaluate (accuracy/precision/recall/F1/ROC-AUC/PR-AUC). |
| `graphsage.py` | `GraphSAGE`: identical wrapper/architecture but `SAGEConv` layers, configurable aggregator (`mean` default; `max`/`lstm` options). Also **full-batch** (no NeighborLoader — exact but memory-bound). |
| `baseline.py` | `BaselineModels`: threshold baseline (sweep 0.1–0.95, optimize F1 on train only), RandomForest (100 trees, depth 20, balanced), LogisticRegression (max_iter 1000, StandardScaler, balanced), `get_feature_importance`, `evaluate_predictions` (accuracy/precision/recall/F1/ROC-AUC/PR-AUC). |

### 5.7 Evaluation layer (`src/evaluation/`)
| File | Purpose |
|---|---|
| `metrics.py` | `EvaluationMetrics`: classification metrics (accuracy, weighted precision/recall/F1, ROC-AUC, PR-AUC), community metrics (ARI, NMI, purity), link-prediction metrics, report formatting, model comparison DataFrame. |
| `experiments.py` | `ExperimentRunner`: `run_model_comparison` (builds the comparison CSV), `save_experiment_config`, `run_paraphrase_robustness` (delegates to the dedicated module). |
| `paraphrase_robustness.py` | `ParaphraseRobustnessExperiment`: samples ~200 known-coordinated pairs (IO accounts, bi-encoder ≥ 0.7), paraphrases one side with Claude (`claude-sonnet-4-6` via the `anthropic` package), re-scores original-original vs original-paraphrased with bi-encoder AND cross-encoder, produces before/after tables + `dist_semantic_similarity.png`. This is the experimental contribution demonstrating **why cross-encoder reranking matters against LLM-paraphrased coordination**. Runnable standalone: `python -m src.evaluation.paraphrase_robustness --sample`. |

### 5.8 Orchestration
**`src/pipeline.py`** (~1,340 lines) — `CoordinationDetectionPipeline`. Wires all 16 components together and runs the 14 stages described in §4. Key internal methods:
- `run()` — the full pipeline; maintains runtime state (`DegradationTracker`, stashed post embeddings, per-model ROC data, canonical split)
- `_load_data()` — provenance-safe loading (see §13)
- `_run_baselines()` — threshold/RF/LR/Louvain all scored on the canonical test indices; threshold baseline input = **mean incident edge coordination score** (was mistakenly post count before the fix)
- `_run_gnn_models()` — node features = 7 behavioral stats (standardized) **+ 16-dim PCA of account content embedding**; masks derived from the canonical split; produces per-node P(IO) used for community re-scoring
- `_run_explainability()` — GNNExplainer (torch_geometric.explain, 20 sampled nodes) + permutation importance over the 6 signals (F1 drop, 10 runs)
- `_run_ablation()` — 7 conditions retraining the **edge-feature-consuming classifier** (so ablation is non-vacuous)
- `_persist_artifacts()` — checkpoints bundled with x/edge_index/test_mask for reproducibility
- `_create_visualizations()`, `_save_results()`

### 5.9 Utilities
**`src/utils/degradations.py`** — `DegradationTracker`: collects every non-fatal fallback/skip as `{component, reason, detail}` events, logs each at WARNING, and surfaces the list in the results JSON under `degraded_components`. Philosophy: **every degradation is explicit, never silent.**

---

## 6. Feature Engineering

### 6.1 Node (account) features — GNN input (7 + 16 dims)
| Feature | Description |
|---|---|
| `post_count` | Number of posts |
| `avg_follower_count` | Mean followers |
| `avg_following_count` | Mean following |
| `repost_ratio` | Reposts / total posts |
| `avg_hashtag_count` | Hashtags per post |
| `avg_url_count` | URLs per post |
| `avg_mention_count` | Mentions per post |
| + PCA content embedding | 16 principal components of the account's mean MiniLM post embedding (item 11 — gives the GNN semantic content at node level) |

Behavioral columns are StandardScaler-normalized; PCA dims are already centered. Extra derived features (`posts_per_day`, `follower_following_ratio`, `reply_ratio`, uniqueness ratios) exist and `posts_per_day` is used by the ML baselines.

### 6.2 Edge (coordination) features
| Signal | Weight | Formula / meaning |
|---|---|---|
| `semantic_similarity` | 0.35 | Best cross-encoder score among matched post pairs (fallback: best bi-encoder score; retrieval floor 0.30, k=20) |
| `temporal_score` | 0.25 | `exp(-mean_min_time_gap / 3600s)` |
| `shared_url_score` | 0.15 | `\|shared URLs\| / max(\|urls_A\|, \|urls_B\|)` |
| `shared_hashtag_score` | 0.10 | common hashtags / unique hashtags |
| `repost_score` | 0.10 | direct reposts + shared repost targets |
| `shared_mention_score` | 0.05 | common mentions / unique mentions |

Edge created iff `C(A,B) ≥ 0.3`; edges below `min_edge_score = 0.1` dropped; ≤50 edges per node.

### 6.3 Temporal features detail
- Exponential time decay with **τ = 1 hour**
- **Burst detection:** clusters of ≥3 posts within a 5-minute window
- **Synchronization score:** fraction of an account pair's posts inside burst windows

### 6.4 Semantic pipeline detail (retrieve-then-rerank)
1. Bi-encoder (MiniLM) embeds every post → FAISS flat IP index
2. Top-20 neighbors per post, floor 0.30 → candidate post pairs
3. Aggregate to account pairs, keep top-3 post pairs per account pair
4. Cross-encoder (`nli-deberta-v3-base`) jointly scores the actual matched texts (sigmoid-normalized); capped at 20,000 pairs/run (of 362,427 candidates in the reported run; 16,833 account pairs got cross-encoder scores)
5. Account-pair semantic score = max cross-encoder score among its pairs; unscored pairs fall back to bi-encoder and are flagged lower-confidence
6. Winning matched pairs are stored **as evidence** for explanations

---

## 7. Graph Construction

- **Node = social media account**; **Edge = evidence of coordination** (score ≥ threshold — no edges for merely existing accounts)
- Undirected, weighted; self-loops explicitly guarded against (a real bug once inflated scores)
- Node features: §6.1 vector; edge features: all 6 signals + combined score
- PyG `Data` for GNN training with `x`, `edge_index`, `edge_attr`, `y`, masks
- Final run graph stats (Honduras, 32k posts): **3,733 nodes, 31,207 edges**, density 4.48e-3, avg degree 16.72, median degree 10, max degree 50 (cap), 620 components, avg clustering 0.253, avg edge weight 0.424 (max 0.934)
- (An earlier full-campaign run over 1.26M posts / 224,685 accounts produced 987,981 edges — the pipeline scales end-to-end)

---

## 8. Community Detection

- **Primary:** Louvain modularity optimization (`python-louvain`); **fallback:** NetworkX greedy modularity (degradation recorded); alternative: label propagation
- **Suspiciousness score** per community:
  `0.3·density + 0.3·avg_edge_coordination + 0.2·size_factor + 0.2·io_ratio`
- Two scoring passes: **pre-GNN** (io_ratio from labels — kept for the integration-diff evidence) and **post-GNN** (io_ratio replaced by mean GNN P(IO))
- **Top-10 suspicious groups** (min size 3) become the flagged output, each with a full numeric explanation
- Worked example (group 121, 4 accounts): semantic 0.790, temporal 0.166, URL 0.0, hashtag 0.0, repost 0.0, mention 1.0 → C = 0.35·0.790 + 0.25·0.166 + 0.05·1.0 = **0.368** (stored: 0.3678); GNN P(IO) = 0.995; combined suspiciousness 0.689
- **Measured GNN effect:** comparing pre- vs post-GNN top-10 groups, 3 of 10 changed in composition (3 dropped, 3 added) — the GNN is not decorative

---

## 9. Models

### 9.1 GCN (`src/models/gcn.py`)
```
Input → GCNConv(in,64) → BatchNorm → ReLU → Dropout(0.5)
      → GCNConv(64,64) → BatchNorm → ReLU → Dropout(0.5)
      → GCNConv(64,2) → logits
```
Adam (lr 1e-3, wd 5e-4), class-weighted CE loss, early stopping (patience 10, val loss), max 100 epochs, full-batch, best-checkpoint restore.

### 9.2 GraphSAGE (`src/models/graphsage.py`)
Same training regime; `SAGEConv` layers with mean aggregation. Full-batch training (no neighbor sampling) — exact but memory-bound on very large graphs.

### 9.3 Baselines (`src/models/baseline.py`)
1. **Threshold baseline** — mean incident-edge coordination score per account; threshold swept 0.1–0.95, optimized for F1 on **train split only**
2. **Random Forest** — 100 trees, max_depth 20, `class_weight="balanced"`, on the 8 behavioral features
3. **Logistic Regression** — StandardScaler + max_iter 1000, balanced
4. **Louvain as an unsupervised baseline** — community predicted IO if its mean edge coordination score ≥ edge threshold (no label leakage; an earlier version used io_ratio, which was leakage)

### 9.4 NLP models
- Bi-encoder: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Cross-encoder: `cross-encoder/nli-deberta-v3-base` (NLI-entailment model used as paraphrase scorer — an acknowledged proxy, see §21)

---

## 10. Evaluation Protocol & Actual Results

### 10.1 Protocol
- **One canonical account-level 70/15/15 split** (seed 42, stratified): **2,613 train / 560 val / 560 test**. Every model — threshold, RF, LR, Louvain, GCN, GraphSAGE — is scored on the **identical 560 held-out accounts** (`n_test=560` stamped on every row of `model_comparison.csv`). Account-level splitting prevents leakage (posts of one account never span splits). This uniformity was itself a major development fix: earlier runs mixed a full-set threshold baseline, an 80/20 RF/LR split, and a random 15% GNN mask, making the original comparison uninterpretable.
- Metrics: accuracy, weighted precision/recall/F1, ROC-AUC, PR-AUC; community metrics: ARI, NMI, purity.

### 10.2 Final results (real Honduras data, from `results/model_comparison.csv`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Louvain (community-membership baseline) | 0.409 | 0.347 | 0.832 | 0.490 | — | — |
| Threshold baseline | 0.341 | 0.341 | 0.995 | 0.507 | 0.126 | 0.208 |
| GCN | 0.959 | 0.959 | 0.959 | 0.959 | 0.989 | 0.986 |
| Logistic regression | 0.982 | 0.984 | 0.963 | 0.974 | 0.993 | 0.992 |
| GraphSAGE | 0.984 | 0.984 | 0.984 | 0.984 | 0.999 | 0.997 |
| **Random Forest** | **0.996** | 0.990 | 1.000 | **0.995** | 0.998 | 0.992 |

Interpretation: the weakest performers are **unsupervised** (Louvain flags whole communities off edge scores alone; threshold predicts nearly everything IO — recall 0.995, precision 0.34). The strongest (RF, F1 0.995) is trained directly on labeled behavioral features. The gap therefore mostly reflects access to labeled training signal, not failure of the graph evidence. GCN/GraphSAGE combine graph structure + behavioral + PCA-content features.

### 10.3 Ablation results (from `results/ablation_results.csv`)
Each condition retrains the edge-feature-consuming classifier (earlier vacuous version — a node-feature-only model unaffected by ablation — was identified and fixed).

| Ablation | Accuracy | F1 |
|---|---|---|
| **Full model** | 0.9813 | 0.9813 |
| no semantic | 0.9639 | 0.9640 (2nd-largest drop) |
| no temporal | 0.9839 | 0.9840 |
| no url | 0.9826 | 0.9826 |
| **no hashtag** | 0.9250 | **0.9256 (largest drop)** |
| no mention | 0.9786 | 0.9785 |
| no repost | 0.9732 | 0.9733 |

Hashtag overlap and semantic similarity are the most load-bearing coordination signals in this dataset; temporal is mildly redundant.

---

## 11. Explainability

Three layers, all implemented:

1. **Per-group evidence** — every flagged group exposes:
   - numeric 6-signal breakdown with the exact weighted-sum arithmetic
   - the specific matched post pairs (post IDs + texts) as evidence
   - GNN-derived P(IO) and combined suspiciousness
   - rendered as `results/figures/group_explanation_*.png` dashboards
2. **Model-level importance** — Random Forest / Logistic Regression feature importances; permutation importance over the 6 coordination signals (F1 drop across 10 runs, seed 42)
3. **GNN-level** — **GNNExplainer** (`torch_geometric.explain`) over the trained GCN: node-feature + edge masks, averaged over 20 sampled nodes → `gnnexplainer_feature_importance` in the results JSON (toggle: `explainability.use_gnnexplainer`)

---

## 12. Paraphrase-Robustness Experiment

**The key experimental contribution** (`src/evaluation/paraphrase_robustness.py`, run standalone via `python train.py --experiment paraphrase_robustness`):

1. Sample ~200 known-coordinated post pairs (IO accounts, bi-encoder ≥ 0.7)
2. Paraphrase one post per pair with **Claude** (`claude-sonnet-4-6`, via `anthropic`)
3. Re-score original-original vs original-paraphrased with **both** bi-encoder and cross-encoder
4. Output: comparison table + `results/figures/dist_semantic_similarity.png`

**Finding it demonstrates:** bi-encoder scores **degrade** on LLM-paraphrased coordinated content while cross-encoder scores **stay high** — i.e., cross-encoder reranking is what makes the system robust to adversaries who paraphrase their messaging. Adversaries mimicking legitimate behavior with paraphrase (LLM-generated) content can evade the bi-encoder stage, but the cross-encoder gate catches it. Outputs include the before/after similarity distributions and a summary table (results saved to `results/`).

---

## 13. Degradation Tracking & Data Provenance

A defining design principle of the mature codebase: **nothing degrades silently**.

- `DegradationTracker` collects every fallback: FAISS→sklearn, Louvain→greedy modularity, cross-encoder bypass/cap, missing optional columns, unparseable timestamps, zero semantic candidates (falls back to top-50-by-volume account pairs tagged `semantic_fallback=True`), rerank-capped pairs, failed persistence, stale synthetic markers, etc.
- Each event: WARNING log at the moment it happens + a row in `results/pipeline_results.json → degraded_components`.

**Data provenance is load-bearing:**
- `data_source: "real" | "synthetic"` is stamped in the results
- Synthetic data is generated **only** on explicit request (`--sample`) or explicit `--allow-synthetic-fallback`; missing real data without the flag is a **fatal error**
- Provenance derives from **which files were actually loaded** (`sample_campaign.csv` present ⇒ synthetic), not from a marker file (a stale `SYNTHETIC_DATA.txt` alongside real data is recorded as a degradation and ignored — a real mis-tagging bug that was caught and fixed)
- Tests write into `tmp_path`, never into `data/raw`, so they can't poison provenance

---

## 14. Configuration Reference

Everything is driven by `configs/config.yaml` (with a full in-code default in `train.py`). Sections:

| Section | Key settings |
|---|---|
| `data` | `raw_dir`, `processed_dir`, `campaign` (Honduras default), `max_files_per_campaign: 10`, `max_posts`, `subsample_fraction`, `random_seed: 42` |
| `preprocessing` | lowercase, strip URLs from text (keep mentions), min_text_length 3, drop_malformed |
| `nlp` | MiniLM model, 384 dims, batch 256, auto device |
| `similarity` | faiss, k=20, **retrieval_floor 0.30** (recall floor, deliberately not a precision filter), semantic_threshold 0.7 (bi-encoder-only mode), faiss_index_type flat, **use_cross_encoder_rerank: true**, `cross-encoder/nli-deberta-v3-base`, batch 64, `cross_encoder_rerank_k: null`, max_post_pairs_per_account_pair 3, **max_rerank_pairs: 20000** |
| `temporal` | tau 3600, burst_window 300s, burst_min_posts 3 |
| `coordination` | edge_threshold 0.3, weights (0.35/0.25/0.15/0.10/0.05/0.10), max_edges_per_node 50, min_edge_score 0.1, `learn_weights` (off) |
| `graph` | 7 node features, 6 edge features listed |
| `models` | GCN & GraphSAGE: hidden 64, 2 layers, dropout 0.5, lr 1e-3, 100 epochs, patience 10; RF: 100/20/seed42; LR: max_iter 1000 |
| `splitting` | account strategy, 0.7/0.15/0.15 |
| `evaluation` | metric lists, cv_folds null |
| `explainability` | feature+permutation importance, 10 runs, shap off, **use_gnnexplainer: true** |
| `visualization` | figures dir, png, dpi 150, max 200 network nodes, color scheme (IO red, control green, suspicious orange) |
| `experiments` | **ablation: true** (6 feature categories), cross_campaign: false, output dir `results` |
| `paraphrase_experiment` | 200 pairs, min_biencoder_score 0.7, model `claude-sonnet-4-6` |
| `streamlit` | port 8501, dark theme, display caps |

---

## 15. Outputs & Artifacts

### `results/` after a full run
| Path | Contents |
|---|---|
| `pipeline_results.json` | Everything: data_source, data/graph stats, split info, coordination weights used, model_comparison, ablation table, suspicious groups with explanations, gnn_integration_diff, explainability (GNNExplainer + permutation importance), degraded_components |
| `model_comparison.csv` | 6 rows × metrics + `n_test=560` proof column |
| `ablation_results.csv` | 7 conditions |
| `models/gcn.pt`, `models/graphsage.pt` | Checkpoints **bundled** with model_state_dict + node features + edge_index + test_mask (reproducible from the file alone — verified by `scripts/reproduce_predictions.py`) |
| `artifacts/coordination_graph.pkl` | The NetworkX graph |
| `artifacts/post_embeddings.npy` + `posts_index.csv` | Post-level embedding matrix + index |
| `artifacts/account_embeddings.csv` | Account-level mean embeddings |
| `figures/network_graph.png` | Coordination network |
| `figures/communities.png` | Louvain communities |
| `figures/coordination_distribution.png` | Edge-score distribution |
| `figures/model_comparison.png`, `ablation_study.png` | Result charts |
| `figures/confusion_matrix.png`, `roc_curve.png`, `pr_curve.png` | Best-model diagnostics (real per-model score data) |
| `figures/dist_semantic_similarity.png` | Paraphrase-robustness distributions |
| `figures/group_explanation_0/1/2.png` | Explanation dashboards |
| `run_honduras*.log` | Run logs (3 versions from development) |

---

## 16. Scripts & Tests

### `scripts/`
| Script | Purpose |
|---|---|
| `fetch_zenodo_sample.py` | Pulls a capped sample (default 30MB/file, 20k bad + 12k good posts) of Zenodo record 13912659 via curl; builds `data/pair_truth.csv` using a text-**shape** heuristic (numbers/long words → type tokens) so positive pairs = same-template coordinated posts; used to measure retrieval recall |
| `verify_checklist.py` | Post-run automated acceptance check: reads pipeline_results.json, comparison/ablation CSVs, artifacts, config → PASS/FAIL verdict per acceptance criterion with logged evidence (12-point checklist from the code review) |
| `reproduce_predictions.py` | Loads the `.pt` checkpoint bundles and re-derives GNN predictions on the held-out test nodes — proves the saved weights work without re-running the pipeline |

### `tests/` (pytest, 2 files)
| File | Covers |
|---|---|
| `test_pipeline.py` (~350 lines) | Core component smoke tests on synthetic tmp-path data: loader, preprocessor, semantic/temporal/edge features, graph builder, community detection, metrics |
| `test_fixes.py` (~520 lines) | **Regression tests for the 25-item code review**, each mapped to an acceptance criterion: ID canonicalization, self-loop guard, graph save/load roundtrip, ROC/PR/confusion plotting, cross-encoder rerank-k truncation, post-level retrieval finds planted pairs, missing-column degradations, timestamp drop-rate fail-loud, learned weights, GNN-scores-influence-flagging (headless matplotlib, sklearn backend for speed) |

Run: `python -m pytest tests/ -v`

---

## 17. Streamlit Dashboard

**`app/streamlit_app.py`** (~470 lines) — `streamlit run app/streamlit_app.py` (port 8501, dark theme).

Interactive tabs/views over `results/pipeline_results.json` + graph + artifacts:
- **Overview metrics** — posts, accounts, edges, suspicious groups (styled metric cards)
- **Suspicious groups explorer** — ranked groups with the numeric 6-signal breakdown, coordination arithmetic, evidence (matched post pairs), GNN P(IO)
- **Network graph** — plotly interactive network (IO=red / control=green / suspicious=orange), capped display
- **Coordination scores** — distributions
- **Coordination weights** — **read live from `configs/config.yaml`** (an earlier version hardcoded them, so config edits didn't reflect — fixed)
- **Model comparison** — charts of the comparison table
- Display caps: 50 groups / 100 accounts

---

## 18. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Data | pandas ≥ 2.0, numpy ≥ 1.23 |
| ML | scikit-learn ≥ 1.1 (RF, LR, PCA, scaler, metrics, threshold tuning) |
| Deep learning | torch ≥ 2.0, torch-geometric ≥ 2.3 |
| NLP | sentence-transformers ≥ 2.2 (MiniLM-L6-v2 bi-encoder), cross-encoder/nli-deberta-v3-base |
| Similarity search | faiss-cpu ≥ 1.7 (primary), sklearn NN (fallback) |
| Graph | networkx ≥ 3.0, python-louvain ≥ 0.16 |
| LLM (experiment) | anthropic ≥ 0.30 (Claude paraphrasing) |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | streamlit ≥ 1.20 |
| Config / utils | PyYAML, tqdm, pytest ≥ 7 |

---

## 19. How to Run Everything

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get data (either)
python train.py --convert --campaign Honduras      # download + convert Zenodo IO data
python train.py --sample                           # synthetic sample (explicitly tagged)

# 3. Full pipeline (trains everything, writes results/)
python train.py --config configs/config.yaml
python train.py --campaign Honduras                # override campaign

# 4. Verify the run against the acceptance checklist
python scripts/verify_checklist.py
python scripts/reproduce_predictions.py            # reload .pt checkpoints

# 5. Dashboard
streamlit run app/streamlit_app.py

# 6. Tests
python -m pytest tests/ -v

# 7. Standalone paraphrase-robustness experiment
python train.py --experiment paraphrase_robustness --n-pairs 200

# Ablation: enabled by default (experiments.ablation: true)
# Cross-campaign eval: set experiments.cross_campaign: true
```
Reported full real-data run time: **~3,295 s (≈55 min) on CPU** (32k posts; dominated by embedding + cross-encoder stages). A 1.26M-post end-to-end run also completed historically.

---

## 20. Development History

### 20.1 Initial implementation
Core computations were genuinely implemented from the start: real SentenceTransformer embeddings, real cross-encoder, real FAISS, real Louvain, real GCN/GraphSAGE training, and a 1.26M-post end-to-end real-data run.

### 20.2 The code review found five defect themes (documented in PROJECT_REVIEW.md / REPORT_SECTION.md)
1. **Data integrity** — pandas float-coercion collapsed large integer account IDs when NaNs present → distinct accounts merged → self-loops inflating coordination scores. Fixed via `canonicalize_account_ids` + self-loop guard.
2. **Core method dilution** — semantic matching operated on account-averaged embeddings and reranked one arbitrary representative post, discarding the post-level paraphrase evidence that motivated the cross-encoder. Replaced with **post-level retrieve-then-rerank** (FAISS → aggregate → cross-encoder on the actual matched texts → winning pairs kept as evidence).
3. **Evaluation validity** — threshold baseline scored on post volume (degenerate F1 0.0); three model families on three different splits; ablation retrained a model that didn't consume the ablated features (7 identical rows). All fixed: canonical split everywhere, coordination-score input for threshold, edge-feature-consuming ablation classifier.
4. **Integration gaps** — GNN output never influenced flagging; explanations covered 4/6 signals. Fixed: GNN P(IO) now re-scores communities (with a recorded pre/post diff proving effect); explanations cover all 6 signals numerically.
5. **Environment/dependencies** — `python-louvain` and `anthropic` missing from requirements (silent weaker fallbacks). Pinned; every silent fallback converted to an explicitly recorded degradation.

A second-order bug caught during fixing: tests wrote synthetic data + its provenance marker into the real `data/raw`, mis-tagging a real run as synthetic — provenance now derives from files actually loaded.

### 20.3 Verification
A 12-point acceptance checklist executed by `scripts/verify_checklist.py` — all passing on a fresh end-to-end real-data run, including: zero self-loops, comparison table fully populated on the shared split (n_test stamped), checkpoints that reload and reproduce predictions on all 560 held-out accounts, GNN-integration diff recorded, degradations surfaced.

---

## 21. Limitations

1. **Single-campaign evaluation** — all metrics are in-campaign (Honduras, 32k posts) on a held-out account split; no held-out campaign/platform test
2. **Cross-encoder is a proxy** — an NLI-entailment model used as a paraphrase detector, not a purpose-trained paraphrase model
3. **Rerank coverage capped** — top 20,000 of 362,427 candidate pairs cross-encoded; the rest carry lower-confidence bi-encoder scores (explicitly recorded)
4. **Full-batch GNN training** — no neighbor sampling; memory-bound at much larger scale
5. **Fixed scoring weights** — manual 0.35/0.25/0.15/0.10/0.10/0.05; learned-weights mode exists but was off and unused for reported results
6. **Dataset access** — requires Zenodo academic request; PII/URLs/IDs are hashed, limiting some analyses
7. **Single platform** — Twitter/X data only
8. **Class imbalance** — IO accounts are outnumbered by controls
9. **Threshold arbitrariness** — the 0.3 edge threshold is a config constant
10. **Generalizability** — models trained on known campaigns may not catch novel IO tactics

---

## 22. Ethical Considerations

1. The system **does not accuse accounts of malice** — it surfaces coordination patterns
2. **Coordination ≠ malice** — legitimate organizations coordinate
3. **False positives are expected** — some legitimate coordinated activity will be flagged
4. **Privacy** — data anonymized per the dataset's policy
5. **Dual use** — detection methods could inform evasion tactics
6. **Human oversight required** — flagged groups are analyst leads, not verdicts
7. **Context matters** — coordination signals must be read alongside other evidence

---

## 23. Citation & License

```bibtex
@inproceedings{seckin2025labeled,
  title={Labeled Datasets for Research on Information Operations},
  author={Seçkin, Özgür Can and Pote, Manita and Nwala, Alexander C. and
          Yin, Lake and Luceri, Luca and Flammini, Alessandro and Menczer, Filippo},
  booktitle={ICWSM},
  year={2025}
}
```
Also relevant: Cima et al., *"Coordinated Behavior in Information Operations on Twitter"*, IEEE Access 2024 (Zenodo record 13912659 — the dataset actually used).

Code is provided for **research purposes**; the dataset is licensed **CC BY-NC-ND 4.0**.

---

## Appendix A: Complete File Map (35 Python files + docs)

```
train.py                          CLI entry point + default config
app/streamlit_app.py              Interactive dashboard
configs/config.yaml               All hyperparameters
requirements.txt                  17 pinned dependencies

data/
  pair_truth.csv                  677 labeled post pairs (retrieval-recall eval)
  raw/  sample_backup/            Real campaign CSVs / synthetic sample backup

src/
  pipeline.py                     14-stage orchestrator (~1,340 lines)
  data/    loader.py preprocessing.py convert_zenodo.py
  nlp/     embeddings.py similarity.py            (bi-encoder + FAISS + cross-encoder)
  features/semantic.py temporal.py edge_features.py network.py
  graph/   builder.py community.py visualization.py
  models/  baseline.py gcn.py graphsage.py
  evaluation/metrics.py experiments.py paraphrase_robustness.py
  utils/   degradations.py

scripts/
  fetch_zenodo_sample.py          Data acquisition + pair_truth builder
  verify_checklist.py             12-point acceptance verification
  reproduce_predictions.py        Checkpoint reproducibility proof

tests/
  test_pipeline.py                Core component tests
  test_fixes.py                   25-item review regression tests

results/
  pipeline_results.json model_comparison.csv ablation_results.csv
  models/ artifacts/ figures/ run logs

docs: README.md (project spec), AI_ML_TECH_STACK.md (stack deep-dive),
      PROJECT_REVIEW.md (review-era results), REPORT_SECTION.md (final report section)
```
