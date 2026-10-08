"""
Main pipeline orchestrating the full coordination detection system.

Pipeline stages:
1. Data loading and preprocessing
2. NLP feature extraction (embeddings + similarity)
3. Temporal/content coordination features
4. Graph construction
5. Baseline models
6. GNN models
7. Evaluation
8. Explainability
9. Visualization
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Local imports
from src.data.loader import DataLoader, create_sample_dataset
from src.data.preprocessing import Preprocessor
from src.nlp.embeddings import EmbeddingModel
from src.nlp.similarity import SimilaritySearch, CrossEncoderReranker
from src.features.semantic import SemanticFeatures
from src.features.temporal import TemporalFeatures
from src.features.network import NetworkFeatures
from src.features.edge_features import EdgeFeatureExtractor
from src.graph.builder import GraphBuilder
from src.graph.community import CommunityDetector
from src.graph.visualization import GraphVisualizer
from src.models.baseline import BaselineModels
from src.models.gcn import GCNModel
from src.models.graphsage import GraphSAGEModel
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.experiments import ExperimentRunner
from src.utils.degradations import DegradationTracker

logger = logging.getLogger(__name__)


class CoordinationDetectionPipeline:
    """End-to-end pipeline for coordinated account detection."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get("experiments", {}).get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "figures"), exist_ok=True)

        # Initialize components
        self.data_loader = DataLoader(config)
        self.preprocessor = Preprocessor(config)
        self.embedding_model = EmbeddingModel(config)
        self.similarity_search = SimilaritySearch(config)
        self.semantic_features = SemanticFeatures(config)
        self.temporal_features = TemporalFeatures(config)
        self.network_features = NetworkFeatures(config)
        self.edge_feature_extractor = EdgeFeatureExtractor(config)
        self.graph_builder = GraphBuilder(config)
        self.community_detector = CommunityDetector(config)
        self.visualizer = GraphVisualizer(config)
        self.baseline_models = BaselineModels(config)
        self.gcn_model = GCNModel(config)
        self.graphsage_model = GraphSAGEModel(config)
        self.evaluation = EvaluationMetrics(config)
        self.experiment_runner = ExperimentRunner(config)

        # Runtime state shared across stages
        self.degradations: Optional[DegradationTracker] = None
        # Per-model (y_true, y_scores) on the canonical test split, used to
        # generate ROC/PR/confusion figures with a real caller (items 15/23).
        self._roc_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._best_model_preds: Optional[Tuple[str, np.ndarray, np.ndarray]] = None
        self._split: Optional[Dict[str, np.ndarray]] = None

    def run(self, use_sample: bool = False) -> Dict:
        """Run the full pipeline."""
        start_time = time.time()
        results = {}
        self.degradations = DegradationTracker()

        logger.info("=" * 60)
        logger.info("COORDINATED MISINFORMATION NETWORK DETECTION SYSTEM")
        logger.info("=" * 60)

        # Stage 1: Data loading
        logger.info("\n[Stage 1] Loading data...")
        df, data_source = self._load_data(use_sample)
        results["data_source"] = data_source
        results["data_stats"] = {
            "total_posts": len(df),
            "unique_accounts": df["accountid"].nunique(),
        }

        # Stage 2: Preprocessing
        logger.info("\n[Stage 2] Preprocessing...")
        df = self.preprocessor.process(df, degradations=self.degradations)

        # Stage 3: Compute embeddings
        logger.info("\n[Stage 3] Computing NLP embeddings...")
        df = self.embedding_model.encode_dataframe(df)
        # Stash for GNN node-feature augmentation (item 11)
        self._last_post_embeddings = (
            df, np.array(df["embedding"].tolist(), dtype=np.float32)
        )

        # Stage 4: Post-level candidate retrieval (item 4a)
        logger.info("\n[Stage 4] Post-level FAISS retrieval...")
        post_ids = df["postid"].astype(str).values
        account_ids_col = df["accountid"].astype(str).values
        post_embeddings = np.array(df["embedding"].tolist(), dtype=np.float32)

        k_neighbors = self.config.get("similarity", {}).get("k_neighbors", 20)
        post_pairs = self.similarity_search.find_post_pairs(
            post_embeddings, post_ids, account_ids_col, k=k_neighbors
        )

        # Attach texts to matched post pairs (needed for rerank + evidence).
        # post_text is normalized in place by the preprocessor (lowercased,
        # URLs stripped), so it IS the clean text.
        postid_to_text = dict(zip(
            df["postid"].astype(str).values,
            df["post_text"].fillna("").astype(str).values,
        ))
        if not post_pairs.empty:
            post_pairs["text_a"] = post_pairs["post_a"].astype(str).map(postid_to_text).fillna("")
            post_pairs["text_b"] = post_pairs["post_b"].astype(str).map(postid_to_text).fillna("")

        # Stage 4a-2: aggregate post candidates up to account pairs (item 4b)
        sim_cfg = self.config.get("similarity", {})
        top_n_per_pair = sim_cfg.get("max_post_pairs_per_account_pair", 3)
        account_candidates = self.semantic_features.aggregate_post_pair_candidates(
            post_pairs, top_n=top_n_per_pair
        )

        # Cross-encoder reranking on the ACTUAL matched post pairs (item 4c)
        crossencoder_scores = {}
        evidence_scores = {}
        fallback_pairs = set()
        use_cross_encoder = sim_cfg.get("use_cross_encoder_rerank", False)
        if use_cross_encoder and not account_candidates.empty:
            logger.info("\n[Stage 4b] Cross-encoder reranking matched post pairs...")
            reranker = CrossEncoderReranker(self.config, degradations=self.degradations)

            # Flatten account-pair candidates into post-pair rows for rerank
            flat_rows = []
            for _, row in account_candidates.iterrows():
                for pair in row["matched"]:
                    flat_rows.append({
                        "account_a": row["account_a"],
                        "account_b": row["account_b"],
                        "post_a": pair["post_a"],
                        "post_b": pair["post_b"],
                        "biencoder_similarity": pair["biencoder_similarity"],
                        "text_a": postid_to_text.get(str(pair["post_a"]), ""),
                        "text_b": postid_to_text.get(str(pair["post_b"]), ""),
                    })
            flat_df = pd.DataFrame(flat_rows)

            # Item 6: cross_encoder_rerank_k caps total rerank invocations
            # (pairs ranked by bi-encoder score). null/None = rerank all.
            rerank_k = sim_cfg.get("cross_encoder_rerank_k")
            max_rerank = sim_cfg.get("max_rerank_pairs")
            if max_rerank is not None and max_rerank > 0:
                rerank_k = (min(int(rerank_k), int(max_rerank))
                            if rerank_k else int(max_rerank))

            reranked = reranker.rerank(
                flat_df,
                texts_a=flat_df["text_a"].tolist(),
                texts_b=flat_df["text_b"].tolist(),
                score_column="biencoder_similarity",
                new_score_column="cross_encoder_score",
                top_k=rerank_k,
            )

            # Item 8c: pairs the reranker did NOT score (beyond the cap) are
            # bi-encoder-only = lower confidence; record the degradation.
            unscored = int(reranked["cross_encoder_score"].isna().sum())
            if unscored > 0:
                self.degradations.record(
                    "cross_encoder",
                    f"{unscored} post pairs beyond the rerank cap left with "
                    "bi-encoder-only scores (lower confidence)",
                )

            # Account-pair score = best cross-encoder score among its matched
            # post pairs; unscored pairs fall back to bi-encoder downstream.
            for (acc_a, acc_b), grp in reranked.groupby(["account_a", "account_b"]):
                pair_key = tuple(sorted([str(acc_a), str(acc_b)]))
                ce = grp["cross_encoder_score"].dropna()
                matched_pairs = []
                for _, r in grp.iterrows():
                    matched_pairs.append({
                        "post_a": r["post_a"], "post_b": r["post_b"],
                        "text_a": r["text_a"], "text_b": r["text_b"],
                        "biencoder_similarity": float(r["biencoder_similarity"]),
                        "cross_encoder_score": (
                            float(r["cross_encoder_score"])
                            if pd.notna(r["cross_encoder_score"]) else None
                        ),
                    })
                evidence_scores[pair_key] = matched_pairs
                if len(ce) > 0:
                    crossencoder_scores[pair_key] = float(ce.max())

        # Stage 5: account-pair evidence features
        logger.info("\n[Stage 5] Computing edge features...")
        use_ce_for_coordination = use_cross_encoder and bool(crossencoder_scores)
        if not account_candidates.empty:
            account_pairs = [
                (str(r["account_a"]), str(r["account_b"]))
                for _, r in account_candidates.iterrows()
            ]
            raw_biencoder_scores = {
                tuple(sorted([str(r["account_a"]), str(r["account_b"])])):
                    float(r["biencoder_similarity"])
                for _, r in account_candidates.iterrows()
            }
            merged = self.semantic_features.merge_semantic_scores(
                biencoder_scores=raw_biencoder_scores,
                crossencoder_scores=crossencoder_scores,
                use_cross_encoder=use_ce_for_coordination,
                degradations=self.degradations,
            )
            semantic_scores = {
                pair_key: entry["semantic_similarity"]
                for pair_key, entry in merged.items()
            }
        else:
            # Item 8d: NEVER present the top-volume fallback as a normal
            # "no semantic similarity found" result. Semantic scores are 0
            # and pairs are tagged semantic_fallback=True on the edge.
            self.degradations.record(
                "semantic_retrieval",
                "post-level retrieval found 0 candidate pairs; falling back "
                "to top-50-by-volume account pairs with semantic score 0 "
                "(tagged semantic_fallback=True)",
            )
            top_accounts = df["accountid"].value_counts().head(50).index.tolist()
            account_pairs = [
                (str(a), str(b)) for i, a in enumerate(top_accounts)
                for b in top_accounts[i+1:]
            ][:500]
            semantic_scores = {}
            fallback_pairs = {
                tuple(sorted(p)) for p in account_pairs
            }

        # Compute temporal features
        temporal_results = self.temporal_features.compute_batch_temporal_features(
            account_pairs, df
        )
        temporal_scores = {
            tuple(sorted([str(row["account_a"]), str(row["account_b"])])): row["temporal_score"]
            for _, row in temporal_results.iterrows()
        }

        # Compute all edge features
        edge_features_df = self.edge_feature_extractor.compute_batch_edge_features(
            account_pairs, df, semantic_scores, temporal_scores,
            fallback_pairs=fallback_pairs,
            evidence_scores=evidence_scores,
            degradations=self.degradations,
        )

        # Stage 6: Build graph
        logger.info("\n[Stage 6] Building coordination graph...")
        account_features_df = self.network_features.compute_account_features(df)

        # Map labels
        account_labels = df.groupby("accountid")["is_control"].all().reset_index()
        account_labels.columns = ["accountid", "is_control"]
        account_features_df = account_features_df.merge(
            account_labels, on="accountid", how="left", suffixes=("", "_label")
        )
        account_features_df["is_io"] = ~account_features_df.get(
            "is_control", account_features_df.get("is_control_label", pd.Series([True]*len(account_features_df)))
        ).fillna(True).astype(bool)

        # Ensure node_id column exists
        if "node_id" not in account_features_df.columns:
            unique_accounts = sorted(account_features_df["accountid"].unique())
            node_map = {a: i for i, a in enumerate(unique_accounts)}
            account_features_df["node_id"] = account_features_df["accountid"].map(node_map)

        G = self.graph_builder.build_networkx_graph(edge_features_df, account_features_df)
        graph_stats = self.graph_builder.get_graph_statistics(G)
        results["graph_stats"] = graph_stats

        # Stage 7: Community detection (Louvain runs BEFORE the GNN)
        logger.info("\n[Stage 7] Detecting communities...")
        communities = self.community_detector.detect_louvain_communities(
            G, degradations=self.degradations
        )
        # Stash for the Louvain baseline row in _run_baselines (item 21)
        self._last_communities = communities
        # Initial scoring WITHOUT GNN scores - this is the pre-GNN baseline
        # partition scoring (item 10 acceptance: the GNN must change the
        # final flagged groups, so we keep the pre-GNN state for the diff).
        pre_gnn_scored = self.community_detector.score_communities(
            communities, G, gnn_scores=None, degradations=self.degradations
        )

        # Canonical 70/15/15 account-level split (item 20): ONE split shared
        # by every model so model_comparison rows are actually comparable.
        # Seeds fixed for reproducibility.
        from sklearn.model_selection import train_test_split as _split
        n_accounts = len(account_features_df)
        idx_all = np.arange(n_accounts)
        y_all = account_features_df["is_io"].astype(int).values
        can_stratify = len(np.unique(y_all)) > 1
        train_idx, hold_idx = _split(
            idx_all, test_size=0.30, random_state=42, stratify=y_all if can_stratify else None
        )
        rel_test_frac = 0.5  # 0.30 split evenly -> 15/15
        val_idx, test_idx = _split(
            hold_idx, test_size=rel_test_frac, random_state=42,
            stratify=y_all[hold_idx] if can_stratify else None,
        )
        split_info = {
            "strategy": "canonical_account_level_70_15_15",
            "random_state": 42,
            "train": int(len(train_idx)), "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        }
        results["split"] = split_info
        self._split = {"train": train_idx, "val": val_idx, "test": test_idx}
        logger.info("Canonical split: %s", split_info)

        # Stage 8: Baseline models (threshold, RF, LR, Louvain) - all
        # evaluated on the canonical test split
        logger.info("\n[Stage 8] Training baseline models...")
        model_results = self._run_baselines(
            account_features_df, edge_features_df, G,
            test_indices=test_idx,
        )

        # Stage 9: GNN models (trained on the same canonical split)
        logger.info("\n[Stage 9] Training GNN models...")
        gnn_results, gnn_artifacts = self._run_gnn_models(
            G, account_features_df,
            train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        )
        model_results.update(gnn_results)

        # Item 10 (Option B): the TRAINED GNN's predicted P(IO) feeds the
        # community suspiciousness score, so the GNN directly influences
        # which groups get flagged.
        gnn_scores = gnn_artifacts.get("gnn_scores", {})
        if gnn_scores:
            logger.info("[Stage 9b] Re-scoring communities with GNN P(IO)...")
        scored_communities = self.community_detector.score_communities(
            communities, G, gnn_scores=gnn_scores, degradations=self.degradations
        )
        suspicious_groups = self.community_detector.get_suspicious_groups(
            scored_communities, top_k=10, min_size=3
        )

        # Acceptance evidence for item 10: diff pre-GNN vs post-GNN flagging
        pre_gnn_groups = self.community_detector.get_suspicious_groups(
            pre_gnn_scored, top_k=10, min_size=3
        )
        pre_ids = set(map(tuple, pre_gnn_groups["accounts"].tolist())) if not pre_gnn_groups.empty else set()
        post_ids = set(map(tuple, suspicious_groups["accounts"].tolist())) if not suspicious_groups.empty else set()
        results["gnn_integration_diff"] = {
            "pre_gnn_groups": sorted(str(g) for g in pre_ids),
            "post_gnn_groups": sorted(str(g) for g in post_ids),
            "changed": sorted(pre_ids) != sorted(post_ids),
        }
        logger.info("GNN integration changed flagging: %s",
                    results["gnn_integration_diff"]["changed"])

        # Stage 10: Evaluation
        logger.info("\n[Stage 10] Evaluation...")

        # Item 19 (learned-weight mode): fit the 6 signal weights by
        # logistic regression against ground-truth labels and re-score the
        # edges. Off by default (coordination.learn_weights: false) because
        # it consumes labels; the fixed config weights remain the default.
        weights_used = self.config.get("coordination", {}).get("weights")
        if self.config.get("coordination", {}).get("learn_weights", False):
            learned = self.edge_feature_extractor.learn_weights(
                account_features_df, edge_features_df
            )
            if learned is not None:
                edge_features_df["coordination_score"] = edge_features_df.apply(
                    lambda r: self.edge_feature_extractor.compute_coordination_score(
                        r.to_dict(), weights=learned
                    ), axis=1,
                )
                weights_used = learned
        results["coordination_weights_used"] = weights_used

        results["model_comparison"] = model_results
        # Item 20 evidence: stamp the canonical test-split size on every row
        # so model_comparison.csv itself proves all models were evaluated on
        # the same held-out set.
        n_test = int(len(self._split["test"])) if self._split else None
        if n_test is not None:
            for model_name, metrics in model_results.items():
                if isinstance(metrics, dict):
                    metrics["n_test"] = n_test
        comparison_df = self.experiment_runner.run_model_comparison(model_results)
        results["comparison_table"] = self.evaluation.format_results_table(comparison_df)

        # Stage 11: Ablation study
        if self.config.get("experiments", {}).get("ablation", False):
            logger.info("\n[Stage 11] Running feature ablation study...")
            ablation_df = self._run_ablation(
                account_features_df, edge_features_df, G
            )
            results["ablation_results"] = ablation_df.to_dict(orient="records")
            try:
                self.visualizer.plot_ablation_results(ablation_df)
            except Exception as e:
                logger.warning(f"Ablation visualization failed: {e}")

        # Stage 12: Explainability
        logger.info("\n[Stage 12] Generating explanations...")
        explanations = self.community_detector.generate_group_explanations(
            suspicious_groups, G, df
        )
        results["suspicious_groups"] = explanations

        # Item 22: GNNExplainer + permutation importance over the 6 signals
        try:
            explainability = self._run_explainability(
                G, account_features_df, edge_features_df,
                gnn_artifacts, train_idx, test_idx,
            )
            results["explainability"] = explainability
        except Exception as e:
            logger.warning(f"Explainability stage failed: {e}")
            if self.degradations is not None:
                self.degradations.record("explainability", f"failed: {e}")

        # Stage 13: Visualizations
        logger.info("\n[Stage 13] Creating visualizations...")
        self._create_visualizations(G, communities, df, edge_features_df,
                                    model_results, account_features_df, explanations)

        # Stage 14: Artifact persistence (item 16)
        logger.info("\n[Stage 14] Persisting artifacts...")
        try:
            self._persist_artifacts(G, account_features_df, df)
        except Exception as e:
            logger.warning(f"Artifact persistence failed: {e}")
            if self.degradations is not None:
                self.degradations.record("persistence", f"artifact persistence failed: {e}")

        # Item 8: every fallback/degradation surfaced in the results JSON
        results.update(self.degradations.to_dict())

        # Save results
        self._save_results(results)

        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline completed in {elapsed:.1f} seconds")
        logger.info(f"{'='*60}")

        return results

    def _load_data(self, use_sample: bool = False):
        """Load data, with SYNTHETIC data strictly opt-in (item 9).

        Returns (df, data_source) where data_source is "real" or "synthetic".

        Behavior:
        - use_sample=True: generate synthetic data explicitly (always allowed,
          the user asked for it) and tag results as synthetic.
        - use_sample=False and real data exists: load it; data_source="real".
        - use_sample=False and NO data: raise FileNotFoundError. The old
          behavior silently generated fake data and produced results that
          looked like real analysis output. Set allow_synthetic_fallback
          (train.py --allow-synthetic-fallback) to restore sample generation,
          which is then prominently tagged in the results.
        """
        if use_sample:
            logger.info("Creating sample dataset for testing...")
            create_sample_dataset(self.config["data"]["raw_dir"])

        # Prove synthetic-ness from what is ACTUALLY loaded, not from the
        # mere presence of the marker file (a stale marker from an old
        # synthetic generation must never mis-tag a real-data run - that
        # happened in practice when a test regenerated the marker).
        marker = os.path.join(
            self.config["data"]["raw_dir"], DataLoader.SYNTHETIC_MARKER_FILE
        )
        synthetic_marker_present = os.path.exists(marker)

        try:
            df = self.data_loader.load_campaign()
        except FileNotFoundError as e:
            allow_synthetic = self.config.get("data", {}).get(
                "allow_synthetic_fallback", False
            )
            if not allow_synthetic:
                logger.error(
                    "No data files found and synthetic fallback is DISABLED "
                    "(item 9). Re-run with --allow-synthetic-fallback to "
                    "generate a tagged synthetic dataset, or download real "
                    "data (see README)."
                )
                raise
            logger.warning(
                "No data files found - SYNTHETIC fallback engaged "
                "(allow_synthetic_fallback=true). Results will be tagged "
                "data_source=synthetic."
            )
            create_sample_dataset(self.config["data"]["raw_dir"])
            df = self.data_loader.load_campaign()
            synthetic_marker_present = True

        # Provenance: the synthetic sample is written to <raw_dir>/sample_campaign.csv
        # (create_sample_dataset). If the files actually loaded include that
        # sample, the run is synthetic. A marker file alone (stale, or left
        # over in a directory holding real data) is NOT proof of synthetic
        # data - record a degradation instead so the stale marker is visible.
        loaded_files = [os.path.normpath(f) for f in self.data_loader.discover_files()]
        sample_file = os.path.normpath(
            os.path.join(self.config["data"]["raw_dir"], "sample_campaign.csv")
        )
        loaded_synthetic_sample = sample_file in loaded_files
        data_source = "synthetic" if loaded_synthetic_sample else "real"
        if synthetic_marker_present and not loaded_synthetic_sample:
            self.degradations.record(
                component="data_loader",
                reason=(
                    "Stale SYNTHETIC_DATA.txt marker present in the data "
                    "directory while real data files were loaded; "
                    "data_source remains 'real'"
                ),
                detail=marker,
            )
            logger.warning(
                "Stale synthetic-data marker found alongside real data - "
                "ignoring it for provenance (degradation recorded)"
            )
        if data_source == "synthetic":
            logger.warning(
                "RUNNING ON SYNTHETIC DATA - results are not real analysis "
                "output and will be tagged data_source=synthetic"
            )

        info = self.data_loader.get_dataset_info(df)
        logger.info(f"Dataset info: {json.dumps(info, indent=2, default=str)}")
        return df, data_source

    def _run_baselines(self, account_df: pd.DataFrame,
                       edge_df: pd.DataFrame,
                       G: "nx.Graph",
                       test_indices: Optional[np.ndarray] = None,
                       ) -> Dict[str, Dict[str, float]]:
        """Run all baseline models on the CANONICAL test split (item 20).

        Every model - threshold, RF, LR, Louvain - is evaluated on the same
        held-out test indices so model_comparison.csv rows are comparable.
        Previously the threshold baseline used the full account set, RF/LR an
        80/20 split, and GNNs a different random mask.
        """
        results = {}

        # Prepare features for account-level classification
        feature_cols = [
            "post_count", "avg_follower_count", "avg_following_count",
            "repost_ratio", "avg_hashtag_count", "avg_url_count",
            "avg_mention_count", "posts_per_day"
        ]
        available_cols = [c for c in feature_cols if c in account_df.columns]

        if not available_cols or "is_io" not in account_df.columns:
            logger.warning("Insufficient features for baseline models")
            return results

        X = account_df[available_cols].fillna(0).values
        y = account_df["is_io"].astype(int).values

        if len(np.unique(y)) < 2:
            logger.warning("Only one class present, skipping baselines")
            return results

        # Canonical split indices come from run(); fall back to a local split
        if test_indices is None or len(test_indices) == 0:
            from sklearn.model_selection import train_test_split
            all_idx = np.arange(len(y))
            tr_idx, test_indices = train_test_split(
                all_idx, test_size=0.15, random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None,
            )
            train_indices = tr_idx
        else:
            test_set = set(test_indices.tolist())
            train_indices = np.array(
                [i for i in range(len(y)) if i not in test_set]
            )

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Threshold baseline.
        # INPUT SIGNAL (item 2 fix): the mean per-account coordination_score
        # over all incident edges in the account-pair evidence table
        # (edge_df['coordination_score'] - the same weighted 6-signal score
        # used to build the graph). Accounts with no scored pairs get 0.0.
        # Previously this fed normalized POST COUNT (X[:, 0]), which made the
        # baseline measure posting volume, not coordination evidence.
        coord_by_account = pd.Series(0.0, index=account_df["accountid"].values)
        if (edge_df is not None and not edge_df.empty
                and "coordination_score" in edge_df.columns):
            # Mean coordination score over all edges incident to the account
            # (each edge contributes to both endpoints).
            both_dirs = pd.concat([
                edge_df[["account_a", "coordination_score"]].rename(
                    columns={"account_a": "account"}),
                edge_df[["account_b", "coordination_score"]].rename(
                    columns={"account_b": "account"}),
            ])
            per_account_mean = both_dirs.groupby("account")["coordination_score"].mean()
            coord_by_account = per_account_mean.reindex(
                account_df["accountid"].values
            ).fillna(0.0)
        threshold_scores = coord_by_account.values

        # Threshold tuned on the TRAIN split only (no test leakage)
        best_threshold = self.baseline_models.optimize_threshold(
            threshold_scores[train_indices], y_train, metric="f1"
        )
        threshold_pred = self.baseline_models.threshold_baseline(
            threshold_scores, threshold=best_threshold
        )
        results["threshold_baseline"] = self.baseline_models.evaluate_predictions(
            y_test, threshold_pred[test_indices], threshold_scores[test_indices]
        )

        # Random Forest
        rf_pred, rf_prob = self.baseline_models.train_random_forest(X_train, y_train, X_test)
        results["random_forest"] = self.baseline_models.evaluate_predictions(
            y_test, rf_pred, rf_prob
        )
        self._roc_data["random_forest"] = (y_test, np.asarray(rf_prob))
        self._best_model_preds = ("random_forest", y_test, np.asarray(rf_pred))

        # Logistic Regression
        lr_pred, lr_prob = self.baseline_models.train_logistic_regression(X_train, y_train, X_test)
        results["logistic_regression"] = self.baseline_models.evaluate_predictions(
            y_test, lr_pred, lr_prob
        )
        self._roc_data["logistic_regression"] = (y_test, np.asarray(lr_prob))

        # Louvain as an evaluated baseline (item 21): use community membership
        # as the prediction. A community is predicted IO if its mean edge
        # coordination score is >= the edge threshold (unsupervised, no
        # ground-truth labels involved - previously io_ratio leakage).
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            pred = np.zeros(len(y), dtype=int)
            for comm_id, members in self._last_communities.items():
                node_set = set(members)
                idxs = [i for i, acc in enumerate(account_df["accountid"].values)
                        if acc in node_set]
                if not idxs:
                    continue
                member_edges = [
                    d.get("coordination_score", 0)
                    for u, v2, d in G.edges(data=True)
                    if u in node_set
                ]
                comm_score = float(np.mean(member_edges)) if member_edges else 0.0
                if comm_score >= self.graph_builder.edge_threshold:
                    for i in idxs:
                        pred[i] = 1
            results["louvain"] = {
                "accuracy": float((pred[test_indices] == y_test).mean()),
                "precision": precision_score(y_test, pred[test_indices], zero_division=0),
                "recall": recall_score(y_test, pred[test_indices], zero_division=0),
                "f1": f1_score(y_test, pred[test_indices], zero_division=0),
            }
        except Exception as e:
            logger.warning(f"Louvain baseline evaluation failed: {e}")

        # Feature importance
        for model_name in ["random_forest", "logistic_regression"]:
            importance = self.baseline_models.get_feature_importance(model_name, available_cols)
            if importance is not None:
                logger.info(f"\n{model_name} feature importance:\n{importance.to_string()}")

        return results

    def _run_gnn_models(self, G: "nx.Graph",
                        account_df: pd.DataFrame,
                        train_idx: Optional[np.ndarray] = None,
                        val_idx: Optional[np.ndarray] = None,
                        test_idx: Optional[np.ndarray] = None,
                        ) -> Tuple[Dict[str, Dict[str, float]], Dict]:
        """Run GCN and GraphSAGE models.

        Item 11: the node feature matrix now includes a dimensionality-
        reduced (PCA) account content embedding alongside the behavioral
        stats, so the GNN sees semantic content at the node level.

        Item 20: train/val/test masks come from the canonical account-level
        split indices (aligned through account_df rows), not a fresh random
        permutation.

        Returns (metrics_per_model, artifacts) where artifacts contains the
        trained models, per-node P(IO) scores (used for item 10 Option B),
        and the PyG data object.
        """
        results: Dict[str, Dict[str, float]] = {}
        artifacts: Dict = {"gnn_scores": {}, "models": {}, "pyg_data": None}

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            logger.warning("Empty graph, skipping GNN models")
            return results, artifacts

        # Behavioral node features
        feature_cols = self.network_features.select_node_features()
        available_cols = [c for c in feature_cols if c in account_df.columns]

        if not available_cols or "is_io" not in account_df.columns:
            logger.warning("Insufficient features for GNN models")
            return results, artifacts

        nodes = sorted(G.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}

        # Item 11: PCA-reduced content embedding per account, computed from
        # the post embeddings stashed by run() (self._last_post_embeddings).
        content_pca = None
        post_emb = getattr(self, "_last_post_embeddings", None)
        if post_emb is not None:
            try:
                posts_df, post_emb_matrix = post_emb
                from sklearn.decomposition import PCA
                emb_by_account = {}
                acc_arr = posts_df["accountid"].astype(str).values
                for i, acc in enumerate(acc_arr):
                    v = emb_by_account.get(acc)
                    if v is None:
                        emb_by_account[acc] = post_emb_matrix[i]
                    else:
                        emb_by_account[acc] = v + post_emb_matrix[i]
                for acc in emb_by_account:
                    emb_by_account[acc] = emb_by_account[acc] / np.linalg.norm(
                        emb_by_account[acc]
                    )
                emb_dim = next(iter(emb_by_account.values())).shape[0]
                n_components = int(min(16, emb_dim))
                pca = PCA(n_components=n_components, random_state=42)
                sample = np.array(list(emb_by_account.values()))
                pca.fit(sample)
                content_pca = (pca, emb_by_account, n_components)
                logger.info(
                    "Node features augmented with %d-dim PCA content embedding "
                    "(item 11)", n_components,
                )
            except Exception as e:
                logger.warning(f"Content-embedding augmentation skipped: {e}")
                content_pca = None

        n_behavioral = len(available_cols)
        n_content = content_pca[2] if content_pca else 0
        node_features = []
        node_labels = []

        for node in nodes:
            row = account_df[account_df["accountid"] == node]
            if len(row) > 0:
                feats = list(row[available_cols].values[0])
                node_labels.append(int(row["is_io"].values[0]))
            else:
                feats = [0.0] * n_behavioral
                node_labels.append(0)
            if content_pca is not None:
                pca, emb_by_account, n_comp = content_pca
                acc_emb = emb_by_account.get(str(node))
                if acc_emb is None:
                    feats.extend([0.0] * n_comp)
                else:
                    feats.extend(pca.transform(acc_emb.reshape(1, -1))[0].tolist())
            node_features.append(feats)

        node_features = np.array(node_features, dtype=np.float32)
        node_labels = np.array(node_labels, dtype=np.int64)

        # Normalize behavioral columns; content PCA dims are already centered
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        node_features[:, :n_behavioral] = scaler.fit_transform(
            node_features[:, :n_behavioral]
        )

        # Build PyG data
        pyg_data = self.graph_builder.build_pyg_data(G, node_features, node_labels)
        artifacts["pyg_data"] = pyg_data
        artifacts["node_map"] = node_map

        # Canonical split -> graph-node masks (item 20). account_df row i has
        # accountid = nodes_at[i]; map it into the graph node index space.
        num_nodes = pyg_data.num_nodes
        account_ids_arr = account_df["accountid"].astype(str).values
        account_pos = {acc: i for i, acc in enumerate(account_ids_arr)}

        def _mask(idx: Optional[np.ndarray]) -> torch.Tensor:
            m = torch.zeros(num_nodes, dtype=torch.bool)
            if idx is None:
                return m
            for i in idx:
                node_idx = node_map.get(account_ids_arr[i]) if i < len(account_ids_arr) else None
                if node_idx is not None:
                    m[node_idx] = True
            return m

        train_mask = _mask(train_idx)
        val_mask = _mask(val_idx)
        test_mask = _mask(test_idx)

        # Guard: any graph node not covered by the split goes to train
        uncovered = ~(train_mask | val_mask | test_mask)
        if uncovered.any():
            train_mask |= uncovered

        out_channels = len(np.unique(node_labels))

        # GCN
        try:
            logger.info("Training GCN...")
            gcn_metrics, gcn_probs = self._train_and_eval_gnn(
                self.gcn_model, pyg_data, train_mask, val_mask, test_mask,
                in_channels=pyg_data.x.shape[1], out_channels=out_channels,
            )
            results["gcn"] = gcn_metrics
            artifacts["models"]["gcn"] = self.gcn_model.model
            # Bundle for checkpoint persistence (item 16): enough to
            # reproduce predictions from the .pt file alone.
            self._last_gnn_data = getattr(self, "_last_gnn_data", {}) or {}
            self._last_gnn_data["gcn"] = {
                "x": pyg_data.x.cpu(),
                "edge_index": pyg_data.edge_index.cpu(),
                "test_mask": test_mask.cpu(),
                "in_channels": int(pyg_data.x.shape[1]),
            }
            # Per-node P(IO) on the FULL graph (used for item 10 Option B)
            _, full_probs = self.gcn_model.predict(pyg_data)
            for node, i in node_map.items():
                artifacts["gnn_scores"][str(node)] = float(full_probs[i, 1])
            test_nodes = [n for n in nodes if test_mask[node_map[n]]]
            if test_nodes:
                self._roc_data["gcn"] = (
                    node_labels[[node_map[n] for n in test_nodes]],
                    full_probs[[node_map[n] for n in test_nodes], 1],
                )
        except Exception as e:
            logger.error(f"GCN training failed: {e}")

        # GraphSAGE
        try:
            logger.info("Training GraphSAGE...")
            sage_metrics, sage_probs = self._train_and_eval_gnn(
                self.graphsage_model, pyg_data, train_mask, val_mask, test_mask,
                in_channels=pyg_data.x.shape[1], out_channels=out_channels,
            )
            results["graphsage"] = sage_metrics
            artifacts["models"]["graphsage"] = self.graphsage_model.model
            self._last_gnn_data = getattr(self, "_last_gnn_data", {}) or {}
            self._last_gnn_data["graphsage"] = {
                "x": pyg_data.x.cpu(),
                "edge_index": pyg_data.edge_index.cpu(),
                "test_mask": test_mask.cpu(),
                "in_channels": int(pyg_data.x.shape[1]),
            }
        except Exception as e:
            logger.error(f"GraphSAGE training failed: {e}")

        return results, artifacts

    def _train_and_eval_gnn(self, model, data: Data,
                            train_mask: torch.Tensor,
                            val_mask: torch.Tensor,
                            test_mask: torch.Tensor,
                            in_channels: int,
                            out_channels: int = 2) -> Tuple[Dict[str, float], np.ndarray]:
        """Train and evaluate a GNN model; return (metrics, test probabilities)."""
        model.build_model(in_channels, out_channels=out_channels)
        history = model.train(data, train_mask, val_mask)
        metrics = model.evaluate(data, test_mask)
        _, probs = model.predict(data)
        test_positions = test_mask.cpu().numpy()
        return metrics, probs[test_positions]

    def _run_explainability(self, G, account_df, edge_features_df,
                            gnn_artifacts, train_idx, test_idx) -> Dict:
        """GNNExplainer over the trained GCN + permutation importance over
        the 6 coordination signals (item 22 - previously documented but not
        implemented).
        """
        out: Dict = {}
        explain_cfg = self.config.get("explainability", {})

        # ---- GNNExplainer (torch_geometric.explain) ----
        if explain_cfg.get("use_gnnexplainer", False):
            pyg_data = gnn_artifacts.get("pyg_data")
            gcn = gnn_artifacts.get("models", {}).get("gcn")
            if pyg_data is not None and gcn is not None:
                try:
                    from torch_geometric.explain import Explainer, GNNExplainer
                    explainer = Explainer(
                        model=gcn,
                        algorithm=GNNExplainer(epochs=50),
                        explanation_type="model",
                        node_mask_type="attributes",
                        edge_mask_type="object",
                        model_config=dict(
                            mode="multiclass_classification",
                            task_level="node",
                            return_type="raw",
                        ),
                    )
                    # Explain a sample of nodes (cost control)
                    rng = np.random.default_rng(42)
                    sample_nodes = rng.choice(
                        pyg_data.num_nodes,
                        size=min(20, pyg_data.num_nodes),
                        replace=False,
                    )
                    feats = pyg_data.x.shape[1]
                    feature_imp = np.zeros(feats)
                    for ni in sample_nodes:
                        exp = explainer(pyg_data.x, pyg_data.edge_index,
                                        index=int(ni))
                        if getattr(exp, "node_mask", None) is not None:
                            feature_imp += (
                                exp.node_mask.numpy().mean(axis=0)
                                if exp.node_mask.ndim > 1
                                else exp.node_mask.numpy()
                            )
                    feature_imp /= max(len(sample_nodes), 1)
                    out["gnnexplainer_feature_importance"] = {
                        f"feature_{i}": float(v) for i, v in enumerate(feature_imp)
                    }
                    logger.info("GNNExplainer completed over %d sample nodes",
                                len(sample_nodes))
                except Exception as e:
                    logger.warning(f"GNNExplainer failed: {e}")
                    if self.degradations is not None:
                        self.degradations.record("gnnexplainer", f"failed: {e}")

        # ---- Permutation importance over the 6 coordination signals ----
        n_runs = int(explain_cfg.get("permutation_runs", 10))
        signal_cols = {
            "semantic": "semantic_similarity",
            "temporal": "temporal_score",
            "url": "shared_url_score",
            "hashtag": "shared_hashtag_score",
            "mention": "shared_mention_score",
            "repost": "repost_score",
        }
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import f1_score

            # Same per-account signal matrix as the ablation study
            all_cols = list(signal_cols.values())
            stacked = pd.concat([
                edge_features_df[["account_a"] + all_cols].rename(
                    columns={"account_a": "account"}),
                edge_features_df[["account_b"] + all_cols].rename(
                    columns={"account_b": "account"}),
            ], ignore_index=True)
            per_account = stacked.groupby("account")[all_cols].mean()
            X = (per_account.reindex(account_df["accountid"].values)
                 .reindex(columns=all_cols).fillna(0.0).values)
            y = account_df["is_io"].astype(int).values

            if test_idx is None or len(test_idx) == 0:
                from sklearn.model_selection import train_test_split
                idx = np.arange(len(y))
                tr, test_idx = train_test_split(
                    idx, test_size=0.15, random_state=42, stratify=y
                )
                train_idx = tr
            train_idx = train_idx if train_idx is not None else np.array(
                [i for i in range(len(y)) if i not in set(test_idx.tolist())]
            )

            if len(np.unique(y)) > 1 and len(test_idx) > 0:
                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=20, random_state=42,
                    class_weight="balanced", n_jobs=-1,
                )
                rf.fit(X[train_idx], y[train_idx])
                base = f1_score(y[test_idx], rf.predict(X[test_idx]),
                                zero_division=0)
                rng = np.random.default_rng(42)
                perm_imp = {}
                for name, col in signal_cols.items():
                    drops = []
                    col_pos = all_cols.index(col)
                    for _ in range(max(n_runs, 1)):
                        X_perm = X[test_idx].copy()
                        rng.shuffle(X_perm[:, col_pos])
                        drops.append(base - f1_score(
                            y[test_idx], rf.predict(X_perm), zero_division=0
                        ))
                    perm_imp[name] = float(np.mean(drops))
                out["permutation_importance"] = {
                    "baseline_f1": float(base),
                    "per_signal_f1_drop": perm_imp,
                    "n_runs": max(n_runs, 1),
                }
                logger.info("Permutation importance: %s",
                            {k: round(v, 4) for k, v in perm_imp.items()})
        except Exception as e:
            logger.warning(f"Permutation importance failed: {e}")

        return out

    def _persist_artifacts(self, G, account_df, posts_df) -> Dict[str, str]:
        """Persist models, graph, and embeddings (item 16).

        - results/models/gcn.pt, graphsage.pt  (state_dicts)
        - results/artifacts/coordination_graph.pkl
        - results/artifacts/post_embeddings.npy + posts_index.csv
        - results/artifacts/account_embeddings.csv
        """
        models_dir = os.path.join(self.output_dir, "models")
        artifacts_dir = os.path.join(self.output_dir, "artifacts")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(artifacts_dir, exist_ok=True)
        paths = {}

        # Models (state_dicts, loadable with Model.load_state_dict).
        # Bundled with the node feature matrix, edge index, and test mask so
        # predictions can be reproduced from the checkpoint alone (item 16
        # acceptance: "loads them back and reproduces predictions").
        data_bundles = getattr(self, "_last_gnn_data", {}) or {}
        for name, wrapper in (("gcn", self.gcn_model),
                              ("graphsage", self.graphsage_model)):
            if wrapper.model is not None:
                p = os.path.join(models_dir, f"{name}.pt")
                bundle = {"model_state_dict": wrapper.model.state_dict()}
                bundle.update(data_bundles.get(name, {}))
                torch.save(bundle, p)
                paths[f"{name}_checkpoint"] = p

        # Graph
        p = os.path.join(artifacts_dir, "coordination_graph.pkl")
        self.graph_builder.save_graph(G, p)
        paths["graph"] = p

        # Post embeddings
        post_emb = getattr(self, "_last_post_embeddings", None)
        if post_emb is not None:
            posts_df, emb_matrix = post_emb
            p = os.path.join(artifacts_dir, "post_embeddings.npy")
            np.save(p, emb_matrix)
            paths["post_embeddings"] = p
            p = os.path.join(artifacts_dir, "posts_index.csv")
            posts_df[["postid", "accountid", "post_time"]].to_csv(p, index=False)
            paths["posts_index"] = p

        # Account-level mean embeddings
        if post_emb is not None:
            posts_df, emb_matrix = post_emb
            acc_ids = posts_df["accountid"].astype(str).values
            uniq = np.unique(acc_ids)
            rows = []
            for acc in uniq:
                rows.append(emb_matrix[acc_ids == acc].mean(axis=0))
            acc_emb_df = pd.DataFrame(
                rows, index=uniq,
                columns=[f"emb_{i}" for i in range(emb_matrix.shape[1])],
            ).reset_index().rename(columns={"index": "accountid"})
            p = os.path.join(artifacts_dir, "account_embeddings.csv")
            acc_emb_df.to_csv(p, index=False)
            paths["account_embeddings"] = p

        logger.info("Persisted artifacts: %s", list(paths.values()))
        return paths

    def _run_ablation(self, account_df: pd.DataFrame,
                       edge_features_df: pd.DataFrame,
                       G: "nx.Graph") -> pd.DataFrame:
        """Run feature ablation study (item 3 fix).

        The ablated model must actually CONSUME the ablated features. The
        previous implementation re-trained a node-feature-only Random Forest
        while dropping edge-feature columns, so every ablation row was
        identical (all coordination signals live on edges, not nodes).

        Design: build a per-account coordination-signal matrix by averaging
        the six edge signals over all edges incident to each account, then
        train/evaluate a Random Forest on THAT representation. Ablating a
        signal = dropping its column(s) from the matrix before retraining.
        One fixed stratified split (seed 42) is reused for every row so the
        comparison is like-for-like.
        """
        signal_column_map = {
            "semantic": ["semantic_similarity"],
            "temporal": ["temporal_score"],
            "url": ["shared_url_score"],
            "hashtag": ["shared_hashtag_score"],
            "mention": ["shared_mention_score"],
            "repost": ["repost_score"],
        }
        categories = self.config.get("experiments", {}).get(
            "ablation_features", list(signal_column_map.keys())
        )

        all_signal_cols = [c for cols in signal_column_map.values() for c in cols]

        if (edge_features_df is None or edge_features_df.empty
                or "is_io" not in account_df.columns
                or not all(c in edge_features_df.columns for c in all_signal_cols)):
            logger.warning("Insufficient edge features for ablation study")
            return pd.DataFrame()

        # Per-account mean of each signal over incident edges (both endpoints)
        cols = ["account_a", "account_b"] + all_signal_cols
        stacked = pd.concat([
            edge_features_df[cols].rename(columns={"account_a": "account"}),
            edge_features_df[cols].rename(columns={"account_b": "account"}),
        ], ignore_index=True)
        per_account = stacked.groupby("account")[all_signal_cols].mean()

        X_full = (
            per_account.reindex(account_df["accountid"].values)
            .reindex(columns=all_signal_cols)
            .fillna(0.0)
            .values
        )
        y = account_df["is_io"].astype(int).values

        if len(np.unique(y)) < 2:
            logger.warning("Only one class present, skipping ablation")
            return pd.DataFrame()

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, accuracy_score

        # One canonical split reused by every ablation row
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.2, random_state=42, stratify=y
        )
        y_train, y_test = y[train_idx], y[test_idx]

        def _eval(X: np.ndarray) -> Dict[str, float]:
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=42,
                class_weight="balanced", n_jobs=-1
            )
            rf.fit(X[train_idx], y_train)
            pred = rf.predict(X[test_idx])
            return {
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            }

        results = []

        # Full signal set
        results.append({"ablation": "none (full)", **_eval(X_full)})

        # Ablate one signal at a time from the matrix the model consumes
        for category in categories:
            keep_cols = [c for c in all_signal_cols
                         if c not in signal_column_map.get(category, [])]
            X_ablated = (
                per_account.reindex(account_df["accountid"].values)
                .reindex(columns=keep_cols)
                .fillna(0.0)
                .values
            )
            results.append({"ablation": f"no {category}", **_eval(X_ablated)})

        results_df = pd.DataFrame(results)

        # Save results
        output_path = os.path.join(self.output_dir, "ablation_results.csv")
        results_df.to_csv(output_path, index=False)
        logger.info(f"Ablation results saved to {output_path}")
        logger.info(f"\nAblation Study Results:\n{results_df.to_string(index=False)}")

        # Ablation figure (items 15/23: wire the previously-dead plotter and
        # match the ablation_study.png filename documented in README §13)
        try:
            self.visualizer.plot_ablation_results(results_df)
        except Exception as e:
            logger.warning(f"Ablation visualization failed: {e}")

        return results_df

    def _create_visualizations(self, G, communities, df, edge_features_df,
                               model_results, account_df, explanations):
        """Create all visualizations, including the ROC/PR/confusion-matrix
        figures the README documents (items 15/23: dead plotting code now has
        real callers fed by the canonical test split)."""
        try:
            # Network graph
            self.visualizer.plot_network(
                G, title="Coordination Network",
                filename="network_graph"
            )
        except Exception as e:
            logger.warning(f"Network visualization failed: {e}")

        # ROC curves for every model with saved test-split scores (item 15)
        if self._roc_data:
            try:
                # Models share the canonical test set when lengths match
                ref_name = next(iter(self._roc_data))
                ref_y, _ = self._roc_data[ref_name]
                aligned = {
                    name: scores for name, (yt, scores) in self._roc_data.items()
                    if len(yt) == len(ref_y)
                }
                if aligned:
                    self.visualizer.plot_roc_curve(ref_y, aligned,
                                                   filename="roc_curve")
                    self.visualizer.plot_pr_curve(ref_y, aligned,
                                                  filename="pr_curve")
            except Exception as e:
                logger.warning(f"ROC/PR visualization failed: {e}")

        # Confusion matrix for the strongest baseline (RF)
        if self._best_model_preds is not None:
            try:
                name, y_true, y_pred = self._best_model_preds
                self.visualizer.plot_confusion_matrix(
                    y_true, y_pred, title=f"Confusion Matrix ({name})",
                    filename="confusion_matrix",
                )
            except Exception as e:
                logger.warning(f"Confusion matrix visualization failed: {e}")

        # Feature distribution split by IO/control (wired dead code, item 15)
        try:
            if (not edge_features_df.empty and "is_io" in account_df.columns
                    and "semantic_similarity" in edge_features_df.columns):
                sig_df = edge_features_df.merge(
                    account_df[["accountid", "is_io"]],
                    left_on="account_a", right_on="accountid", how="left",
                )
                self.visualizer.plot_feature_distribution(
                    sig_df, "semantic_similarity", group_col="is_io",
                    title="Semantic similarity by account class",
                    filename="dist_semantic_similarity",
                )
        except Exception as e:
            logger.warning(f"Feature distribution visualization failed: {e}")

        try:
            # Community detection
            self.visualizer.plot_community_detection(
                G, communities, filename="communities"
            )
        except Exception as e:
            logger.warning(f"Community visualization failed: {e}")

        try:
            # Coordination score distribution
            if not edge_features_df.empty:
                self.visualizer.plot_coordination_score_distribution(
                    edge_features_df, filename="coordination_distribution"
                )
        except Exception as e:
            logger.warning(f"Coordination distribution plot failed: {e}")

        try:
            # Model comparison
            if model_results:
                comparison_df = pd.DataFrame([
                    {"model": k, **v} for k, v in model_results.items()
                ])
                self.visualizer.plot_model_comparison(
                    comparison_df, filename="model_comparison"
                )
        except Exception as e:
            logger.warning(f"Model comparison plot failed: {e}")

        try:
            # Group explanations
            if explanations:
                for i, exp in enumerate(explanations[:3]):
                    self.visualizer.plot_explanation_dashboard(
                        exp, df, filename=f"group_explanation_{i}"
                    )
        except Exception as e:
            logger.warning(f"Explanation dashboard failed: {e}")

    def _save_results(self, results: Dict):
        """Save all results to disk."""
        # Save results JSON (without non-serializable items)
        serializable = {}
        for key, value in results.items():
            if key == "suspicious_groups":
                serializable[key] = value  # Already serializable
            elif isinstance(value, dict):
                serializable[key] = {
                    k: v for k, v in value.items()
                    if isinstance(v, (int, float, str, list, dict, type(None)))
                }
            else:
                try:
                    json.dumps(value)
                    serializable[key] = value
                except (TypeError, ValueError):
                    pass

        output_path = os.path.join(self.output_dir, "pipeline_results.json")
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)

        if "comparison_table" in results:
            logger.info(f"\n{results['comparison_table']}")

        if "suspicious_groups" in results:
            logger.info(f"\nSuspicious Groups Found: {len(results['suspicious_groups'])}")
            for group in results["suspicious_groups"][:5]:
                logger.info(f"\n{group['summary']}")
