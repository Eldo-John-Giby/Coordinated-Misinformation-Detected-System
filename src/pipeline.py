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

    def run(self, use_sample: bool = False) -> Dict:
        """Run the full pipeline."""
        start_time = time.time()
        results = {}

        logger.info("=" * 60)
        logger.info("COORDINATED MISINFORMATION NETWORK DETECTION SYSTEM")
        logger.info("=" * 60)

        # Stage 1: Data loading
        logger.info("\n[Stage 1] Loading data...")
        df = self._load_data(use_sample)
        results["data_stats"] = {
            "total_posts": len(df),
            "unique_accounts": df["accountid"].nunique(),
        }

        # Stage 2: Preprocessing
        logger.info("\n[Stage 2] Preprocessing...")
        df = self.preprocessor.process(df)

        # Stage 3: Compute embeddings
        logger.info("\n[Stage 3] Computing NLP embeddings...")
        df = self.embedding_model.encode_dataframe(df)

        # Stage 4: Find similar post pairs
        logger.info("\n[Stage 4] Finding similar posts...")
        account_embeddings = self.embedding_model.compute_account_embeddings(df)
        embeddings_matrix = np.array([
            account_embeddings.iloc[i, 1:].values.astype(np.float32)
            for i in range(len(account_embeddings))
        ])
        account_ids = account_embeddings["accountid"].values

        self.similarity_search.build_index(embeddings_matrix)
        account_similarities = self.similarity_search.find_account_similarities(
            embeddings_matrix, account_ids, k=self.config.get("similarity", {}).get("k_neighbors", 20)
        )

        # Cross-encoder reranking (optional)
        crossencoder_scores = {}
        use_cross_encoder = self.config.get("similarity", {}).get(
            "use_cross_encoder_rerank", False
        )
        if use_cross_encoder and not account_similarities.empty:
            logger.info("\n[Stage 4b] Cross-encoder reranking...")
            reranker = CrossEncoderReranker(self.config)

            # Build text lookup: account_id -> representative post text
            # Use the first (non-empty) post text per account
            account_texts = {}
            for acc_id, group in df.groupby("accountid"):
                non_empty = group["clean_text"][
                    group["clean_text"].astype(str).str.strip().str.len() > 0
                ]
                if len(non_empty) > 0:
                    account_texts[acc_id] = non_empty.iloc[0]
                else:
                    account_texts[acc_id] = ""

            # Prepare texts for each candidate pair
            texts_a = [
                account_texts.get(row["account_a"], "")
                for _, row in account_similarities.iterrows()
            ]
            texts_b = [
                account_texts.get(row["account_b"], "")
                for _, row in account_similarities.iterrows()
            ]

            reranked_df = reranker.rerank(
                account_similarities,
                texts_a=texts_a,
                texts_b=texts_b,
                score_column="similarity",
                new_score_column="cross_encoder_score",
            )
            # Build cross-encoder scores dict
            crossencoder_scores = {}
            for _, row in reranked_df.iterrows():
                pair_key = tuple(sorted([row["account_a"], row["account_b"]]))
                crossencoder_scores[pair_key] = row["cross_encoder_score"]

        # Stage 5: Compute edge features
        logger.info("\n[Stage 5] Computing edge features...")
        use_ce_for_coordination = use_cross_encoder and bool(crossencoder_scores)
        if not account_similarities.empty:
            account_pairs = list(zip(
                account_similarities["account_a"],
                account_similarities["account_b"]
            ))
            raw_biencoder_scores = {
                tuple(sorted([row["account_a"], row["account_b"]])): row["similarity"]
                for _, row in account_similarities.iterrows()
            }
            # Merge bi-encoder + cross-encoder scores; the active semantic_similarity
            # key will be whichever the config selects for the coordination formula
            merged = self.semantic_features.merge_semantic_scores(
                biencoder_scores=raw_biencoder_scores,
                crossencoder_scores=crossencoder_scores,
                use_cross_encoder=use_ce_for_coordination,
            )
            # Extract the active score for edge feature computation
            semantic_scores = {
                pair_key: entry["semantic_similarity"]
                for pair_key, entry in merged.items()
            }
        else:
            # If no similar pairs, use top accounts by post count
            top_accounts = df["accountid"].value_counts().head(50).index.tolist()
            account_pairs = [
                (a, b) for i, a in enumerate(top_accounts)
                for b in top_accounts[i+1:]
            ][:500]
            semantic_scores = {}

        # Compute temporal features
        temporal_results = self.temporal_features.compute_batch_temporal_features(
            account_pairs, df
        )
        temporal_scores = {
            tuple(sorted([row["account_a"], row["account_b"]])): row["temporal_score"]
            for _, row in temporal_results.iterrows()
        }

        # Compute all edge features
        edge_features_df = self.edge_feature_extractor.compute_batch_edge_features(
            account_pairs, df, semantic_scores, temporal_scores
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

        # Stage 7: Community detection
        logger.info("\n[Stage 7] Detecting communities...")
        communities = self.community_detector.detect_louvain_communities(G)
        scored_communities = self.community_detector.score_communities(communities, G)
        suspicious_groups = self.community_detector.get_suspicious_groups(
            scored_communities, top_k=10, min_size=3
        )

        # Stage 8: Baseline models
        logger.info("\n[Stage 8] Training baseline models...")
        model_results = self._run_baselines(
            account_features_df, edge_features_df, G
        )

        # Stage 9: GNN models
        logger.info("\n[Stage 9] Training GNN models...")
        gnn_results = self._run_gnn_models(G, account_features_df)
        model_results.update(gnn_results)

        # Stage 10: Evaluation
        logger.info("\n[Stage 10] Evaluation...")
        results["model_comparison"] = model_results
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

        # Stage 13: Visualizations
        logger.info("\n[Stage 13] Creating visualizations...")
        self._create_visualizations(G, communities, df, edge_features_df,
                                    model_results, account_features_df, explanations)

        # Save results
        self._save_results(results)

        elapsed = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"Pipeline completed in {elapsed:.1f} seconds")
        logger.info(f"{'='*60}")

        return results

    def _load_data(self, use_sample: bool = False) -> pd.DataFrame:
        """Load data, creating sample if needed."""
        if use_sample:
            logger.info("Creating sample dataset for testing...")
            create_sample_dataset(self.config["data"]["raw_dir"])

        try:
            df = self.data_loader.load_campaign()
        except FileNotFoundError as e:
            logger.warning(f"{e}")
            logger.info("Creating sample dataset for testing...")
            create_sample_dataset(self.config["data"]["raw_dir"])
            df = self.data_loader.load_campaign()

        info = self.data_loader.get_dataset_info(df)
        logger.info(f"Dataset info: {json.dumps(info, indent=2, default=str)}")
        return df

    def _run_baselines(self, account_df: pd.DataFrame,
                       edge_df: pd.DataFrame,
                       G: "nx.Graph") -> Dict[str, Dict[str, float]]:
        """Run all baseline models."""
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

        # Simple train/test split (80/20)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Threshold baseline (using coordination score as proxy)
        threshold_scores = X[:, 0] / max(X[:, 0].max(), 1)  # Normalized post count
        threshold_pred = self.baseline_models.threshold_baseline(threshold_scores)
        results["threshold_baseline"] = self.baseline_models.evaluate_predictions(
            y, threshold_pred, threshold_scores
        )

        # Random Forest
        rf_pred, rf_prob = self.baseline_models.train_random_forest(X_train, y_train, X_test)
        results["random_forest"] = self.baseline_models.evaluate_predictions(
            y_test, rf_pred, rf_prob
        )

        # Logistic Regression
        lr_pred, lr_prob = self.baseline_models.train_logistic_regression(X_train, y_train, X_test)
        results["logistic_regression"] = self.baseline_models.evaluate_predictions(
            y_test, lr_pred, lr_prob
        )

        # Feature importance
        for model_name in ["random_forest", "logistic_regression"]:
            importance = self.baseline_models.get_feature_importance(model_name, available_cols)
            if importance is not None:
                logger.info(f"\n{model_name} feature importance:\n{importance.to_string()}")

        return results

    def _run_gnn_models(self, G: "nx.Graph",
                        account_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Run GCN and GraphSAGE models."""
        results = {}

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            logger.warning("Empty graph, skipping GNN models")
            return results

        # Prepare PyG data
        feature_cols = self.network_features.select_node_features()
        available_cols = [c for c in feature_cols if c in account_df.columns]

        if not available_cols or "is_io" not in account_df.columns:
            logger.warning("Insufficient features for GNN models")
            return results

        # Create node features and labels
        nodes = sorted(G.nodes())
        node_features = []
        node_labels = []
        node_map = {n: i for i, n in enumerate(nodes)}

        for node in nodes:
            row = account_df[account_df["accountid"] == node]
            if len(row) > 0:
                feats = row[available_cols].values[0]
                node_features.append(feats)
                node_labels.append(int(row["is_io"].values[0]))
            else:
                node_features.append(np.zeros(len(available_cols)))
                node_labels.append(0)

        node_features = np.array(node_features, dtype=np.float32)
        node_labels = np.array(node_labels, dtype=np.int64)

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)

        # Build PyG data
        pyg_data = self.graph_builder.build_pyg_data(G, node_features, node_labels)

        # Create masks
        num_nodes = pyg_data.num_nodes
        indices = np.random.permutation(num_nodes)
        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size+val_size]] = True
        test_mask[indices[train_size+val_size:]] = True

        # GCN
        try:
            logger.info("Training GCN...")
            gcn_results = self._train_and_eval_gnn(
                self.gcn_model, pyg_data, train_mask, val_mask, test_mask,
                in_channels=pyg_data.x.shape[1]
            )
            results["gcn"] = gcn_results
        except Exception as e:
            logger.error(f"GCN training failed: {e}")

        # GraphSAGE
        try:
            logger.info("Training GraphSAGE...")
            sage_results = self._train_and_eval_gnn(
                self.graphsage_model, pyg_data, train_mask, val_mask, test_mask,
                in_channels=pyg_data.x.shape[1]
            )
            results["graphsage"] = sage_results
        except Exception as e:
            logger.error(f"GraphSAGE training failed: {e}")

        return results

    def _train_and_eval_gnn(self, model, data: Data,
                            train_mask: torch.Tensor,
                            val_mask: torch.Tensor,
                            test_mask: torch.Tensor,
                            in_channels: int) -> Dict[str, float]:
        """Train and evaluate a GNN model."""
        model.build_model(in_channels, out_channels=len(data.y.unique()))
        history = model.train(data, train_mask, val_mask)
        metrics = model.evaluate(data, test_mask)
        return metrics

    def _run_ablation(self, account_df: pd.DataFrame,
                       edge_features_df: pd.DataFrame,
                       G: "nx.Graph") -> pd.DataFrame:
        """Run feature ablation study.

        Removes one feature category at a time, rebuilds the graph,
        and re-trains a Random Forest baseline to measure impact.
        """
        feature_column_map = {
            "semantic": ["semantic_similarity"],
            "temporal": ["temporal_score"],
            "url": ["shared_url_score", "shared_url_count"],
            "hashtag": ["shared_hashtag_score", "shared_hashtag_count"],
            "mention": ["shared_mention_score", "shared_mention_count"],
            "repost": ["repost_score"],
        }
        categories = self.config.get("experiments", {}).get(
            "ablation_features", list(feature_column_map.keys())
        )

        # Prepare account-level labels and features
        feature_cols = [
            "post_count", "avg_follower_count", "avg_following_count",
            "repost_ratio", "avg_hashtag_count", "avg_url_count",
            "avg_mention_count", "posts_per_day"
        ]
        available_cols = [c for c in feature_cols if c in account_df.columns]

        if not available_cols or "is_io" not in account_df.columns:
            logger.warning("Insufficient features for ablation study")
            return pd.DataFrame()

        X_full = account_df[available_cols].fillna(0).values
        y = account_df["is_io"].astype(int).values

        if len(np.unique(y)) < 2:
            logger.warning("Only one class present, skipping ablation")
            return pd.DataFrame()

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score, accuracy_score

        X_train_full, X_test_full, y_train, y_test = train_test_split(
            X_full, y, test_size=0.2, random_state=42, stratify=y
        )

        results = []

        # Full model baseline
        rf_full = RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
        rf_full.fit(X_train_full, y_train)
        full_pred = rf_full.predict(X_test_full)
        results.append({
            "ablation": "none (full)",
            "accuracy": accuracy_score(y_test, full_pred),
            "f1": f1_score(y_test, full_pred, average="weighted", zero_division=0),
        })

        # Ablate each coordination feature category
        for category in categories:
            cols_to_remove = feature_column_map.get(category, [])
            ablated_edge_df = edge_features_df.drop(
                columns=[c for c in cols_to_remove if c in edge_features_df.columns],
                errors="ignore"
            )

            # Recompute coordination scores without the ablated feature
            ablated_edge_df["coordination_score"] = ablated_edge_df.apply(
                lambda row: self.edge_feature_extractor.compute_coordination_score(
                    row.to_dict()
                ), axis=1
            )

            # Rebuild graph with ablated edges
            try:
                ablated_G = self.graph_builder.build_networkx_graph(
                    ablated_edge_df, account_df
                )
                ablated_nodes = set(ablated_G.nodes())

                # Use only accounts present in the ablated graph
                mask = account_df["accountid"].isin(ablated_nodes)
                X_ablated = account_df.loc[mask, available_cols].fillna(0).values
                y_ablated = account_df.loc[mask, "is_io"].astype(int).values

                if len(np.unique(y_ablated)) < 2 or len(X_ablated) < 10:
                    logger.warning(
                        f"Skipping ablation for {category}: insufficient data"
                    )
                    results.append({
                        "ablation": f"no {category}",
                        "accuracy": 0.0, "f1": 0.0,
                    })
                    continue

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_ablated, y_ablated, test_size=0.2,
                    random_state=42, stratify=y_ablated
                )

                rf = RandomForestClassifier(
                    n_estimators=100, max_depth=20, random_state=42,
                    class_weight="balanced", n_jobs=-1
                )
                rf.fit(X_tr, y_tr)
                pred = rf.predict(X_te)
                results.append({
                    "ablation": f"no {category}",
                    "accuracy": accuracy_score(y_te, pred),
                    "f1": f1_score(y_te, pred, average="weighted", zero_division=0),
                })
            except Exception as e:
                logger.warning(f"Ablation failed for {category}: {e}")
                results.append({
                    "ablation": f"no {category}",
                    "accuracy": 0.0, "f1": 0.0,
                })

        results_df = pd.DataFrame(results)

        # Save results
        output_path = os.path.join(self.output_dir, "ablation_results.csv")
        results_df.to_csv(output_path, index=False)
        logger.info(f"Ablation results saved to {output_path}")
        logger.info(f"\nAblation Study Results:\n{results_df.to_string(index=False)}")

        return results_df

    def _create_visualizations(self, G, communities, df, edge_features_df,
                               model_results, account_df, explanations):
        """Create all visualizations."""
        try:
            # Network graph
            self.visualizer.plot_network(
                G, title="Coordination Network",
                filename="network_graph"
            )
        except Exception as e:
            logger.warning(f"Network visualization failed: {e}")

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
