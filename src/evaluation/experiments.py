"""
Experiment runner for reproducible experiments and ablation studies.

Runs:
1. Feature ablation experiments
2. Model comparison experiments
3. Cross-campaign evaluation
"""

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run and manage experiments."""

    def __init__(self, config: dict):
        self.config = config
        self.exp_config = config.get("experiments", {})
        self.output_dir = self.exp_config.get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_feature_ablation(self,
                             full_features_df: pd.DataFrame,
                             edge_features_df: pd.DataFrame,
                             y_true: np.ndarray,
                             model_fn,
                             feature_categories: Optional[List[str]] = None
                             ) -> pd.DataFrame:
        """
        Run feature ablation study.

        Remove one feature category at a time and measure performance.

        Args:
            full_features_df: Complete feature DataFrame
            edge_features_df: Edge-level features
            y_true: Ground truth labels
            model_fn: Function that takes features and returns predictions
            feature_categories: List of feature categories to ablate

        Returns:
            DataFrame with ablation results
        """
        if feature_categories is None:
            feature_categories = self.exp_config.get("ablation_features", [
                "semantic", "temporal", "url", "hashtag", "mention", "repost"
            ])

        feature_column_map = {
            "semantic": ["semantic_similarity"],
            "temporal": ["temporal_score"],
            "url": ["shared_url_score", "shared_url_count"],
            "hashtag": ["shared_hashtag_score", "shared_hashtag_count"],
            "mention": ["shared_mention_score", "shared_mention_count"],
            "repost": ["repost_score"],
        }

        results = []

        # Full model baseline
        full_pred = model_fn(edge_features_df)
        full_metrics = self._quick_metrics(y_true, full_pred)
        full_metrics["ablation"] = "none (full)"
        results.append(full_metrics)

        # Ablate each category
        for category in feature_categories:
            cols_to_remove = feature_column_map.get(category, [])
            ablated_df = edge_features_df.drop(
                columns=[c for c in cols_to_remove if c in edge_features_df.columns],
                errors="ignore"
            )

            try:
                ablated_pred = model_fn(ablated_df)
                metrics = self._quick_metrics(y_true, ablated_pred)
                metrics["ablation"] = f"no {category}"
            except Exception as e:
                logger.warning(f"Ablation failed for {category}: {e}")
                metrics = {
                    "ablation": f"no {category}",
                    "accuracy": 0, "f1": 0, "roc_auc": 0, "pr_auc": 0
                }

            results.append(metrics)

        results_df = pd.DataFrame(results)

        # Save results
        output_path = os.path.join(self.output_dir, "ablation_results.csv")
        results_df.to_csv(output_path, index=False)
        logger.info(f"Ablation results saved to {output_path}")

        return results_df

    def run_model_comparison(self,
                             model_results: Dict[str, Dict[str, float]]
                             ) -> pd.DataFrame:
        """
        Run model comparison experiment.

        Args:
            model_results: Dict mapping model names to their metrics

        Returns:
            Comparison DataFrame
        """
        rows = []
        for model_name, metrics in model_results.items():
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)

        comparison_df = pd.DataFrame(rows)

        # Save results
        output_path = os.path.join(self.output_dir, "model_comparison.csv")
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Model comparison saved to {output_path}")

        return comparison_df

    def run_experiment_suite(self,
                             results: Dict[str, Dict[str, float]]
                             ) -> Dict:
        """
        Run the full experiment suite.

        Experiments:
        1. Semantic only
        2. Semantic + Temporal
        3. Semantic + Temporal + URL
        4. All features
        5. Traditional ML baseline
        6. GCN
        7. GraphSAGE
        """
        suite_results = {}

        # Record all results
        for name, metrics in results.items():
            suite_results[name] = metrics

        # Save full results
        output_path = os.path.join(self.output_dir, "experiment_suite.json")
        with open(output_path, "w") as f:
            json.dump(suite_results, f, indent=2, default=str)

        logger.info(f"Experiment suite results saved to {output_path}")
        return suite_results

    def _quick_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Quick metric computation for internal use."""
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score, average_precision_score
        )

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred.astype(float))
                metrics["pr_auc"] = average_precision_score(y_true, y_pred.astype(float))
            except ValueError:
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        return metrics

    def save_experiment_config(self, config: dict):
        """Save experiment configuration for reproducibility."""
        output_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Experiment config saved to {output_path}")
