"""
Evaluation metrics for coordinated account detection.

Supports:
- Account-level classification metrics
- Community detection metrics
- Edge/link prediction metrics
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Compute evaluation metrics for different task formulations."""

    def __init__(self, config: dict):
        self.config = config
        self.eval_config = config.get("evaluation", {})

    def compute_classification_metrics(self, y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_scores: np.ndarray = None
                                       ) -> Dict[str, float]:
        """
        Compute classification metrics for account-level IO detection.

        Metrics:
        - Accuracy
        - Precision (weighted)
        - Recall (weighted)
        - F1 (weighted)
        - ROC-AUC
        - PR-AUC
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # Binary metrics
        if len(np.unique(y_true)) == 2:
            metrics["precision_binary"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall_binary"] = recall_score(y_true, y_pred, zero_division=0)
            metrics["f1_binary"] = f1_score(y_true, y_pred, zero_division=0)

        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                if y_scores.ndim > 1:
                    scores = y_scores[:, 1]
                else:
                    scores = y_scores
                metrics["roc_auc"] = roc_auc_score(y_true, scores)
                metrics["pr_auc"] = average_precision_score(y_true, scores)
            except ValueError:
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    def compute_community_metrics(self, true_labels: np.ndarray,
                                  predicted_labels: np.ndarray
                                  ) -> Dict[str, float]:
        """
        Compute community detection quality metrics.

        Metrics:
        - Adjusted Rand Index (ARI)
        - Normalized Mutual Information (NMI)
        """
        metrics = {}

        if len(np.unique(true_labels)) > 1 and len(np.unique(predicted_labels)) > 1:
            metrics["adjusted_rand_index"] = adjusted_rand_score(
                true_labels, predicted_labels
            )
            metrics["normalized_mutual_info"] = normalized_mutual_info_score(
                true_labels, predicted_labels
            )
        else:
            metrics["adjusted_rand_index"] = 0.0
            metrics["normalized_mutual_info"] = 0.0

        return metrics

    def compute_link_prediction_metrics(self, y_true: np.ndarray,
                                        y_scores: np.ndarray
                                        ) -> Dict[str, float]:
        """
        Compute link prediction metrics.

        Metrics:
        - ROC-AUC
        - Average Precision
        """
        metrics = {}
        if len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
                metrics["average_precision"] = average_precision_score(y_true, y_scores)
            except ValueError:
                metrics["roc_auc"] = 0.0
                metrics["average_precision"] = 0.0
        return metrics

    def compute_group_purity(self, communities: Dict[int, List[str]],
                             true_labels: Dict[str, bool]) -> float:
        """
        Compute community purity.

        Purity = fraction of accounts in each community that share
        the same ground-truth label.
        """
        total_correct = 0
        total_accounts = 0

        for comm_id, members in communities.items():
            if len(members) == 0:
                continue

            labels = [true_labels.get(m, False) for m in members]
            majority_label = max(set(labels), key=labels.count)
            correct = sum(1 for l in labels if l == majority_label)

            total_correct += correct
            total_accounts += len(members)

        return total_correct / total_accounts if total_accounts > 0 else 0.0

    def generate_report(self, metrics: Dict[str, float],
                        model_name: str = "Model") -> str:
        """Generate a human-readable evaluation report."""
        lines = [f"=== {model_name} Evaluation Report ===\n"]

        for key, value in metrics.items():
            if key == "confusion_matrix":
                lines.append(f"\nConfusion Matrix:")
                cm = np.array(value)
                lines.append(f"  {cm}")
            elif isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create comparison table of multiple models."""
        rows = []
        for model_name, metrics in results.items():
            row = {"model": model_name}
            row.update({k: v for k, v in metrics.items()
                       if isinstance(v, (int, float)) and k != "confusion_matrix"})
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def format_results_table(self, results_df: pd.DataFrame) -> str:
        """Format results as a readable table."""
        key_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        available_cols = [c for c in key_cols if c in results_df.columns]

        return results_df[available_cols].to_string(index=False, float_format="%.4f")
