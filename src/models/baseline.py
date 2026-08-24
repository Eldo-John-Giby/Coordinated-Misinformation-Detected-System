"""
Baseline ML models for coordinated account detection.

Includes:
1. Threshold-based detection
2. Random Forest classifier
3. Logistic Regression
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

logger = logging.getLogger(__name__)


class BaselineModels:
    """Baseline models for comparison with GNN approaches."""

    def __init__(self, config: dict):
        self.config = config
        self.baseline_config = config.get("models", {}).get("baseline", {})
        self.scaler = StandardScaler()
        self._models = {}

    def threshold_baseline(self, scores: np.ndarray,
                           threshold: float = 0.5) -> np.ndarray:
        """
        Simple threshold-based classification.

        Args:
            scores: Coordination scores for each account
            threshold: Classification threshold

        Returns:
            Binary predictions (0=control, 1=IO)
        """
        return (scores >= threshold).astype(int)

    def optimize_threshold(self, scores: np.ndarray,
                           y_true: np.ndarray,
                           metric: str = "f1") -> float:
        """Find optimal threshold for a given metric."""
        from sklearn.metrics import f1_score as f1

        best_threshold = 0.5
        best_score = 0

        for threshold in np.arange(0.1, 0.95, 0.05):
            preds = (scores >= threshold).astype(int)
            if metric == "f1":
                score = f1(y_true, preds, zero_division=0)
            elif metric == "balanced":
                p = precision_score(y_true, preds, zero_division=0)
                r = recall_score(y_true, preds, zero_division=0)
                score = 2 * p * r / (p + r) if (p + r) > 0 else 0
            else:
                score = f1(y_true, preds, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        logger.info(f"Optimal threshold: {best_threshold:.3f} "
                   f"(best {metric}: {best_score:.4f})")
        return best_threshold

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Train Random Forest and return predictions."""
        rf_config = self.baseline_config.get("random_forest", {})
        model = RandomForestClassifier(
            n_estimators=rf_config.get("n_estimators", 100),
            max_depth=rf_config.get("max_depth", 20),
            random_state=rf_config.get("random_state", 42),
            class_weight="balanced",
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        self._models["random_forest"] = model

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        logger.info("Random Forest trained successfully")
        return predictions, probabilities

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Train Logistic Regression and return predictions."""
        lr_config = self.baseline_config.get("logistic_regression", {})
        model = LogisticRegression(
            max_iter=lr_config.get("max_iter", 1000),
            random_state=lr_config.get("random_state", 42),
            class_weight="balanced"
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        self._models["logistic_regression"] = model

        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]

        logger.info("Logistic Regression trained successfully")
        return predictions, probabilities

    def get_feature_importance(self, model_name: str,
                               feature_names: list) -> Optional[pd.DataFrame]:
        """Get feature importance from a trained model."""
        model = self._models.get(model_name)
        if model is None:
            return None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            return None

        df = pd.DataFrame({
            "feature": feature_names[:len(importance)],
            "importance": importance
        }).sort_values("importance", ascending=False)

        return df

    def evaluate_predictions(self, y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_scores: np.ndarray = None) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
                metrics["pr_auc"] = average_precision_score(y_true, y_scores)
            except ValueError:
                metrics["roc_auc"] = 0.0
                metrics["pr_auc"] = 0.0

        return metrics
