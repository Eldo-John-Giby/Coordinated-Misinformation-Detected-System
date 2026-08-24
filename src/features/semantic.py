"""
Semantic features for coordination detection.

Computes semantic similarity between posts and accounts using
Sentence Transformer embeddings and cosine similarity.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_sim

logger = logging.getLogger(__name__)


class SemanticFeatures:
    """Extract semantic coordination features."""

    def __init__(self, config: dict):
        self.config = config
        self.sim_config = config.get("similarity", {})
        self.threshold = self.sim_config.get("semantic_threshold", 0.7)

    def compute_post_pair_similarity(self, emb_a: np.ndarray,
                                     emb_b: np.ndarray) -> float:
        """Compute cosine similarity between two post embeddings."""
        a = emb_a.reshape(1, -1)
        b = emb_b.reshape(1, -1)
        return float(sklearn_cosine_sim(a, b)[0, 0])

    def compute_account_pair_semantic_score(
        self,
        posts_a_embeddings: np.ndarray,
        posts_b_embeddings: np.ndarray,
        method: str = "max_mean"
    ) -> Dict[str, float]:
        """
        Compute semantic coordination score between two accounts.

        Methods:
        - max_mean: average of top-5 pairwise similarities
        - centroid: similarity between account centroid embeddings
        - max_pair: maximum pairwise similarity

        Returns dict with multiple semantic metrics.
        """
        if len(posts_a_embeddings) == 0 or len(posts_b_embeddings) == 0:
            return {
                "semantic_similarity": 0.0,
                "semantic_max_similarity": 0.0,
                "semantic_centroid_similarity": 0.0,
            }

        # Compute pairwise similarity matrix
        sim_matrix = sklearn_cosine_sim(
            posts_a_embeddings.astype(np.float32),
            posts_b_embeddings.astype(np.float32)
        )

        # Max mean: average of top-5 similarities from each direction
        k = min(5, sim_matrix.shape[1])
        top_k_b = np.sort(sim_matrix, axis=1)[:, -k:].mean(axis=1)
        top_k_a = np.sort(sim_matrix, axis=0)[-k:, :].mean(axis=0)

        max_mean_score = (top_k_b.mean() + top_k_a.mean()) / 2

        # Maximum pairwise similarity
        max_sim = float(sim_matrix.max())

        # Centroid similarity
        centroid_a = posts_a_embeddings.mean(axis=0, keepdims=True)
        centroid_b = posts_b_embeddings.mean(axis=0, keepdims=True)
        centroid_sim = float(sklearn_cosine_sim(centroid_a, centroid_b)[0, 0])

        return {
            "semantic_similarity": float(max_mean_score),
            "semantic_max_similarity": max_sim,
            "semantic_centroid_similarity": centroid_sim,
        }

    def aggregate_post_embeddings(self, df: pd.DataFrame,
                                  embedding_col: str = "embedding"
                                  ) -> Dict[str, np.ndarray]:
        """Aggregate post embeddings by account."""
        account_embeddings = {}
        for account, group in df.groupby("accountid"):
            embeddings = np.array(group[embedding_col].tolist())
            account_embeddings[account] = embeddings
        return account_embeddings

    def compute_batch_account_similarities(
        self,
        account_pairs: List[Tuple[str, str]],
        account_embeddings: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Compute semantic features for a batch of account pairs."""
        results = []
        for acc_a, acc_b in account_pairs:
            emb_a = account_embeddings.get(acc_a, np.array([]))
            emb_b = account_embeddings.get(acc_b, np.array([]))
            scores = self.compute_account_pair_semantic_score(emb_a, emb_b)
            scores["account_a"] = acc_a
            scores["account_b"] = acc_b
            results.append(scores)

        return pd.DataFrame(results)
