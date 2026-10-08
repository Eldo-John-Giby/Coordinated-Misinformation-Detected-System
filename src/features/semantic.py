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

    def aggregate_post_pair_candidates(
        self,
        post_pairs_df: pd.DataFrame,
        top_n: int = 3,
    ) -> pd.DataFrame:
        """Aggregate post-level candidates up to account-pair rows (item 4b).

        For each account pair, keeps the top_n highest-scoring post pairs
        (by bi-encoder similarity). The best post pair defines the account
        pair's candidate score; the full matched post pairs are preserved so
        the cross-encoder reranks the ACTUAL matched texts, and the winning
        pairs can later be surfaced as evidence (item 4d).

        Returns DataFrame: account_a, account_b, biencoder_similarity,
        n_post_pairs, matched_post_pairs (list of dicts with post ids + texts
        keys to be filled by the caller).
        """
        if post_pairs_df is None or post_pairs_df.empty:
            return pd.DataFrame()

        pp = post_pairs_df.sort_values("biencoder_similarity", ascending=False)
        ranked = pp.groupby(["account_a", "account_b"], group_keys=False)

        # Top-N post pairs per account pair
        top_pairs = ranked.head(top_n).copy()

        # Best score per account pair
        best = (
            pp.groupby(["account_a", "account_b"], as_index=False)
            .first()
            [["account_a", "account_b", "biencoder_similarity"]]
            .rename(columns={"biencoder_similarity": "biencoder_similarity"})
        )

        agg = best.merge(
            top_pairs.assign(
                matched=lambda d: d.apply(
                    lambda r: {
                        "post_a": r["post_a"], "post_b": r["post_b"],
                        "biencoder_similarity": float(r["biencoder_similarity"]),
                    }, axis=1,
                )
            )[["account_a", "account_b", "matched"]]
            .groupby(["account_a", "account_b"], as_index=False)
            .agg({"matched": lambda x: list(x)}),
            on=["account_a", "account_b"],
            how="left",
        )
        agg["n_post_pairs"] = agg["matched"].apply(len)

        logger.info(
            "Aggregated %d post-pair candidates into %d account pairs "
            "(top_%d post pairs kept per pair)",
            len(pp), len(agg), top_n,
        )
        return agg

    def merge_semantic_scores(
        self,
        biencoder_scores: Dict[Tuple[str, str], float],
        crossencoder_scores: Optional[Dict[Tuple[str, str], float]] = None,
        use_cross_encoder: bool = False,
        degradations=None,
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Merge bi-encoder and cross-encoder scores for account pairs.

        Args:
            biencoder_scores: Dict mapping (acc_a, acc_b) -> bi-encoder cosine similarity.
            crossencoder_scores: Optional dict mapping (acc_a, acc_b) -> cross-encoder score.
            use_cross_encoder: If True, the "semantic_similarity" key in the returned dict
                uses the cross-encoder score (for the coordination formula). If False,
                the bi-encoder score is used.
            degradations: Optional DegradationTracker. Any pair that ends up
                scored by the bi-encoder only is recorded there (item 8c):
                a bi-encoder-only semantic score is lower-confidence and must
                not silently pass as if it had been reranked.

        Returns:
            Dict mapping (acc_a, acc_b) -> dict with keys:
                "semantic_similarity": the active score used in coordination formula
                "semantic_score_biencoder": the bi-encoder score (always present)
                "semantic_score_crossencoder": the cross-encoder score (if provided)
        """
        crossencoder_scores = crossencoder_scores or {}
        merged = {}
        bi_only_pairs = 0

        for pair_key, bi_score in biencoder_scores.items():
            cross_score = crossencoder_scores.get(pair_key, None)
            entry = {"semantic_score_biencoder": bi_score}
            if cross_score is not None:
                entry["semantic_score_crossencoder"] = cross_score
            # The active semantic_similarity used by the coordination formula
            if use_cross_encoder and cross_score is not None:
                entry["semantic_similarity"] = cross_score
            else:
                entry["semantic_similarity"] = bi_score
                if use_cross_encoder and cross_score is None:
                    bi_only_pairs += 1
            merged[pair_key] = entry

        # Also include pairs that only have cross-encoder scores (unusual but safe)
        for pair_key, cross_score in crossencoder_scores.items():
            if pair_key not in merged:
                merged[pair_key] = {
                    "semantic_score_biencoder": 0.0,
                    "semantic_score_crossencoder": cross_score,
                    "semantic_similarity": cross_score if use_cross_encoder else 0.0,
                }

        if bi_only_pairs > 0:
            reason = (
                f"{bi_only_pairs} account pairs scored by bi-encoder only "
                "(cross-encoder disabled or missed these candidates); "
                "semantic scores are lower-confidence"
            )
            if degradations is not None:
                degradations.record("cross_encoder", reason)
            else:
                logger.warning(reason)

        logger.info(
            f"Merged semantic scores for {len(merged)} pairs "
            f"({len(crossencoder_scores)} with cross-encoder scores, "
            f"use_cross_encoder={use_cross_encoder})")
        return merged
