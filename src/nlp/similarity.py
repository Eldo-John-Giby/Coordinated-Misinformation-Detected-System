"""
Efficient nearest-neighbor similarity search.

Uses FAISS (preferred) or sklearn NearestNeighbors to find
semantically similar post pairs without O(N²) comparison.
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """Find semantically similar post pairs using efficient NN search."""

    def __init__(self, config: dict):
        self.config = config
        self.sim_config = config.get("similarity", {})
        self.method = self.sim_config.get("method", "faiss")
        self.k_neighbors = self.sim_config.get("k_neighbors", 20)
        self.threshold = self.sim_config.get("semantic_threshold", 0.7)
        self.index_type = self.sim_config.get("faiss_index_type", "flat")

        self._index = None
        self._sklearn_model = None

    def build_index(self, embeddings: np.ndarray):
        """Build a nearest-neighbor index from embeddings."""
        if self.method == "faiss":
            self._build_faiss_index(embeddings)
        else:
            self._build_sklearn_index(embeddings)

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build a FAISS index for cosine similarity search."""
        try:
            import faiss

            dim = embeddings.shape[1]
            # Since embeddings are L2-normalized, inner product = cosine similarity
            if self.index_type == "flat":
                self._index = faiss.IndexFlatIP(dim)  # Inner product
            elif self.index_type == "ivf":
                nlist = min(100, embeddings.shape[0] // 10)
                quantizer = faiss.IndexFlatIP(dim)
                self._index = faiss.IndexIVFFlat(quantizer, dim, nlist)
                self._index.train(embeddings.astype(np.float32))
            else:
                self._index = faiss.IndexFlatIP(dim)

            self._index.add(embeddings.astype(np.float32))
            logger.info(f"Built FAISS index ({self.index_type}): "
                       f"{self._index.ntotal} vectors")
        except ImportError:
            logger.warning("FAISS not available, falling back to sklearn")
            self.method = "sklearn"
            self._build_sklearn_index(embeddings)

    def _build_sklearn_index(self, embeddings: np.ndarray):
        """Build sklearn NearestNeighbors index."""
        from sklearn.neighbors import NearestNeighbors

        self._sklearn_model = NearestNeighbors(
            n_neighbors=min(self.k_neighbors, embeddings.shape[0]),
            metric="cosine",
            algorithm="brute"
        )
        self._sklearn_model.fit(embeddings.astype(np.float32))
        logger.info(f"Built sklearn NearestNeighbors index: "
                   f"{embeddings.shape[0]} vectors")

    def search(self, query_embeddings: np.ndarray,
               k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for each query embedding.

        Returns:
            indices: (n_queries, k) array of neighbor indices
            distances: (n_queries, k) array of distances/similarities
        """
        k = k or self.k_neighbors
        if self._index is not None:
            return self._search_faiss(query_embeddings, k)
        elif self._sklearn_model is not None:
            return self._search_sklearn(query_embeddings, k)
        else:
            raise RuntimeError("No index built. Call build_index() first.")

    def _search_faiss(self, query_embeddings: np.ndarray,
                      k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index."""
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(
            query_embeddings.astype(np.float32), k
        )
        return indices, distances  # distances are cosine similarities

    def _search_sklearn(self, query_embeddings: np.ndarray,
                        k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using sklearn index."""
        k = min(k, self._sklearn_model.n_neighbors)
        distances, indices = self._sklearn_model.kneighbors(
            query_embeddings.astype(np.float32), n_neighbors=k
        )
        # Convert cosine distances to similarities
        similarities = 1.0 - distances
        return indices, similarities

    def find_similar_pairs(self, embeddings: np.ndarray,
                           post_ids: Optional[np.ndarray] = None,
                           threshold: Optional[float] = None
                           ) -> pd.DataFrame:
        """
        Find all similar post pairs above the threshold.

        Args:
            embeddings: (n_posts, dim) embedding matrix
            post_ids: Optional array of post IDs
            threshold: Similarity threshold (default from config)

        Returns:
            DataFrame with columns: post_a, post_b, similarity
        """
        threshold = threshold or self.threshold

        if post_ids is None:
            post_ids = np.arange(len(embeddings))

        self.build_index(embeddings)
        indices, similarities = self.search(embeddings, k=self.k_neighbors)

        pairs = []
        for i in range(len(embeddings)):
            for j_idx in range(1, indices.shape[1]):  # Skip self (index 0)
                neighbor_idx = indices[i, j_idx]
                sim = similarities[i, j_idx]
                if sim >= threshold:
                    pairs.append({
                        "post_a": post_ids[i],
                        "post_b": post_ids[neighbor_idx],
                        "similarity": float(sim),
                    })

        pairs_df = pd.DataFrame(pairs)

        # Remove duplicate pairs (A-B and B-A)
        if not pairs_df.empty:
            pairs_df["pair_key"] = pairs_df.apply(
                lambda row: tuple(sorted([row["post_a"], row["post_b"]])),
                axis=1
            )
            pairs_df = pairs_df.drop_duplicates(subset="pair_key")
            pairs_df = pairs_df.drop(columns=["pair_key"])

        logger.info(f"Found {len(pairs_df)} similar pairs "
                   f"(threshold={threshold:.3f})")
        return pairs_df

    def find_account_similarities(self, account_embeddings: np.ndarray,
                                  account_ids: np.ndarray,
                                  k: int = 10
                                  ) -> pd.DataFrame:
        """
        Find similar account pairs based on average post embeddings.

        Args:
            account_embeddings: (n_accounts, dim) embedding matrix
            account_ids: Array of account IDs
            k: Number of neighbors to find

        Returns:
            DataFrame with columns: account_a, account_b, similarity
        """
        self.build_index(account_embeddings)
        indices, similarities = self.search(account_embeddings, k=k + 1)

        pairs = []
        for i in range(len(account_embeddings)):
            for j_idx in range(1, k + 1):
                neighbor_idx = indices[i, j_idx]
                sim = similarities[i, j_idx]
                if sim >= self.threshold:
                    pairs.append({
                        "account_a": account_ids[i],
                        "account_b": account_ids[neighbor_idx],
                        "similarity": float(sim),
                    })

        pairs_df = pd.DataFrame(pairs)

        if not pairs_df.empty:
            pairs_df["pair_key"] = pairs_df.apply(
                lambda row: tuple(sorted([row["account_a"], row["account_b"]])),
                axis=1
            )
            pairs_df = pairs_df.drop_duplicates(subset="pair_key")
            pairs_df = pairs_df.drop(columns=["pair_key"])

        logger.info(f"Found {len(pairs_df)} similar account pairs")
        return pairs_df
