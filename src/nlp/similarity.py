"""
Efficient nearest-neighbor similarity search.

Uses FAISS (preferred) or sklearn NearestNeighbors to find
semantically similar post pairs without O(N²) comparison.

Also provides CrossEncoderReranker for a retrieve-then-rerank pattern:
bi-encoder + FAISS retrieves top-k candidates cheaply, then a cross-encoder
re-scores those pairs with full cross-attention for higher precision.
"""

import logging
import time
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SimilaritySearch:
    """Find semantically similar post pairs using efficient NN search."""

    def __init__(self, config: dict, degradations=None):
        self.config = config
        self.sim_config = config.get("similarity", {})
        self.method = self.sim_config.get("method", "faiss")
        self.k_neighbors = self.sim_config.get("k_neighbors", 20)
        self.threshold = self.sim_config.get("semantic_threshold", 0.7)
        self.index_type = self.sim_config.get("faiss_index_type", "flat")
        # Item 5: retrieval is recall-oriented. This floor only removes
        # obviously-unrelated pairs BEFORE the cross-encoder rerank; the
        # cross-encoder score is the precision gate. Kept deliberately low
        # (0.3): for L2-normalized MiniLM embeddings, unrelated text rarely
        # exceeds ~0.3 cosine, while paraphrases stay far above it.
        self.retrieval_threshold = self.sim_config.get("retrieval_floor", 0.3)

        # Optional DegradationTracker (item 8a): FAISS -> sklearn fallback
        # changes the retrieval algorithm and must be surfaced in results.
        self._degradations = degradations

        self._index = None
        self._sklearn_model = None
        self._fallback_recorded = False

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
            reason = (
                "faiss not installed; fell back to sklearn "
                "NearestNeighbors (brute-force cosine search - slower, "
                "exhaustive, different recall characteristics)"
            )
            logger.warning(reason)
            if not self._fallback_recorded:
                if self._degradations is not None:
                    self._degradations.record("faiss", reason)
                else:
                    logger.warning("DEGRADED [faiss]: %s", reason)
                self._fallback_recorded = True
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

    def find_post_pairs(self, embeddings: np.ndarray,
                        post_ids: np.ndarray,
                        account_ids: np.ndarray,
                        k: Optional[int] = None,
                        threshold: Optional[float] = None,
                        ) -> pd.DataFrame:
        """Post-level candidate retrieval (item 4a).

        Runs the FAISS/sklearn index over ALL post embeddings (not account
        averages) and returns candidate POST pairs with their bi-encoder
        similarity. This stage is intentionally recall-oriented: `threshold`
        is a low floor that only removes obviously-unrelated pairs; precision
        comes later from the cross-encoder rerank (item 5).

        Args:
            embeddings: (n_posts, dim) L2-normalized post embedding matrix.
            post_ids: array of post IDs aligned with embeddings rows.
            account_ids: array of account IDs aligned with embeddings rows.
            k: neighbors per post (default from config k_neighbors).
            threshold: similarity floor for keeping a candidate pair.

        Returns:
            DataFrame: post_a, account_a, post_b, account_b, biencoder_similarity
            Cross-account pairs only; (A,B) and (B,A) deduplicated; self-pairs
            (same account) excluded so account-level aggregation can never
            produce a self-loop edge.
        """
        k = k or self.k_neighbors
        threshold = self.retrieval_threshold if threshold is None else threshold

        self.build_index(embeddings)
        # +1 because the nearest neighbor of a post is usually itself
        indices, similarities = self.search(embeddings, k=min(k + 1, self._index_size()))

        pairs = []
        n = len(embeddings)
        for i in range(n):
            acc_a = account_ids[i]
            for j_idx in range(indices.shape[1]):
                neighbor_idx = indices[i, j_idx]
                if neighbor_idx == i:
                    continue  # self
                acc_b = account_ids[neighbor_idx]
                if acc_b == acc_a:
                    continue  # same account: not a coordination pair
                sim = float(similarities[i, j_idx])
                if sim < threshold:
                    continue
                pairs.append({
                    "post_a": post_ids[i],
                    "account_a": acc_a,
                    "post_b": post_ids[neighbor_idx],
                    "account_b": acc_b,
                    "biencoder_similarity": sim,
                })

        pairs_df = pd.DataFrame(pairs)
        if pairs_df.empty:
            logger.warning("Post-level retrieval found 0 candidate pairs "
                           "(threshold=%.3f)", threshold)
            return pairs_df

        # Deduplicate symmetric pairs, keeping the higher score
        pairs_df["pair_key"] = pairs_df.apply(
            lambda r: tuple(sorted([str(r["post_a"]), str(r["post_b"])])),
            axis=1,
        )
        pairs_df = (
            pairs_df.sort_values("biencoder_similarity", ascending=False)
            .drop_duplicates(subset="pair_key")
            .drop(columns=["pair_key"])
            .reset_index(drop=True)
        )

        logger.info(
            "Post-level retrieval: %d candidate post pairs across %d account "
            "pairs (k=%d, retrieval_floor=%.3f)",
            len(pairs_df),
            pairs_df.groupby(
                [pairs_df["account_a"], pairs_df["account_b"]]
            ).ngroups,
            k, threshold,
        )
        return pairs_df

    def _index_size(self) -> int:
        """Number of vectors in the active index (for k clamping)."""
        if self._index is not None:
            return self._index.ntotal
        if self._sklearn_model is not None:
            return getattr(self._sklearn_model, "n_samples_fit_", self.k_neighbors)
        return self.k_neighbors

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


class CrossEncoderReranker:
    """Re-score bi-encoder candidate pairs with a cross-encoder.

    Cross-encoders process both texts in a single forward pass with full
    cross-attention, making them much better at recognizing paraphrases
    than independent bi-encoder embeddings.

    Typical usage in a retrieve-then-rerank pipeline:
    1. SimilaritySearch.find_similar_pairs() retrieves top-k candidates
    2. CrossEncoderReranker.rerank() re-scores those candidates

    Example::

        reranker = CrossEncoderReranker(config)
        reranked_df = reranker.rerank(
            candidate_pairs_df,
            texts_a=["post text A", ...],
            texts_b=["post text B", ...],
        )
    """

    def __init__(self, config: dict, degradations=None):
        self.config = config
        sim_config = config.get("similarity", {})
        self.model_name = sim_config.get(
            "cross_encoder_model", "cross-encoder/nli-deberta-v3-base"
        )
        self.batch_size = sim_config.get("cross_encoder_batch_size", 64)
        self._model = None
        # Optional DegradationTracker (item 8): pairs that keep a NaN
        # cross-encoder score (cap/disabled/missed) are lower-confidence and
        # must surface in pipeline_results.json, not just the logs.
        self.degradations = degradations
        self._warned_lower_confidence = False

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Cross-encoder model loaded successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            )

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Normalize raw model outputs to [0, 1] as a numpy array.

        Entailment models (nli-deberta-v3-base) output 3 logits per pair
        (contradiction, neutral, entailment). We take the entailment logit
        and apply sigmoid. Paraphrase models output a single logit; we apply
        sigmoid directly.
        """
        if raw_scores.ndim == 2 and raw_scores.shape[1] == 3:
            # Entailment model: 3 classes -> use entailment column (index 2).
            # torch.sigmoid (item 7): the free F.sigmoid function is deprecated
            # and will be removed in a future torch release.
            import torch
            return torch.sigmoid(torch.as_tensor(raw_scores[:, 2])).numpy()
        elif raw_scores.ndim == 2 and raw_scores.shape[1] == 1:
            return 1.0 / (1.0 + np.exp(-raw_scores[:, 0]))
        else:
            # Already 1D or unexpected shape: apply sigmoid
            return 1.0 / (1.0 + np.exp(-raw_scores))

    def rerank(
        self,
        candidate_df: pd.DataFrame,
        texts_a: List[str],
        texts_b: List[str],
        score_column: str = "semantic_similarity",
        new_score_column: str = "cross_encoder_score",
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Re-score candidate pairs with the cross-encoder.

        Args:
            candidate_df: DataFrame with columns post_a/post_b or account_a/account_b
                and a similarity score column.
            texts_a: List of text strings corresponding to the first entity in each pair.
            texts_b: List of text strings corresponding to the second entity in each pair.
            score_column: Name of the existing bi-encoder score column.
            new_score_column: Name for the new cross-encoder score column.
            top_k: If set, only the top_k rows (ranked by score_column) are
                re-scored with the cross-encoder; all other rows keep a NaN
                cross-encoder score. Implements the cost-control documented by
                config key cross_encoder_rerank_k (item 6 - previously this
                config was read nowhere). None means rerank everything.

        Returns:
            DataFrame with the new cross-encoder score column added.
        """
        self._load_model()

        if candidate_df.empty:
            candidate_df[new_score_column] = []
            return candidate_df

        result = candidate_df.copy()

        # Optional cost-control truncation (item 6)
        rerank_mask = pd.Series(True, index=result.index)
        if top_k is not None and top_k > 0 and len(result) > top_k:
            if score_column in result.columns:
                order = result[score_column].rank(ascending=False, method="first")
                rerank_mask = order <= top_k
            else:
                rerank_mask = pd.Series(
                    [i < top_k for i in range(len(result))], index=result.index
                )
            logger.info(
                "Cross-encoder cost control: reranking top %d of %d candidates "
                "(cross_encoder_rerank_k=%d)",
                int(rerank_mask.sum()), len(result), top_k,
            )
        rerank_idx = result.index[rerank_mask]

        positions = [result.index.get_loc(i) for i in rerank_idx]
        texts_a_r = [texts_a[p] for p in positions]
        texts_b_r = [texts_b[p] for p in positions]

        n_pairs = len(texts_a_r)
        assert len(texts_b_r) == n_pairs, (
            f"texts_a and texts_b must have same length, got {n_pairs} vs {len(texts_b_r)}"
        )

        logger.info(
            f"Cross-encoder reranking {n_pairs} pairs with model {self.model_name}"
        )
        t0 = time.time()

        # Batch inference
        raw_scores = self._model.predict(
            list(zip(texts_a_r, texts_b_r)),
            batch_size=self.batch_size,
            show_progress_bar=n_pairs > 100,
        )
        raw_scores = np.asarray(raw_scores)
        scores = self._normalize_scores(raw_scores)

        elapsed = time.time() - t0
        logger.info(
            f"Cross-encoder reranked {n_pairs} pairs in {elapsed:.2f}s "
            f"({n_pairs / max(elapsed, 0.001):.1f} pairs/sec)"
        )

        result[new_score_column] = np.nan
        result.loc[rerank_idx, new_score_column] = scores

        n_missed = int(result[new_score_column].isna().sum())
        if n_missed > 0 and not self._warned_lower_confidence:
            reason = (
                f"{n_missed} candidate pairs not reranked by the cross-encoder "
                "(rerank cap or disabled); their semantic scores fall back to "
                "the lower-confidence bi-encoder score"
            )
            if self.degradations is not None:
                self.degradations.record("cross_encoder", reason)
            else:
                logger.warning(reason)
            self._warned_lower_confidence = True
        return result
