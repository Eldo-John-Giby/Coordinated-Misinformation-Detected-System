"""
Embedding computation using Sentence Transformers.

Uses sentence-transformers/all-MiniLM-L6-v2 to produce 384-dimensional
embeddings for each post's text content.
"""

import os
import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Compute text embeddings using Sentence Transformers."""

    def __init__(self, config: dict):
        self.config = config
        self.nlp_config = config.get("nlp", {})
        self.model_name = self.nlp_config.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_dim = self.nlp_config.get("embedding_dim", 384)
        self.batch_size = self.nlp_config.get("batch_size", 256)
        self.device = self.nlp_config.get("device") or self._detect_device()

        self._model = None
        self._cache_dir = os.path.join("data", "processed", "embeddings_cache")

    def _detect_device(self) -> str:
        """Auto-detect available device."""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            logger.info("Using CPU for embeddings")
        return device

    def _load_model(self):
        """Lazy-load the Sentence Transformer model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name, device=self.device
            )
            logger.info(f"Loaded model: {self.model_name} on {self.device}")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

    def encode_texts(self, texts: List[str],
                     show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts: List of text strings to encode.
            show_progress: Whether to show progress bar.

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        self._load_model()

        # Replace None/NaN with empty strings
        clean_texts = [str(t) if t and str(t).strip() else "" for t in texts]

        embeddings = self._model.encode(
            clean_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )

        logger.info(f"Computed embeddings: shape={embeddings.shape}")
        return embeddings

    def encode_dataframe(self, df: pd.DataFrame,
                         text_column: str = "post_text",
                         output_column: str = "embedding",
                         show_progress: bool = True) -> pd.DataFrame:
        """Compute embeddings for a DataFrame and add as a column."""
        texts = df[text_column].fillna("").tolist()
        embeddings = self.encode_texts(texts, show_progress=show_progress)
        df[output_column] = list(embeddings)
        return df

    def compute_account_embeddings(self, df: pd.DataFrame,
                                   text_column: str = "post_text",
                                   show_progress: bool = True) -> pd.DataFrame:
        """
        Compute average embedding per account.

        Aggregates all post embeddings for each account into a single
        account-level embedding by averaging.
        """
        if "embedding" not in df.columns:
            df = self.encode_dataframe(df, text_column, show_progress=show_progress)

        # Convert embeddings column to numpy matrix
        emb_matrix = np.array(df["embedding"].tolist())
        account_ids = df["accountid"].values

        # Group by account and average embeddings
        unique_accounts = np.unique(account_ids)
        account_emb_dict = {}
        for acc in unique_accounts:
            mask = account_ids == acc
            account_emb_dict[acc] = emb_matrix[mask].mean(axis=0)

        # Build DataFrame
        emb_df = pd.DataFrame.from_dict(account_emb_dict, orient="index")
        emb_df.index.name = "accountid"
        emb_df = emb_df.reset_index()
        emb_df.columns = ["accountid"] + [f"emb_{i}" for i in range(emb_df.shape[1] - 1)]

        logger.info(f"Computed account embeddings for "
                   f"{len(emb_df)} accounts")
        return emb_df

    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Save embeddings to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        np.save(path, embeddings)
        logger.info(f"Saved embeddings to {path}: shape={embeddings.shape}")

    def load_embeddings(self, path: str) -> np.ndarray:
        """Load embeddings from disk."""
        embeddings = np.load(path)
        logger.info(f"Loaded embeddings from {path}: shape={embeddings.shape}")
        return embeddings
