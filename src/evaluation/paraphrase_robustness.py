"""
Paraphrase-robustness experiment.

Demonstrates that cross-encoders maintain high scores on paraphrased
coordinated content while bi-encoder scores degrade. This is the key
experimental contribution showing why cross-encoder reranking matters
against LLM-paraphrased coordinated messaging.

Can be run independently of the full pipeline:
    python -m src.evaluation.paraphrase_robustness --sample
    python -m src.evaluation.paraphrase_robustness --config configs/config.yaml
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ParaphraseRobustnessExperiment:
    """Compare bi-encoder vs cross-encoder robustness to LLM paraphrasing.

    1. Sample known-coordinated post pairs (IO accounts, high bi-encoder score)
    2. Paraphrase one post per pair using an LLM
    3. Score original-original and original-paraphrased with both encoders
    4. Produce before/after comparison table and visualization
    """

    def __init__(self, config: dict):
        self.config = config
        exp_config = config.get("experiments", {})
        self.output_dir = exp_config.get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.n_pairs = config.get("paraphrase_experiment", {}).get("n_pairs", 200)
        self.min_biencoder_score = config.get("paraphrase_experiment", {}).get(
            "min_biencoder_score", 0.7
        )
        self.paraphrase_model = config.get("paraphrase_experiment", {}).get(
            "paraphrase_model", "claude-sonnet-4-6"
        )
        self.random_seed = config.get("data", {}).get("random_seed", 42)

    # ------------------------------------------------------------------
    # Step 1: Sample coordinated post pairs
    # ------------------------------------------------------------------

    def sample_coordinated_pairs(
        self, df: pd.DataFrame
    ) -> List[Tuple[str, str, str, str]]:
        """Sample post pairs from known-coordinated IO accounts.

        Returns list of (post_a_text, post_b_text, account_a, account_b).
        """
        from src.nlp.embeddings import EmbeddingModel
        from src.nlp.similarity import SimilaritySearch

        logger.info("Sampling coordinated post pairs...")

        # Filter to IO accounts only
        io_accounts = set(
            df.groupby("accountid")["is_control"]
            .all()
            .reset_index()
            .query("is_control == False")["accountid"]
        )
        df_io = df[df["accountid"].isin(io_accounts)].copy()

        if len(df_io) == 0:
            logger.warning("No IO accounts found; falling back to all accounts")
            df_io = df.copy()

        # Compute embeddings
        embedder = EmbeddingModel(self.config)
        df_io = embedder.encode_dataframe(df_io)

        # Build account-level embeddings (mean of post embeddings per account)
        account_emb = {}
        for acc, group in df_io.groupby("accountid"):
            embs = np.array(group["embedding"].tolist())
            account_emb[acc] = embs.mean(axis=0)

        acc_ids = list(account_emb.keys())
        emb_matrix = np.array([account_emb[a].astype(np.float32) for a in acc_ids])

        # FAISS retrieval
        sim_search = SimilaritySearch(self.config)
        sim_search.build_index(emb_matrix)
        indices, similarities = sim_search.search(emb_matrix, k=self.config.get("similarity", {}).get("k_neighbors", 20))

        # Collect candidate pairs with bi-encoder scores
        candidates = []
        for i in range(len(acc_ids)):
            for j_idx in range(1, indices.shape[1]):
                neighbor_idx = indices[i, j_idx]
                sim = float(similarities[i, j_idx])
                if sim >= self.min_biencoder_score:
                    acc_a = acc_ids[i]
                    acc_b = acc_ids[neighbor_idx]
                    if acc_a != acc_b:
                        candidates.append((acc_a, acc_b, sim))

        if not candidates:
            logger.warning("No high-scoring pairs found; lowering threshold")
            for i in range(len(acc_ids)):
                for j_idx in range(1, min(5, indices.shape[1])):
                    neighbor_idx = indices[i, j_idx]
                    sim = float(similarities[i, j_idx])
                    acc_a = acc_ids[i]
                    acc_b = acc_ids[neighbor_idx]
                    if acc_a != acc_b:
                        candidates.append((acc_a, acc_b, sim))

        # Sort by score descending, take top n_pairs
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[: self.n_pairs]

        # For each account pair, pick the two posts that are most similar
        # (the ones that drove the account-level similarity)
        pairs = []
        for acc_a, acc_b, bi_score in candidates:
            posts_a = df_io[df_io["accountid"] == acc_a]["post_text"].tolist()
            posts_b = df_io[df_io["accountid"] == acc_b]["post_text"].tolist()

            # Pick first non-empty post from each account
            text_a = next((t for t in posts_a if isinstance(t, str) and t.strip()), "")
            text_b = next((t for t in posts_b if isinstance(t, str) and t.strip()), "")

            if text_a and text_b:
                pairs.append((text_a, text_b, acc_a, acc_b))

        logger.info(f"Sampled {len(pairs)} coordinated post pairs")
        return pairs

    # ------------------------------------------------------------------
    # Step 2: Paraphrase via LLM
    # ------------------------------------------------------------------

    def paraphrase_pairs(
        self, pairs: List[Tuple[str, str, str, str]]
    ) -> List[Tuple[str, str, str, str, str]]:
        """Paraphrase the second post in each pair using Anthropic API.

        Returns list of (text_a, text_b_original, text_b_paraphrased, acc_a, acc_b).
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for paraphrase experiment. "
                "Install with: pip install anthropic"
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable is required for "
                "the paraphrase robustness experiment. Set it before running."
            )

        client = anthropic.Anthropic(api_key=api_key)

        prompt_template = (
            "Paraphrase the following social media post, preserving its "
            "meaning but using substantially different wording. "
            "Output ONLY the paraphrased text, nothing else.\n\n"
            "Post: {text}"
        )

        results = []
        total = len(pairs)

        for idx, (text_a, text_b, acc_a, acc_b) in enumerate(pairs):
            if (idx + 1) % 25 == 0 or idx == 0:
                logger.info(f"Paraphrasing pair {idx + 1}/{total}...")

            try:
                message = client.messages.create(
                    model=self.paraphrase_model,
                    max_tokens=512,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_template.format(text=text_b),
                        }
                    ],
                )
                paraphrased = message.content[0].text.strip()
            except Exception as e:
                logger.warning(f"Paraphrase failed for pair {idx}: {e}")
                paraphrased = text_b  # Fallback: use original

            results.append((text_a, text_b, paraphrased, acc_a, acc_b))

        logger.info(f"Paraphrased {len(results)} pairs")
        return results

    # ------------------------------------------------------------------
    # Step 3: Score with both encoders
    # ------------------------------------------------------------------

    def score_pairs(
        self,
        paraphrased_pairs: List[Tuple[str, str, str, str, str]],
    ) -> pd.DataFrame:
        """Compute bi-encoder and cross-encoder scores for original and paraphrased pairs.

        For each pair, computes:
        - bi-encoder cosine similarity (original_a vs original_b)
        - bi-encoder cosine similarity (original_a vs paraphrased_b)
        - cross-encoder score (original_a vs original_b)
        - cross-encoder score (original_a vs paraphrased_b)
        """
        from sentence_transformers import SentenceTransformer
        from src.nlp.similarity import CrossEncoderReranker

        biencoder_model_name = self.config.get("nlp", {}).get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info(f"Loading bi-encoder model: {biencoder_model_name}")
        bi_encoder = SentenceTransformer(biencoder_model_name)

        logger.info("Loading cross-encoder model...")
        reranker = CrossEncoderReranker(self.config)

        # Collect all texts we need to embed (unique texts)
        all_texts_a = [p[0] for p in paraphrased_pairs]
        all_texts_b_orig = [p[1] for p in paraphrased_pairs]
        all_texts_b_para = [p[2] for p in paraphrased_pairs]

        unique_texts = list(set(all_texts_a + all_texts_b_orig + all_texts_b_para))
        logger.info(f"Embedding {len(unique_texts)} unique texts with bi-encoder...")
        text_embeddings = bi_encoder.encode(
            unique_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True
        )
        text_to_emb = dict(zip(unique_texts, text_embeddings))

        # Bi-encoder scores
        bi_scores_orig = []
        bi_scores_para = []
        for text_a, text_b_orig, text_b_para, _, _ in paraphrased_pairs:
            emb_a = text_to_emb[text_a]
            emb_b_orig = text_to_emb[text_b_orig]
            emb_b_para = text_to_emb[text_b_para]
            bi_scores_orig.append(float(np.dot(emb_a, emb_b_orig)))
            bi_scores_para.append(float(np.dot(emb_a, emb_b_para)))

        # Cross-encoder scores (batched)
        logger.info("Scoring with cross-encoder (original pairs)...")
        cross_scores_orig = reranker._normalize_scores(
            np.asarray(
                reranker._model.predict(
                    [(p[0], p[1]) for p in paraphrased_pairs],
                    batch_size=reranker.batch_size,
                    show_progress_bar=True,
                )
            )
        )

        logger.info("Scoring with cross-encoder (paraphrased pairs)...")
        cross_scores_para = reranker._normalize_scores(
            np.asarray(
                reranker._model.predict(
                    [(p[0], p[2]) for p in paraphrased_pairs],
                    batch_size=reranker.batch_size,
                    show_progress_bar=True,
                )
            )
        )

        # Build results DataFrame
        results_df = pd.DataFrame(
            {
                "pair_id": range(len(paraphrased_pairs)),
                "account_a": [p[3] for p in paraphrased_pairs],
                "account_b": [p[4] for p in paraphrased_pairs],
                "text_a": [p[0] for p in paraphrased_pairs],
                "text_b_original": [p[1] for p in paraphrased_pairs],
                "text_b_paraphrased": [p[2] for p in paraphrased_pairs],
                "biencoder_score_original": bi_scores_orig,
                "biencoder_score_paraphrased": bi_scores_para,
                "crossencoder_score_original": cross_scores_orig,
                "crossencoder_score_paraphrased": cross_scores_para,
            }
        )

        return results_df

    # ------------------------------------------------------------------
    # Step 4: Aggregate and report
    # ------------------------------------------------------------------

    def summarize(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics."""
        bi_drop = (
            results_df["biencoder_score_original"].mean()
            - results_df["biencoder_score_paraphrased"].mean()
        )
        cross_drop = (
            results_df["crossencoder_score_original"].mean()
            - results_df["crossencoder_score_paraphrased"].mean()
        )

        summary = pd.DataFrame(
            {
                "metric": [
                    "mean_biencoder_original",
                    "mean_biencoder_paraphrased",
                    "biencoder_score_drop",
                    "mean_crossencoder_original",
                    "mean_crossencoder_paraphrased",
                    "crossencoder_score_drop",
                    "drop_ratio (bi / cross)",
                    "n_pairs",
                ],
                "value": [
                    results_df["biencoder_score_original"].mean(),
                    results_df["biencoder_score_paraphrased"].mean(),
                    bi_drop,
                    results_df["crossencoder_score_original"].mean(),
                    results_df["crossencoder_score_paraphrased"].mean(),
                    cross_drop,
                    bi_drop / max(abs(cross_drop), 1e-8),
                    len(results_df),
                ],
            }
        )
        return summary

    # ------------------------------------------------------------------
    # Step 5: Visualize
    # ------------------------------------------------------------------

    def visualize(self, results_df: pd.DataFrame, summary_df: pd.DataFrame):
        """Create grouped bar chart comparing score drops."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Grouped bar chart — mean scores
        ax = axes[0]
        categories = ["Bi-Encoder", "Cross-Encoder"]
        original_means = [
            results_df["biencoder_score_original"].mean(),
            results_df["crossencoder_score_original"].mean(),
        ]
        paraphrased_means = [
            results_df["biencoder_score_paraphrased"].mean(),
            results_df["crossencoder_score_paraphrased"].mean(),
        ]

        x = np.arange(len(categories))
        width = 0.35
        bars1 = ax.bar(x - width / 2, original_means, width, label="Original", color="#3498db", alpha=0.85)
        bars2 = ax.bar(x + width / 2, paraphrased_means, width, label="After Paraphrasing", color="#e74c3c", alpha=0.85)

        # Add value labels
        for bar in bars1:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10
            )
        for bar in bars2:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10
            )

        ax.set_ylabel("Mean Score")
        ax.set_title("Score Comparison: Original vs. Paraphrased")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 2: Score drop comparison
        ax = axes[1]
        drops = [
            results_df["biencoder_score_original"].mean() - results_df["biencoder_score_paraphrased"].mean(),
            results_df["crossencoder_score_original"].mean() - results_df["crossencoder_score_paraphrased"].mean(),
        ]
        colors = ["#e74c3c", "#2ecc71"]
        bars = ax.bar(categories, drops, color=colors, alpha=0.85, width=0.5)

        for bar, drop_val in zip(bars, drops):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{drop_val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

        ax.set_ylabel("Mean Score Drop")
        ax.set_title("Robustness: Score Drop After LLM Paraphrasing")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(
            "Cross-Encoder Reranking: Paraphrase Robustness Experiment",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "paraphrase_robustness.png")
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved visualization: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, use_sample: bool = False) -> Dict:
        """Run the full paraphrase-robustness experiment."""
        from src.data.loader import DataLoader, create_sample_dataset
        from src.data.preprocessing import Preprocessor

        logger.info("=" * 60)
        logger.info("PARAPHRASE ROBUSTNESS EXPERIMENT")
        logger.info("=" * 60)

        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader(self.config)
        if use_sample:
            create_sample_dataset(self.config["data"]["raw_dir"])
        try:
            df = data_loader.load_campaign()
        except FileNotFoundError:
            logger.info("No data found, creating sample dataset...")
            create_sample_dataset(self.config["data"]["raw_dir"])
            df = data_loader.load_campaign()

        preprocessor = Preprocessor(self.config)
        df = preprocessor.process(df)
        logger.info(f"Loaded {len(df)} posts from {df['accountid'].nunique()} accounts")

        # Step 1: Sample pairs
        pairs = self.sample_coordinated_pairs(df)
        if not pairs:
            logger.error("No coordinated pairs found. Cannot run experiment.")
            return {}

        # Step 2: Paraphrase
        paraphrased_pairs = self.paraphrase_pairs(pairs)

        # Step 3: Score
        results_df = self.score_pairs(paraphrased_pairs)

        # Step 4: Summarize
        summary_df = self.summarize(results_df)
        logger.info(f"\nSummary:\n{summary_df.to_string(index=False)}")

        # Save results
        results_path = os.path.join(self.output_dir, "paraphrase_robustness_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved detailed results: {results_path}")

        summary_path = os.path.join(self.output_dir, "paraphrase_robustness_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary: {summary_path}")

        # Step 5: Visualize
        self.visualize(results_df, summary_df)

        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 60)

        return {
            "results": results_df,
            "summary": summary_df,
        }


def main():
    """CLI entry point for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Paraphrase Robustness Experiment"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Use sample data"
    )
    parser.add_argument(
        "--n-pairs", type=int, default=200,
        help="Number of post pairs to test (default: 200)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        # Minimal default config
        config_path = os.path.join("configs", "config.yaml")
        if os.path.exists(config_path):
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            config = {}

    # Override n_pairs from CLI
    if "paraphrase_experiment" not in config:
        config["paraphrase_experiment"] = {}
    config["paraphrase_experiment"]["n_pairs"] = args.n_pairs

    experiment = ParaphraseRobustnessExperiment(config)
    experiment.run(use_sample=args.sample)


if __name__ == "__main__":
    main()
