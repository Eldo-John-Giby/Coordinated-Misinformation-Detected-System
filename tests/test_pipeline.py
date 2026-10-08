"""
Basic tests for the coordination detection pipeline.

Run with: python -m pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader, create_sample_dataset
from src.data.preprocessing import Preprocessor
from src.features.semantic import SemanticFeatures
from src.features.temporal import TemporalFeatures
from src.features.edge_features import EdgeFeatureExtractor
from src.graph.builder import GraphBuilder
from src.graph.community import CommunityDetector
from src.evaluation.metrics import EvaluationMetrics


@pytest.fixture
def sample_config(tmp_path):
    """Create a test configuration.

    Uses tmp_path so tests NEVER write into the real data/raw directory
    (writing there would regenerate the synthetic-data marker and cause
    real pipeline runs to be mis-tagged as synthetic)."""
    return {
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": "data/processed",
            "campaign": "test",
            "max_files_per_campaign": 1,
            "max_posts": 100,
            "subsample_fraction": None,
            "random_seed": 42,
        },
        "preprocessing": {
            "lowercase": True,
            "remove_urls_from_text": True,
            "min_text_length": 3,
            "drop_malformed": True,
        },
        "nlp": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "batch_size": 32,
        },
        "similarity": {
            "method": "sklearn",
            "k_neighbors": 5,
            "semantic_threshold": 0.5,
        },
        "temporal": {
            "tau": 3600,
            "burst_window": 300,
            "burst_min_posts": 2,
        },
        "coordination": {
            "edge_threshold": 0.2,
            "weights": {
                "semantic": 0.35,
                "temporal": 0.25,
                "url": 0.15,
                "hashtag": 0.10,
                "mention": 0.05,
                "repost": 0.10,
            },
            "max_edges_per_node": 20,
            "min_edge_score": 0.1,
        },
        "graph": {
            "use_node_features": True,
            "node_features": ["post_count", "repost_ratio"],
        },
        "models": {
            "gcn": {"hidden_channels": 32, "num_layers": 2, "epochs": 5},
            "graphsage": {"hidden_channels": 32, "num_layers": 2, "epochs": 5},
            "baseline": {},
        },
        "evaluation": {},
        "explainability": {},
        "visualization": {
            "output_dir": "results/figures_test",
            "format": "png",
            "dpi": 72,
        },
        "experiments": {
            "output_dir": "results_test",
        },
    }


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    records = []
    accounts_io = [f"IO_{i}" for i in range(10)]
    accounts_ctrl = [f"CTRL_{i}" for i in range(10)]

    for acc in accounts_io:
        for j in range(5):
            records.append({
                "postid": f"post_{acc}_{j}",
                "post_text": f"IO post about politics topic {j}",
                "post_time": pd.Timestamp("2019-06-01") + pd.Timedelta(hours=j),
                "accountid": acc,
                "is_control": False,
                "follower_count": np.random.randint(10, 100),
                "following_count": np.random.randint(50, 200),
                "is_repost": False,
                "reposted_accountid": None,
                "hashtags": "politics,election",
                "urls": "url1",
                "account_mentions": None,
                "url_list": ["url1"],
                "hashtag_list": ["politics", "election"],
                "mention_list": [],
                "account_is_control": False,
                "node_id": accounts_io.index(acc),
            })

    for acc in accounts_ctrl:
        for j in range(5):
            records.append({
                "postid": f"post_{acc}_{j}",
                "post_text": f"Normal post about sports topic {j}",
                "post_time": pd.Timestamp("2019-06-01") + pd.Timedelta(days=j),
                "accountid": acc,
                "is_control": True,
                "follower_count": np.random.randint(100, 1000),
                "following_count": np.random.randint(100, 500),
                "is_repost": False,
                "reposted_accountid": None,
                "hashtags": "sports,news",
                "urls": None,
                "account_mentions": None,
                "url_list": [],
                "hashtag_list": ["sports", "news"],
                "mention_list": [],
                "account_is_control": True,
                "node_id": len(accounts_io) + accounts_ctrl.index(acc),
            })

    return pd.DataFrame(records)


class TestDataLoader:
    """Test data loading and sample creation."""

    def test_create_sample_dataset(self, sample_config):
        path = create_sample_dataset(
            output_dir=sample_config["data"]["raw_dir"],
            num_accounts=20,
            posts_per_account=5
        )
        assert os.path.exists(path)

        df = pd.read_csv(path)
        assert len(df) > 0
        assert "accountid" in df.columns
        assert "is_control" in df.columns
        assert "post_text" in df.columns

    def test_loader_discovery(self, sample_config):
        create_sample_dataset(
            output_dir=sample_config["data"]["raw_dir"],
            num_accounts=10,
            posts_per_account=3
        )
        loader = DataLoader(sample_config)
        files = loader.discover_files()
        assert len(files) > 0


class TestPreprocessing:
    """Test preprocessing pipeline."""

    def test_validate_schema(self, sample_config, sample_df):
        preprocessor = Preprocessor(sample_config)
        df = preprocessor.validate_schema(sample_df)
        assert len(df) == len(sample_df)

    def test_parse_timestamps(self, sample_config, sample_df):
        preprocessor = Preprocessor(sample_config)
        df = preprocessor.parse_timestamps(sample_df)
        assert pd.api.types.is_datetime64_any_dtype(df["post_time"])

    def test_extract_hashtags(self, sample_config, sample_df):
        preprocessor = Preprocessor(sample_config)
        df = preprocessor.extract_hashtags(sample_df)
        assert "hashtag_list" in df.columns
        assert all(isinstance(x, list) for x in df["hashtag_list"])

    def test_build_account_mappings(self, sample_config, sample_df):
        preprocessor = Preprocessor(sample_config)
        df = preprocessor.build_account_mappings(sample_df)
        assert "node_id" in df.columns
        assert len(preprocessor.account_to_node_id) > 0


class TestSemanticFeatures:
    """Test semantic feature extraction."""

    def test_compute_similarity(self, sample_config):
        sf = SemanticFeatures(sample_config)
        emb_a = np.random.randn(384).astype(np.float32)
        emb_b = np.random.randn(384).astype(np.float32)

        # Normalize
        emb_a = emb_a / np.linalg.norm(emb_a)
        emb_b = emb_b / np.linalg.norm(emb_b)

        score = sf.compute_post_pair_similarity(emb_a, emb_b)
        assert -1 <= score <= 1

    def test_account_pair_score(self, sample_config):
        sf = SemanticFeatures(sample_config)
        emb_a = np.random.randn(5, 384).astype(np.float32)
        emb_b = np.random.randn(3, 384).astype(np.float32)

        scores = sf.compute_account_pair_semantic_score(emb_a, emb_b)
        assert "semantic_similarity" in scores
        assert -1 <= scores["semantic_similarity"] <= 1


class TestTemporalFeatures:
    """Test temporal feature extraction."""

    def test_temporal_proximity(self, sample_config):
        tf = TemporalFeatures(sample_config)
        times_a = pd.Series([
            pd.Timestamp("2019-06-01 10:00:00"),
            pd.Timestamp("2019-06-01 10:01:00"),
        ])
        times_b = pd.Series([
            pd.Timestamp("2019-06-01 10:00:30"),
            pd.Timestamp("2019-06-01 10:02:00"),
        ])

        result = tf.compute_temporal_proximity(times_a, times_b)
        assert "temporal_score" in result
        assert 0 <= result["temporal_score"] <= 1


class TestEdgeFeatures:
    """Test edge feature extraction."""

    def test_shared_url_score(self, sample_config):
        ef = EdgeFeatureExtractor(sample_config)

        # Same URLs
        score = ef.compute_shared_url_score({"url1", "url2"}, {"url1", "url3"})
        assert score > 0

        # No overlap
        score = ef.compute_shared_url_score({"url1"}, {"url2"})
        assert score == 0

        # Empty
        score = ef.compute_shared_url_score(set(), set())
        assert score == 0

    def test_coordination_score(self, sample_config):
        ef = EdgeFeatureExtractor(sample_config)
        features = {
            "semantic_similarity": 0.8,
            "temporal_score": 0.6,
            "shared_url_score": 0.5,
            "shared_hashtag_score": 0.4,
            "shared_mention_score": 0.3,
            "repost_score": 0.2,
        }
        score = ef.compute_coordination_score(features)
        assert 0 <= score <= 1


class TestGraphBuilder:
    """Test graph construction."""

    def test_build_graph(self, sample_config, sample_df):
        import networkx as nx

        builder = GraphBuilder(sample_config)

        # Create minimal edge features
        edge_df = pd.DataFrame({
            "account_a": ["IO_0", "IO_0", "IO_1"],
            "account_b": ["IO_1", "IO_2", "IO_2"],
            "coordination_score": [0.8, 0.6, 0.4],
            "semantic_similarity": [0.9, 0.7, 0.5],
            "temporal_score": [0.8, 0.6, 0.3],
            "shared_url_score": [0.5, 0.3, 0.1],
            "shared_hashtag_score": [0.6, 0.4, 0.2],
            "shared_mention_score": [0.3, 0.2, 0.1],
            "repost_score": [0.1, 0.1, 0.0],
            "shared_url_count": [1, 0, 0],
            "shared_hashtag_count": [2, 1, 0],
            "shared_mention_count": [0, 0, 0],
        })

        # Create minimal account features
        account_df = pd.DataFrame({
            "accountid": ["IO_0", "IO_1", "IO_2", "CTRL_0"],
            "node_id": [0, 1, 2, 3],
            "post_count": [5, 5, 5, 5],
            "avg_follower_count": [50, 50, 50, 500],
            "avg_following_count": [100, 100, 100, 500],
            "repost_ratio": [0.1, 0.1, 0.1, 0.0],
            "is_io": [True, True, True, False],
        })

        G = builder.build_networkx_graph(edge_df, account_df)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() > 0


class TestEvaluation:
    """Test evaluation metrics."""

    def test_classification_metrics(self, sample_config):
        em = EvaluationMetrics(sample_config)
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])

        metrics = em.compute_classification_metrics(y_true, y_pred)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_community_metrics(self, sample_config):
        em = EvaluationMetrics(sample_config)
        true_labels = np.array([0, 0, 1, 1])
        pred_labels = np.array([0, 0, 1, 1])

        metrics = em.compute_community_metrics(true_labels, pred_labels)
        assert "adjusted_rand_index" in metrics
        assert "normalized_mutual_info" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
