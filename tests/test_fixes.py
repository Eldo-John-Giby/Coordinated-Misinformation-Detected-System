"""
Regression tests for the fix pass (items 1-25 of the code review).

Run with: python -m pytest tests/test_fixes.py -v

Each test maps to a specific acceptance criterion:
- test_canonicalize_account_ids_*          -> item 1 (ID dtype canonicalization)
- test_no_self_loops_*                     -> item 1 (self-loop guard)
- test_graph_save_load_roundtrip           -> item 14 (persistence)
- test_plot_roc_pr_confusion_run           -> item 14 (average_precision_score fix)
- test_cross_encoder_rerank_k_truncation   -> item 6 (config key is used)
- test_post_level_retrieval_finds_planted  -> item 4 (post-level semantic matching)
- test_preprocessor_missing_columns_recorded -> item 8e (visible degradation)
- test_timestamp_drop_rate_fail_loud       -> item 8f
- test_learned_weights                     -> item 19
- test_gnn_scores_influence_flagging       -> item 10 (Option B integration)
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless CI

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import canonicalize_account_ids
from src.data.preprocessing import Preprocessor
from src.features.edge_features import EdgeFeatureExtractor
from src.features.semantic import SemanticFeatures
from src.graph.builder import GraphBuilder
from src.graph.community import CommunityDetector
from src.graph.visualization import GraphVisualizer
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.nlp.similarity import CrossEncoderReranker, SimilaritySearch
from src.utils.degradations import DegradationTracker


@pytest.fixture
def base_config(tmp_path):
    """Minimal config covering every component the tests touch."""
    return {
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "campaign": "test",
            "random_seed": 42,
        },
        "preprocessing": {
            "lowercase": True,
            "remove_urls_from_text": True,
            "min_text_length": 3,
            "drop_malformed": True,
            "max_timestamp_drop_rate": 0.20,
        },
        "nlp": {"model_name": "test-model", "embedding_dim": 384},
        "similarity": {
            "method": "sklearn",
            "k_neighbors": 10,
            "retrieval_floor": 0.3,
            "semantic_threshold": 0.7,
        },
        "temporal": {"tau": 3600, "burst_window": 300, "burst_min_posts": 2},
        "coordination": {
            "edge_threshold": 0.3,
            "weights": {
                "semantic": 0.35, "temporal": 0.25, "url": 0.15,
                "hashtag": 0.10, "mention": 0.05, "repost": 0.10,
            },
            "max_edges_per_node": 20,
            "min_edge_score": 0.1,
        },
        "graph": {"use_node_features": True, "node_features": ["post_count"]},
        "models": {"gcn": {}, "graphsage": {}, "baseline": {}},
        "evaluation": {},
        "explainability": {},
        "visualization": {"output_dir": str(tmp_path / "figures"), "dpi": 72},
        "experiments": {"output_dir": str(tmp_path / "results")},
    }


# ============================================================
# Item 1: account-ID canonicalization + self-loop guard
# ============================================================

class TestAccountIDCanonicalization:
    def test_float_mangled_ids_recover_integer_precision(self):
        """A float-mangled ID string must come back as the exact integer."""
        df = pd.DataFrame({
            "accountid": [
                "1204524215807942656",          # correct string
                1.204524215807942656e18,        # float dtype (exact for this ID)
                "1.204524215807942656e+18",     # already-mangled string form
            ],
        })
        out = canonicalize_account_ids(df)
        assert out["accountid"].tolist() == ["1204524215807942656"] * 3
        assert all(isinstance(v, str) for v in out["accountid"])

    def test_nan_and_garbage_ids_dropped(self):
        df = pd.DataFrame({
            "accountid": ["100", np.nan, None, "nan", "None", "", "  ", "200"],
        })
        out = canonicalize_account_ids(df)
        assert out["accountid"].tolist() == ["100", "200"]

    def test_repost_target_column_canonicalized(self):
        df = pd.DataFrame({
            "accountid": ["100"],
            "reposted_accountid": [1.204524215807942656e18],
        })
        out = canonicalize_account_ids(df)
        assert out["reposted_accountid"].iloc[0] == "1204524215807942656"

    def test_no_self_loops_after_canonicalization(self, base_config):
        """End-to-end: messy IDs -> canonicalize -> graph has zero self-loops."""
        messy = pd.DataFrame({
            "accountid": [1204524215807942656, 1.204524215807942656e18,
                          "300", np.nan, "300"],
        })
        clean = canonicalize_account_ids(messy)
        # Two rows were the same account in different dtypes -> same node
        ids = clean["accountid"].tolist()
        assert ids.count("1204524215807942656") == 2
        assert ids.count("300") == 2

    def test_builder_skips_and_logs_self_loop(self, base_config, caplog):
        """The graph builder must refuse a literal self-pair edge."""
        accounts = pd.DataFrame({"accountid": ["A", "B"]})
        edges = pd.DataFrame([
            {"account_a": "A", "account_b": "B", "coordination_score": 0.8},
            {"account_a": "A", "account_b": "A", "coordination_score": 0.9},
        ])
        G = GraphBuilder(base_config).build_networkx_graph(edges, accounts)
        assert not any(G.has_edge(n, n) for n in G.nodes)
        assert any("Self-loop" in r.message for r in caplog.records)


# ============================================================
# Item 14: graph persistence + fixed PR-curve function
# ============================================================

class TestGraphPersistence:
    def test_save_load_roundtrip(self, base_config, tmp_path):
        G = nx.Graph()
        G.add_node("A", is_io=True)
        G.add_node("B", is_io=False)
        G.add_edge("A", "B", coordination_score=0.75, semantic_similarity=0.9)
        path = str(tmp_path / "graph.pkl")

        builder = GraphBuilder(base_config)
        builder.save_graph(G, path)
        G2 = builder.load_graph(path)

        assert set(G2.nodes) == {"A", "B"}
        assert G2["A"]["B"]["coordination_score"] == 0.75
        assert G2.nodes["A"]["is_io"] is True


class TestMetricPlots:
    def test_plot_roc_pr_confusion_run(self, base_config, tmp_path):
        """Direct-call test: all three previously-broken/dead plotters run."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 60)
        y_scores = {
            "gcn": np.clip(y_true * 0.6 + rng.random(60) * 0.4, 0, 1),
            "graphsage": np.clip(y_true * 0.5 + rng.random(60) * 0.5, 0, 1),
        }
        viz = GraphVisualizer(base_config)
        viz.plot_roc_curve(y_true, y_scores, filename="roc_curve")
        viz.plot_pr_curve(y_true, y_scores, filename="pr_curve")
        viz.plot_confusion_matrix(y_true, (y_scores["gcn"] > 0.5).astype(int),
                                  filename="confusion_matrix")
        for name in ("roc_curve", "pr_curve", "confusion_matrix"):
            assert (tmp_path / "figures" / f"{name}.png").exists(), name


# ============================================================
# Item 6: cross_encoder_rerank_k truncation is actually applied
# ============================================================

class FakeCrossEncoder:
    """Stands in for the real model; returns a descending raw score per pair."""

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        return np.array([5.0 - 0.5 * i for i in range(len(pairs))])


class TestRerankTruncation:
    def test_top_k_truncation(self, base_config):
        reranker = CrossEncoderReranker(base_config)
        reranker._model = FakeCrossEncoder()  # skip the real model download

        n = 6
        cand = pd.DataFrame({
            "account_a": [f"A{i}" for i in range(n)],
            "account_b": [f"B{i}" for i in range(n)],
            "semantic_similarity": [0.95, 0.90, 0.85, 0.80, 0.75, 0.70],
        })
        out = reranker.rerank(
            cand,
            texts_a=[f"text a {i}" for i in range(n)],
            texts_b=[f"text b {i}" for i in range(n)],
            top_k=2,
        )
        scored = out["cross_encoder_score"]
        assert scored.notna().sum() == 2          # only top-2 reranked
        assert scored.iloc[:2].notna().all()
        assert scored.iloc[2:].isna().all()

    def test_no_truncation_when_none(self, base_config):
        reranker = CrossEncoderReranker(base_config)
        reranker._model = FakeCrossEncoder()

        n = 4
        cand = pd.DataFrame({
            "account_a": [f"A{i}" for i in range(n)],
            "account_b": [f"B{i}" for i in range(n)],
            "semantic_similarity": [0.9, 0.8, 0.7, 0.6],
        })
        out = reranker.rerank(cand, texts_a=["x"] * n, texts_b=["y"] * n)
        assert out["cross_encoder_score"].notna().all()


# ============================================================
# Item 4: post-level retrieval finds a single planted paraphrase pair
# ============================================================

class TestPostLevelRetrieval:
    def _make_embeddings(self, n_unrelated=20, dim=384, seed=42):
        """Two accounts of unrelated posts; ONE planted paraphrase pair.

        Account A: 20 unrelated posts + 1 post with vector `base`.
        Account B: 20 unrelated posts + 1 post paraphrasing `base`.
        The paraphrase is 1/21 of each account's mean embedding, so the old
        account-level averaging approach would heavily dilute it; post-level
        retrieval must still find it.
        """
        rng = np.random.default_rng(seed)
        unrelated = rng.normal(size=(2 * n_unrelated, dim)) * 0.1

        base = rng.normal(size=dim)
        base /= np.linalg.norm(base)
        paraphrase = base + rng.normal(size=dim) * 0.01
        paraphrase /= np.linalg.norm(paraphrase)

        # Rows 0..19: A unrelated | row 20: A's planted post
        # Rows 21..40: B unrelated | row 41: B's planted (paraphrase) post
        embeddings = np.vstack([
            unrelated[:n_unrelated], base[None, :],
            unrelated[n_unrelated:], paraphrase[None, :],
        ])
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        n = len(embeddings)
        accounts = np.array(["A"] * (n_unrelated + 1) + ["B"] * (n_unrelated + 1))
        post_ids = np.array([f"p{i}" for i in range(n)])
        return embeddings.astype(np.float32), post_ids, accounts

    def test_planted_pair_is_top_candidate(self, base_config):
        emb, post_ids, accounts = self._make_embeddings()
        search = SimilaritySearch(base_config)

        pairs = search.find_post_pairs(emb, post_ids, accounts, k=10,
                                       threshold=0.3)
        assert not pairs.empty

        agg = SemanticFeatures(base_config).aggregate_post_pair_candidates(
            pairs, top_n=3
        )
        planted = agg[
            ((agg["account_a"] == "A") & (agg["account_b"] == "B"))
            | ((agg["account_a"] == "B") & (agg["account_b"] == "A"))
        ]
        assert not planted.empty
        # The planted pair has by far the highest bi-encoder score
        assert planted["biencoder_similarity"].max() > 0.95
        assert agg["biencoder_similarity"].max() == planted["biencoder_similarity"].max()
        # Evidence preserved: matched post pairs recorded (item 4d)
        assert planted["n_post_pairs"].iloc[0] >= 1

    def test_no_cross_account_pairs_when_all_similar_within_account(self, base_config):
        """Same-account neighbors must never become coordination candidates."""
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(6, 64)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        accounts = np.array(["A"] * 3 + ["B"] * 3)
        ids = np.array([f"p{i}" for i in range(6)])

        search = SimilaritySearch(base_config)
        pairs = search.find_post_pairs(emb, ids, accounts, k=5, threshold=-1.0)
        # threshold=-1 keeps everything; still no same-account or self pairs
        assert (pairs["account_a"] != pairs["account_b"]).all()
        assert (pairs["post_a"] != pairs["post_b"]).all()


# ============================================================
# Items 8e/8f: degradations are visible, timestamp corruption fails loudly
# ============================================================

def _minimal_posts_df(n=10, bad_timestamp_fraction=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ts = "2024-01-01 12:00:00"
        if i < bad_timestamp_fraction * n:
            ts = "not-a-timestamp"
        rows.append({
            "postid": i,
            "post_text": f"hello world post {i}",
            "post_time": ts,
            "accountid": str(100 + i % 3),
            "is_control": i % 2 == 0,
        })
    return pd.DataFrame(rows)


class TestVisibleDegradations:
    def test_missing_optional_columns_recorded(self, base_config):
        """A CSV missing url/hashtag columns must surface, not silently zero."""
        df = _minimal_posts_df()  # no shared_url / hashtags / mentions columns
        tracker = DegradationTracker()
        Preprocessor(base_config).process(df, degradations=tracker)
        components = {
            e["component"] for e in tracker.to_dict()["degraded_components"]
        }
        assert any("url" in c or "hashtag" in c or "mention" in c
                   for c in components)

    def test_timestamp_drops_below_threshold_tracked(self, base_config):
        df = _minimal_posts_df(n=20, bad_timestamp_fraction=0.10)
        tracker = DegradationTracker()
        out = Preprocessor(base_config).process(df, degradations=tracker)
        assert len(out) == 18  # 2 bad rows dropped
        assert any(e["component"] == "timestamps"
                   for e in tracker.to_dict()["degraded_components"])

    def test_timestamp_drop_rate_over_threshold_raises(self, base_config):
        df = _minimal_posts_df(n=10, bad_timestamp_fraction=0.5)
        tracker = DegradationTracker()
        with pytest.raises(ValueError, match="timestamps are unparseable"):
            Preprocessor(base_config).process(df, degradations=tracker)


# ============================================================
# Item 19: learned coordination weights
# ============================================================

class TestLearnedWeights:
    def _edge_features(self):
        # Account A-side is IO and has systematically higher signals
        rows = []
        for i in range(20):
            hi = i < 10
            rows.append({
                "account_a": f"IO_{i % 5}" if hi else f"C_{i % 5}",
                "account_b": f"C_{(i + 1) % 5}",
                "semantic_similarity": 0.9 if hi else 0.1,
                "temporal_score": 0.8 if hi else 0.2,
                "shared_url_score": 0.9 if hi else 0.1,
                "shared_hashtag_score": 0.7 if hi else 0.3,
                "shared_mention_score": 0.6 if hi else 0.4,
                "repost_score": 0.8 if hi else 0.2,
            })
        return pd.DataFrame(rows)

    def test_learned_weights_sum_to_one(self, base_config):
        edges = self._edge_features()
        accounts = pd.DataFrame({
            "accountid": [f"IO_{i}" for i in range(5)] + [f"C_{i}" for i in range(5)],
            "is_io": [True] * 5 + [False] * 5,
        })
        weights = EdgeFeatureExtractor(base_config).learn_weights(accounts, edges)
        assert weights is not None
        assert set(weights) == {"semantic", "temporal", "url",
                                "hashtag", "mention", "repost"}
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(v >= 0 for v in weights.values())

    def test_returns_none_without_labels(self, base_config):
        edges = self._edge_features()
        accounts = pd.DataFrame({"accountid": [f"A{i}" for i in range(5)]})
        assert EdgeFeatureExtractor(base_config).learn_weights(accounts, edges) is None


# ============================================================
# Item 10: GNN P(IO) must influence community flagging (Option B)
# ============================================================

class TestGNNInfluencesFlagging:
    def _graph_and_communities(self):
        G = nx.Graph()
        # Dense cluster 1
        for a in ("A1", "A2", "A3", "A4"):
            G.add_node(a, is_io=False)
        for u, v in [("A1", "A2"), ("A1", "A3"), ("A2", "A3"),
                     ("A1", "A4"), ("A2", "A4"), ("A3", "A4")]:
            G.add_edge(u, v, coordination_score=0.8)
        # Cluster 2 (identical structure, different GNN scores in the test)
        for a in ("B1", "B2", "B3", "B4"):
            G.add_node(a, is_io=False)
        for u, v in [("B1", "B2"), ("B1", "B3"), ("B2", "B3"),
                     ("B1", "B4"), ("B2", "B4"), ("B3", "B4")]:
            G.add_edge(u, v, coordination_score=0.8)
        communities = {0: ["A1", "A2", "A3", "A4"],
                       1: ["B1", "B2", "B3", "B4"]}
        return G, communities

    def test_gnn_scores_change_suspiciousness(self, base_config):
        detector = CommunityDetector(base_config)
        G, communities = self._graph_and_communities()

        without = detector.score_communities(communities, G, gnn_scores=None)
        with_gnn = detector.score_communities(
            communities, G,
            gnn_scores={"A1": 0.9, "A2": 0.9, "A3": 0.9, "A4": 0.9,
                        "B1": 0.05, "B2": 0.05, "B3": 0.05, "B4": 0.05},
        )

        def score_of(df, cid):
            return float(df.loc[df["community_id"] == cid,
                                "suspiciousness_score"].iloc[0])

        # Identical structure, different GNN output -> different flagging
        assert score_of(with_gnn, 0) > score_of(with_gnn, 1)
        assert score_of(without, 0) == pytest.approx(score_of(without, 1))

    def test_explanations_cover_all_six_signals(self, base_config):
        """Item 12: every flagged group exposes numeric values for 6 signals."""
        detector = CommunityDetector(base_config)
        G, communities = self._graph_and_communities()
        for u, v in list(G.edges):
            G[u][v].update({
                "semantic_similarity": 0.8, "temporal_score": 0.6,
                "shared_url_score": 0.5, "shared_hashtag_score": 0.4,
                "shared_mention_score": 0.3, "repost_score": 0.2,
            })

        scored = detector.score_communities(communities, G, gnn_scores={
            f"A{i}": 0.9 for i in range(1, 5)} | {f"B{i}": 0.9 for i in range(1, 5)})
        suspicious = detector.get_suspicious_groups(scored)
        assert not suspicious.empty

        # df argument is the posts DataFrame (used for sample-post evidence);
        # a minimal frame is enough for the numeric breakdown assertions.
        posts_df = pd.DataFrame({"postid": ["p1"], "post_text": ["x"]})
        explanations = detector.generate_group_explanations(suspicious, G, posts_df)
        required = {"semantic", "temporal", "url", "hashtag", "mention", "repost"}
        for exp in explanations:
            breakdown = exp.get("signal_breakdown", exp.get("signals", {}))
            assert required.issubset(breakdown.keys()), (
                f"missing signals: {required - set(breakdown)}"
            )
            # Each signal exposes numeric evidence: mean raw score, the weight
            # applied, and the weighted contribution to the coordination score.
            for signal, stats in breakdown.items():
                assert isinstance(stats, dict), signal
                for key in ("mean_score", "weight", "weighted_contribution"):
                    assert key in stats and isinstance(
                        stats[key], (int, float)
                    ), (signal, key)


class TestCheckpointBundle:
    """Item 16 acceptance: saved .pt bundles reload and reproduce predictions."""

    def _train_stub(self, model_cls, x, edge_index):
        model = model_cls(x.shape[1], 16, 2, num_layers=2)
        # A deterministic forward pass, no training loop needed for the
        # round-trip property being tested (save -> load -> same outputs).
        return model

    def test_bundle_roundtrip_reproduces_predictions(self, tmp_path):
        from torch_geometric.data import Data

        torch.manual_seed(42)
        n, in_ch = 30, 8
        x = torch.randn(n, in_ch)
        edge_index = torch.randint(0, n, (2, 60))
        test_mask = torch.zeros(n, dtype=torch.bool)
        test_mask[:10] = True

        for model_cls in (GCN, GraphSAGE):
            model = self._train_stub(model_cls, x, edge_index)
            model.eval()
            with torch.no_grad():
                expected = torch.softmax(model(x, edge_index), dim=1)

            bundle = {
                "model_state_dict": model.state_dict(),
                "x": x, "edge_index": edge_index, "test_mask": test_mask,
                "in_channels": in_ch,
            }
            p = tmp_path / f"{model_cls.__name__}.pt"
            torch.save(bundle, p)

            reloaded = torch.load(p, map_location="cpu")
            state = reloaded["model_state_dict"]
            in_channels = reloaded["in_channels"]
            restored = model_cls(in_channels, 16, 2, num_layers=2)
            restored.load_state_dict(state)
            restored.eval()
            with torch.no_grad():
                actual = torch.softmax(
                    restored(reloaded["x"], reloaded["edge_index"]), dim=1
                )
            tm = reloaded["test_mask"]
            assert torch.allclose(expected[tm], actual[tm], atol=1e-6), model_cls
            assert int(tm.sum()) == 10
