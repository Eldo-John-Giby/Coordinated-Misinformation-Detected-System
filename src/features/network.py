"""
Network-level feature extraction.

Computes account-level features based on their posting behavior,
network position, and interaction patterns.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class NetworkFeatures:
    """Extract network and account-level features."""

    def __init__(self, config: dict):
        self.config = config
        self.graph_config = config.get("graph", {})
        self.node_feature_types = self.graph_config.get("node_features", [
            "post_count", "avg_follower_count", "avg_following_count",
            "repost_ratio", "avg_hashtag_count", "avg_url_count",
            "avg_mention_count"
        ])

    def compute_account_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-account features from the post data.

        Features include:
        - post_count: number of posts
        - follower/following stats
        - repost_ratio: fraction of posts that are reposts
        - hashtag/url/mention counts
        - posting frequency
        - language distribution
        """
        account_feats = df.groupby("accountid").agg(
            post_count=("postid", "count"),
            unique_post_count=("postid", "nunique"),
            avg_follower_count=("follower_count", "mean"),
            max_follower_count=("follower_count", "max"),
            avg_following_count=("following_count", "mean"),
            max_following_count=("following_count", "max"),
            repost_ratio=("is_repost", "mean"),
            num_reposts_made=("repost_of_account",
                             lambda x: x.notna().sum()),
            avg_hashtag_count=("hashtag_list", lambda x: x.apply(len).mean()),
            avg_url_count=("url_list", lambda x: x.apply(len).mean()),
            avg_mention_count=("mention_list", lambda x: x.apply(len).mean()),
            num_unique_hashtags=("hashtag_list",
                                lambda x: len(set(h for hl in x for h in hl))),
            num_unique_urls=("url_list",
                           lambda x: len(set(u for ul in x for u in ul))),
            num_unique_mentions=("mention_list",
                               lambda x: len(set(m for ml in x for m in ml))),
            first_post_time=("post_time", "min"),
            last_post_time=("post_time", "max"),
            is_io=("is_control", lambda x: (~x).any()),
            has_replies=("in_reply_to_postid",
                        lambda x: x.notna().sum()),
            num_languages=("post_language", "nunique"),
        ).reset_index()

        # Compute derived features
        account_feats["posting_duration_days"] = (
            (account_feats["last_post_time"] - account_feats["first_post_time"])
            .dt.total_seconds() / 86400
        ).clip(lower=1)

        account_feats["posts_per_day"] = (
            account_feats["post_count"] / account_feats["posting_duration_days"]
        )

        account_feats["follower_following_ratio"] = (
            account_feats["avg_follower_count"] /
            account_feats["avg_following_count"].clip(lower=1)
        )

        account_feats["reply_ratio"] = (
            account_feats["has_replies"] / account_feats["post_count"]
        )

        account_feats["unique_hashtag_ratio"] = (
            account_feats["num_unique_hashtags"] /
            account_feats["post_count"].clip(lower=1)
        )

        account_feats["unique_url_ratio"] = (
            account_feats["num_unique_urls"] /
            account_feats["post_count"].clip(lower=1)
        )

        # Fill NaN
        account_feats = account_feats.fillna(0)

        logger.info(f"Computed account features for {len(account_feats)} accounts")
        return account_feats

    def compute_network_centrality(self, G: nx.Graph) -> pd.DataFrame:
        """Compute network centrality measures for each node."""
        if len(G.nodes) == 0:
            return pd.DataFrame()

        # Degree centrality
        degree_cent = nx.degree_centrality(G)

        # Betweenness centrality (approximate for large graphs)
        if len(G.nodes) < 5000:
            betweenness = nx.betweenness_centrality(G)
        else:
            betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))

        # Clustering coefficient
        clustering = nx.clustering(G)

        centrality_df = pd.DataFrame({
            "node_id": list(G.nodes()),
            "degree_centrality": [degree_cent[n] for n in G.nodes()],
            "betweenness_centrality": [betweenness[n] for n in G.nodes()],
            "clustering_coefficient": [clustering[n] for n in G.nodes()],
        })

        logger.info(f"Computed network centrality for {len(centrality_df)} nodes")
        return centrality_df

    def compute_repost_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features based on repost interactions."""
        reposts = df[df["repost_of_account"].notna()].copy()

        if len(reposts) == 0:
            return pd.DataFrame()

        # Build repost graph
        repost_edges = reposts[["accountid", "repost_of_account"]].dropna()
        repost_edges = repost_edges[repost_edges["accountid"] != repost_edges["repost_of_account"]]

        if len(repost_edges) == 0:
            return pd.DataFrame()

        G = nx.DiGraph()
        for _, row in repost_edges.iterrows():
            if G.has_edge(row["accountid"], row["repost_of_account"]):
                G[row["accountid"]][row["repost_of_account"]]["weight"] += 1
            else:
                G.add_edge(row["accountid"], row["repost_of_account"], weight=1)

        # In-degree = how many times this account is reposted
        in_degree = dict(G.in_degree(weight="weight"))
        # Out-degree = how many accounts this account reposts
        out_degree = dict(G.out_degree(weight="weight"))

        repost_features = pd.DataFrame({
            "accountid": list(G.nodes()),
            "repost_in_degree": [in_degree.get(n, 0) for n in G.nodes()],
            "repost_out_degree": [out_degree.get(n, 0) for n in G.nodes()],
        })

        return repost_features

    def build_interaction_network(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build an interaction network from replies and mentions.

        Edges represent reply or mention relationships between accounts.
        """
        G = nx.Graph()

        # Add reply edges
        replies = df[df["in_reply_to_accountid"].notna()].copy()
        for _, row in replies.iterrows():
            if row["accountid"] != row["in_reply_to_accountid"]:
                if G.has_edge(row["accountid"], row["in_reply_to_accountid"]):
                    G[row["accountid"]][row["in_reply_to_accountid"]]["reply_weight"] += 1
                else:
                    G.add_edge(row["accountid"], row["in_reply_to_accountid"],
                             reply_weight=1, mention_weight=0, repost_weight=0)

        # Add mention edges
        mentions_df = df[df["mention_list"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
        for _, row in mentions_df.iterrows():
            for mentioned in row["mention_list"]:
                if row["accountid"] != mentioned:
                    if G.has_edge(row["accountid"], mentioned):
                        G[row["accountid"]][mentioned]["mention_weight"] += 1
                    else:
                        G.add_edge(row["accountid"], mentioned,
                                 reply_weight=0, mention_weight=1, repost_weight=0)

        logger.info(f"Built interaction network: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges")
        return G

    def select_node_features(self) -> List[str]:
        """Select which node features to use based on config."""
        available = [
            "post_count", "avg_follower_count", "avg_following_count",
            "repost_ratio", "avg_hashtag_count", "avg_url_count",
            "avg_mention_count", "posts_per_day", "follower_following_ratio",
            "reply_ratio", "unique_hashtag_ratio", "unique_url_ratio"
        ]
        selected = [f for f in self.node_feature_types if f in available]
        return selected
