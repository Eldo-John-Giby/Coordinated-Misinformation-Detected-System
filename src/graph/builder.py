"""
Graph construction from coordination evidence.

NODE = social media account
EDGE = evidence of coordination between two accounts

Edges are created ONLY when coordination evidence exceeds a threshold.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build coordination graphs from account-pair features."""

    def __init__(self, config: dict):
        self.config = config
        self.coord_config = config.get("coordination", {})
        self.graph_config = config.get("graph", {})
        self.edge_threshold = self.coord_config.get("edge_threshold", 0.3)
        self.max_edges_per_node = self.coord_config.get("max_edges_per_node", 50)
        self.min_edge_score = self.coord_config.get("min_edge_score", 0.1)

        self.account_to_node: Dict[str, int] = {}
        self.node_to_account: Dict[int, str] = {}

    def build_networkx_graph(self,
                             edge_features_df: pd.DataFrame,
                             account_features_df: pd.DataFrame
                             ) -> nx.Graph:
        """
        Build a NetworkX graph from edge and account features.

        Args:
            edge_features_df: DataFrame with columns account_a, account_b,
                            coordination_score, and other edge features
            account_features_df: DataFrame with per-account features

        Returns:
            NetworkX Graph with coordination edges
        """
        G = nx.Graph()

        # Add nodes with features
        for _, row in account_features_df.iterrows():
            node_id = row.get("node_id", row.get("accountid"))
            attrs = {
                "accountid": row["accountid"],
                "is_io": bool(row.get("is_io", row.get("account_is_control", False))
                           if "is_io" in row else False),
                "post_count": row.get("post_count", 0),
                "follower_count": row.get("avg_follower_count", 0),
                "following_count": row.get("avg_following_count", 0),
                "repost_ratio": row.get("repost_ratio", 0),
            }
            G.add_node(row["accountid"], **attrs)

            # Track mappings
            if "node_id" in row:
                self.account_to_node[row["accountid"]] = int(row["node_id"])
                self.node_to_account[int(row["node_id"])] = row["accountid"]

        # Filter edges by threshold
        edges_df = edge_features_df[
            edge_features_df["coordination_score"] >= self.edge_threshold
        ].copy()

        logger.info(f"Edges above threshold {self.edge_threshold}: "
                   f"{len(edges_df)} / {len(edge_features_df)}")

        # Add edges with features
        for _, row in edges_df.iterrows():
            if row["account_a"] in G.nodes and row["account_b"] in G.nodes:
                attrs = {
                    "coordination_score": row["coordination_score"],
                    "semantic_similarity": row.get("semantic_similarity", 0),
                    "temporal_score": row.get("temporal_score", 0),
                    "shared_url_score": row.get("shared_url_score", 0),
                    "shared_hashtag_score": row.get("shared_hashtag_score", 0),
                    "shared_mention_score": row.get("shared_mention_score", 0),
                    "repost_score": row.get("repost_score", 0),
                    "shared_url_count": row.get("shared_url_count", 0),
                    "shared_hashtag_count": row.get("shared_hashtag_count", 0),
                    "shared_mention_count": row.get("shared_mention_count", 0),
                }
                G.add_edge(row["account_a"], row["account_b"], **attrs)

        # Limit edges per node (for scalability)
        if self.max_edges_per_node:
            self._prune_edges(G)

        logger.info(f"Built NetworkX graph: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges")
        return G

    def _prune_edges(self, G: nx.Graph):
        """Remove lowest-weight edges to limit degree per node."""
        for node in list(G.nodes()):
            if G.degree(node) > self.max_edges_per_node:
                edges = sorted(
                    G.edges(node, data=True),
                    key=lambda x: x[2].get("coordination_score", 0),
                    reverse=True
                )
                to_remove = edges[self.max_edges_per_node:]
                for u, v, _ in to_remove:
                    G.remove_edge(u, v)

    def build_pyg_data(self, G: nx.Graph,
                       node_features: np.ndarray,
                       node_labels: np.ndarray,
                       edge_features_list: Optional[List] = None
                       ) -> Data:
        """
        Convert a NetworkX graph to PyTorch Geometric Data object.

        Args:
            G: NetworkX graph
            node_features: (num_nodes, num_features) array
            node_labels: (num_nodes,) array of labels (0=control, 1=IO)
            edge_features_list: Optional list of edge feature vectors

        Returns:
            PyTorch Geometric Data object
        """
        # Create node mapping for contiguous indices
        nodes = sorted(G.nodes())
        node_map = {n: i for i, n in enumerate(nodes)}

        # Node features tensor
        x = torch.tensor(node_features, dtype=torch.float)

        # Node labels tensor
        y = torch.tensor(node_labels, dtype=torch.long)

        # Edge index (2, num_edges)
        edge_index = []
        edge_attr = []
        for u, v, data in G.edges(data=True):
            if u in node_map and v in node_map:
                edge_index.append([node_map[u], node_map[v]])
                edge_index.append([node_map[v], node_map[u]])

                # Edge features
                feat = [
                    data.get("coordination_score", 0),
                    data.get("semantic_similarity", 0),
                    data.get("temporal_score", 0),
                    data.get("shared_url_score", 0),
                    data.get("shared_hashtag_score", 0),
                    data.get("shared_mention_score", 0),
                    data.get("repost_score", 0),
                ]
                edge_attr.append(feat)
                edge_attr.append(feat)  # Same features for reverse edge

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 7), dtype=torch.float)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(nodes),
        )

        # Store node mapping as metadata
        data.node_map = node_map
        data.inv_node_map = {v: k for k, v in node_map.items()}

        logger.info(f"Built PyG Data: {data.num_nodes} nodes, "
                   f"{data.num_edges} edges, "
                   f"node_features={data.x.shape}, "
                   f"edge_features={data.edge_attr.shape}")
        return data

    def get_graph_statistics(self, G: nx.Graph) -> Dict:
        """Compute graph statistics."""
        if G.number_of_nodes() == 0:
            return {"num_nodes": 0, "num_edges": 0}

        stats = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_degree": sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1),
            "num_components": nx.number_connected_components(G),
            "avg_clustering": nx.average_clustering(G),
        }

        if G.number_of_edges() > 0:
            # Edge weight statistics
            weights = [d.get("coordination_score", 0) for _, _, d in G.edges(data=True)]
            stats["avg_edge_weight"] = np.mean(weights)
            stats["max_edge_weight"] = np.max(weights)
            stats["min_edge_weight"] = np.min(weights)

            # Degree statistics
            degrees = [d for _, d in G.degree()]
            stats["max_degree"] = max(degrees)
            stats["median_degree"] = np.median(degrees)

        return stats

    def save_graph(self, G: nx.Graph, path: str):
        """Save NetworkX graph to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        nx.write_gpickle(G, path)
        logger.info(f"Saved graph to {path}")

    def load_graph(self, path: str) -> nx.Graph:
        """Load NetworkX graph from disk."""
        G = nx.read_gpickle(path)
        logger.info(f"Loaded graph from {path}: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges")
        return G
