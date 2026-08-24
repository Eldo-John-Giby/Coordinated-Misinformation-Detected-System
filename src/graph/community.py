"""
Community detection for identifying coordinated account groups.

Uses Louvain community detection and other methods to identify
suspicious clusters of accounts.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detect communities of coordinated accounts."""

    def __init__(self, config: dict):
        self.config = config

    def detect_louvain_communities(self, G: nx.Graph,
                                    resolution: float = 1.0
                                    ) -> Dict[int, List[str]]:
        """
        Detect communities using Louvain method.

        Args:
            G: NetworkX graph with coordination edges
            resolution: Resolution parameter (higher = more communities)

        Returns:
            Dict mapping community_id to list of account IDs
        """
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(
                G, resolution=resolution, weight="coordination_score"
            )
        except ImportError:
            logger.warning("python-louvain not available, using nx greedy_modularity")
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(
                G, weight="coordination_score"
            )
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i

        # Group by community
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)

        logger.info(f"Detected {len(communities)} Louvain communities")
        return communities

    def detect_label_propagation(self, G: nx.Graph) -> Dict[int, List[str]]:
        """Detect communities using Label Propagation."""
        from networkx.algorithms.community import label_propagation_communities

        communities_list = list(label_propagation_communities(G))
        communities = {i: list(comm) for i, comm in enumerate(communities_list)}

        logger.info(f"Detected {len(communities)} label propagation communities")
        return communities

    def score_communities(self, communities: Dict[int, List[str]],
                          G: nx.Graph) -> pd.DataFrame:
        """
        Score each community based on internal coordination strength.

        A suspicious community has:
        - High internal edge density
        - High average coordination score
        - Many members with IO-like behavior
        """
        results = []

        for comm_id, members in communities.items():
            if len(members) < 2:
                continue

            # Get subgraph
            subgraph = G.subgraph(members)

            # Internal metrics
            internal_edges = subgraph.number_of_edges()
            max_possible_edges = len(members) * (len(members) - 1) / 2
            density = internal_edges / max_possible_edges if max_possible_edges > 0 else 0

            # Average coordination score of internal edges
            if internal_edges > 0:
                weights = [d.get("coordination_score", 0)
                          for _, _, d in subgraph.edges(data=True)]
                avg_coordination = np.mean(weights)
                max_coordination = np.max(weights)
            else:
                avg_coordination = 0
                max_coordination = 0

            # IO label ratio
            io_count = sum(
                1 for n in members
                if G.nodes[n].get("is_io", False)
            )
            io_ratio = io_count / len(members)

            # Overall suspiciousness score
            suspiciousness = (
                0.3 * density +
                0.3 * avg_coordination +
                0.2 * min(len(members) / 10, 1.0) +  # Size factor
                0.2 * io_ratio
            )

            results.append({
                "community_id": comm_id,
                "num_accounts": len(members),
                "internal_edges": internal_edges,
                "density": density,
                "avg_coordination_score": avg_coordination,
                "max_coordination_score": max_coordination,
                "io_account_count": io_count,
                "io_ratio": io_ratio,
                "suspiciousness_score": suspiciousness,
                "accounts": members,
            })

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.sort_values(
                "suspiciousness_score", ascending=False
            ).reset_index(drop=True)

        logger.info(f"Scored {len(results_df)} communities")
        return results_df

    def get_suspicious_groups(self, communities_df: pd.DataFrame,
                              top_k: int = 10,
                              min_size: int = 3,
                              min_suspiciousness: float = 0.3
                              ) -> pd.DataFrame:
        """
        Get the most suspicious groups.

        Args:
            communities_df: Scored community DataFrame
            top_k: Maximum groups to return
            min_size: Minimum group size
            min_suspiciousness: Minimum suspiciousness threshold

        Returns:
            DataFrame of suspicious groups
        """
        filtered = communities_df[
            (communities_df["num_accounts"] >= min_size) &
            (communities_df["suspiciousness_score"] >= min_suspiciousness)
        ]

        return filtered.head(top_k)

    def generate_group_explanations(self, suspicious_groups: pd.DataFrame,
                                    G: nx.Graph,
                                    df: pd.DataFrame) -> List[Dict]:
        """
        Generate human-readable explanations for each suspicious group.

        Provides interpretable evidence of why each group was flagged.
        """
        explanations = []

        for _, group in suspicious_groups.iterrows():
            members = group["accounts"]
            comm_id = group["community_id"]

            # Find key evidence edges (highest coordination scores)
            subgraph = G.subgraph(members)
            top_edges = sorted(
                subgraph.edges(data=True),
                key=lambda x: x[2].get("coordination_score", 0),
                reverse=True
            )[:5]

            evidence_items = []
            for u, v, data in top_edges:
                sem = data.get("semantic_similarity", 0)
                temp = data.get("temporal_score", 0)
                shared_urls = data.get("shared_url_count", 0)
                shared_ht = data.get("shared_hashtag_count", 0)

                explanation_parts = []
                if sem > 0.7:
                    explanation_parts.append(
                        f"high semantic similarity ({sem:.2f})"
                    )
                if temp > 0.7:
                    explanation_parts.append(
                        f"temporal coordination ({temp:.2f})"
                    )
                if shared_urls > 0:
                    explanation_parts.append(f"shared {shared_urls} URL(s)")
                if shared_ht > 0:
                    explanation_parts.append(f"shared {shared_ht} hashtag(s)")

                if explanation_parts:
                    evidence_items.append({
                        "accounts": (u, v),
                        "coordination_score": data.get("coordination_score", 0),
                        "description": f"Accounts {u} and {v}: " +
                                      ", ".join(explanation_parts),
                    })

            # Account details
            account_details = []
            for acc in members:
                node_data = G.nodes[acc]
                account_details.append({
                    "accountid": acc,
                    "is_io": node_data.get("is_io", False),
                    "post_count": node_data.get("post_count", 0),
                })

            explanation = {
                "group_id": comm_id,
                "accounts": members,
                "num_accounts": len(members),
                "coordination_score": group["avg_coordination_score"],
                "suspiciousness_score": group["suspiciousness_score"],
                "io_ratio": group["io_ratio"],
                "evidence": evidence_items,
                "account_details": account_details,
                "summary": (
                    f"Group {comm_id}: {len(members)} accounts with "
                    f"avg coordination score {group['avg_coordination_score']:.3f}. "
                    f"{group['io_account_count']} accounts match IO labels "
                    f"({group['io_ratio']:.1%}). "
                    f"{len(evidence_items)} evidence edges found."
                ),
            }
            explanations.append(explanation)

        return explanations
