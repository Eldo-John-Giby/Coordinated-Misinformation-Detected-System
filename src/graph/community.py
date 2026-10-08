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

        # Optional DegradationTracker (item 8b): the Louvain -> greedy
        # modularity fallback silently changed the detection algorithm on
        # every fresh install (python-louvain was missing from
        # requirements.txt). It must be visible in the results, not just logs.
        self._degradations = None

    def _note_degradation(self, component: str, reason: str, detail: str = ""):
        if self._degradations is not None:
            self._degradations.record(component, reason, detail)
        else:
            logger.warning("DEGRADED [%s]: %s%s", component, reason,
                           f" ({detail})" if detail else "")

    def detect_louvain_communities(self, G: nx.Graph,
                                    resolution: float = 1.0,
                                    degradations=None
                                    ) -> Dict[int, List[str]]:
        """
        Detect communities using Louvain method.

        Args:
            G: NetworkX graph with coordination edges
            resolution: Resolution parameter (higher = more communities)
            degradations: optional DegradationTracker; records the fallback
                to greedy modularity when python-louvain is unavailable.

        Returns:
            Dict mapping community_id to list of account IDs
        """
        self._degradations = degradations
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(
                G, resolution=resolution, weight="coordination_score"
            )
        except ImportError:
            self._note_degradation(
                "louvain",
                "python-louvain not installed; fell back to networkx "
                "greedy_modularity_communities (different algorithm, "
                "different partitions)",
            )
            from networkx.algorithms.community import greedy_modularity_communities
            communities = greedy_modularity_communities(
                G, weight="coordination_score"
            )
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
        self._degradations = None

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
                          G: nx.Graph,
                          gnn_scores: Optional[Dict[str, float]] = None,
                          degradations=None) -> pd.DataFrame:
        """
        Score each community based on internal coordination strength.

        A suspicious community has:
        - High internal edge density
        - High average coordination score
        - Many members the GNN classifies as coordinated/IO (item 10, Option B)

        The fourth term uses the TRAINED GNN's predicted P(IO) per node, so
        the GNN directly influences which groups get flagged. When GNN scores
        are unavailable the term contributes 0.0 and the omission is recorded
        (previously this term was ground-truth io_ratio, which leaked labels
        into the unsupervised flagging step).

        Args:
            communities: community_id -> member account list (from Louvain).
            G: the coordination graph (edge attrs carry signal scores).
            gnn_scores: optional account -> P(IO) from the trained GNN.
            degradations: optional DegradationTracker.
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

            # Ground-truth IO ratio: reported for audit ONLY, not used in the
            # score (using it for flagging was label leakage).
            io_count = sum(
                1 for n in members
                if G.nodes[n].get("is_io", False)
            )
            io_ratio = io_count / len(members)

            # GNN-predicted P(IO), averaged over members (item 10 Option B).
            if gnn_scores:
                gnn_values = [gnn_scores.get(n, 0.0) for n in members]
                gnn_ratio = float(np.mean(gnn_values))
            else:
                gnn_ratio = 0.0
                if degradations is not None:
                    degradations.record(
                        "gnn_integration",
                        "community scoring ran without GNN scores; the "
                        "GNN term contributed 0 to suspiciousness",
                    )

            # Overall suspiciousness score
            suspiciousness = (
                0.3 * density +
                0.3 * avg_coordination +
                0.2 * min(len(members) / 10, 1.0) +  # Size factor
                0.2 * gnn_ratio  # GNN-driven term (item 10)
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
                "gnn_io_ratio": gnn_ratio,
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

    # The six coordination signals and the edge attribute each comes from
    # (item 12: all six must appear in the numeric per-group breakdown).
    SIGNAL_EDGE_ATTRS = [
        ("semantic", "semantic_similarity"),
        ("temporal", "temporal_score"),
        ("url", "shared_url_score"),
        ("hashtag", "shared_hashtag_score"),
        ("repost", "repost_score"),
        ("mention", "shared_mention_score"),
    ]

    def generate_group_explanations(self, suspicious_groups: pd.DataFrame,
                                    G: nx.Graph,
                                    df: pd.DataFrame) -> List[Dict]:
        """
        Generate human-readable explanations for each suspicious group.

        Provides interpretable evidence of why each group was flagged:
        - a numeric per-signal breakdown (mean of each of the 6 coordination
          signals across the group's internal edges, with weights and weighted
          contributions), not just prose naming which signals fired;
        - matched post pairs for the semantic signal (item 4d evidence).
        """
        import json as _json

        weights = self.config.get("coordination", {}).get("weights", {})
        explanations = []

        for _, group in suspicious_groups.iterrows():
            members = group["accounts"]
            comm_id = group["community_id"]

            subgraph = G.subgraph(members)
            top_edges = sorted(
                subgraph.edges(data=True),
                key=lambda x: x[2].get("coordination_score", 0),
                reverse=True
            )[:5]

            # ---- numeric per-signal breakdown across ALL internal edges
            signal_breakdown = {}
            if subgraph.number_of_edges() > 0:
                for signal_name, attr in self.SIGNAL_EDGE_ATTRS:
                    vals = [d.get(attr, 0) for _, _, d in subgraph.edges(data=True)]
                    mean_val = float(np.mean(vals))
                    w = float(weights.get(signal_name, 0.0))
                    signal_breakdown[signal_name] = {
                        "mean_score": round(mean_val, 4),
                        "weight": w,
                        "weighted_contribution": round(w * mean_val, 4),
                    }
            else:
                for signal_name, _ in self.SIGNAL_EDGE_ATTRS:
                    signal_breakdown[signal_name] = {
                        "mean_score": 0.0,
                        "weight": float(weights.get(signal_name, 0.0)),
                        "weighted_contribution": 0.0,
                    }

            evidence_items = []
            semantic_post_pairs = []
            for u, v, data in top_edges:
                sem = data.get("semantic_similarity", 0)
                temp = data.get("temporal_score", 0)
                shared_urls = data.get("shared_url_count", 0)
                shared_ht = data.get("shared_hashtag_count", 0)
                shared_mn = data.get("shared_mention_count", 0)
                rep = data.get("repost_score", 0)

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
                if shared_mn > 0:
                    explanation_parts.append(f"shared {shared_mn} mention(s)")
                if rep > 0:
                    explanation_parts.append(f"repost linkage ({rep:.2f})")

                if explanation_parts:
                    evidence_items.append({
                        "accounts": (u, v),
                        "coordination_score": data.get("coordination_score", 0),
                        "description": f"Accounts {u} and {v}: " +
                                      ", ".join(explanation_parts),
                    })

                # Item 4d: matched post pairs stored on the edge
                for pair in data.get("semantic_evidence", []) or []:
                    semantic_post_pairs.append({
                        "accounts": [str(u), str(v)],
                        "post_a": str(pair.get("post_a", "")),
                        "post_b": str(pair.get("post_b", "")),
                        "text_a": pair.get("text_a", ""),
                        "text_b": pair.get("text_b", ""),
                        "cross_encoder_score": pair.get("cross_encoder_score"),
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
                "gnn_io_ratio": group.get("gnn_io_ratio", 0.0),
                # Item 12: numeric values for all 6 signals
                "signal_breakdown": signal_breakdown,
                "evidence": evidence_items,
                "semantic_post_pairs": semantic_post_pairs[:10],
                "account_details": account_details,
                "summary": (
                    f"Group {comm_id}: {len(members)} accounts with "
                    f"avg coordination score {group['avg_coordination_score']:.3f}. "
                    f"GNN P(IO) {group.get('gnn_io_ratio', 0.0):.2f}. "
                    f"Top signals: " + ", ".join(
                        sorted(
                            signal_breakdown.items(),
                            key=lambda kv: kv[1]["weighted_contribution"],
                            reverse=True,
                        )[:3]
                        and [
                            f"{k}={v['mean_score']:.2f}"
                            for k, v in sorted(
                                signal_breakdown.items(),
                                key=lambda kv: kv[1]["weighted_contribution"],
                                reverse=True,
                            )[:3]
                        ]
                    ) + f". {len(evidence_items)} evidence edges found."
                ),
            }
            explanations.append(explanation)

        return explanations
