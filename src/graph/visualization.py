"""
Visualization module for coordination analysis.

Creates network visualizations, metric distributions,
confusion matrices, ROC/PR curves, and model comparison plots.
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Create visualizations for coordination analysis."""

    def __init__(self, config: dict):
        self.config = config
        self.viz_config = config.get("visualization", {})
        self.output_dir = self.viz_config.get("output_dir", "results/figures")
        self.format = self.viz_config.get("format", "png")
        self.dpi = self.viz_config.get("dpi", 150)
        self.max_nodes = self.viz_config.get("max_network_nodes", 200)
        self.colors = self.viz_config.get("color_scheme", {
            "io_account": "#e74c3c",
            "control_account": "#2ecc71",
            "suspicious_group": "#e67e22",
            "background": "#ffffff",
        })
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_fig(self, fig, name: str):
        """Save figure to disk."""
        path = os.path.join(self.output_dir, f"{name}.{self.format}")
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight",
                   facecolor="white", edgecolor="none")
        plt.close(fig)
        logger.info(f"Saved figure: {path}")
        return path

    def plot_network(self, G: nx.Graph, title: str = "Coordination Network",
                     highlight_groups: Optional[List[List[str]]] = None,
                     filename: str = "network_graph"):
        """
        Plot the coordination network.

        Nodes = accounts, colored by ground truth or prediction.
        Edges = coordination relationships, thickness proportional to score.
        """
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph, skipping network plot")
            return

        # Subsample if too large
        if G.number_of_nodes() > self.max_nodes:
            # Take the most connected nodes
            degrees = dict(G.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:self.max_nodes]
            G = G.subgraph(top_nodes).copy()
            logger.info(f"Subsampled network to {self.max_nodes} nodes for visualization")

        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Node colors based on IO/control status
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get("is_io", False):
                node_colors.append(self.colors["io_account"])
            else:
                node_colors.append(self.colors["control_account"])

        # Node sizes based on degree
        degrees = [G.degree(n) for n in G.nodes()]
        max_deg = max(degrees) if degrees else 1
        node_sizes = [100 + 400 * (d / max_deg) for d in degrees]

        # Edge widths based on coordination score
        edge_weights = [G[u][v].get("coordination_score", 0.1)
                       for u, v in G.edges()]
        max_ew = max(edge_weights) if edge_weights else 1
        edge_widths = [0.5 + 3 * (w / max_ew) for w in edge_weights]
        edge_colors = [plt.cm.YlOrRd(w / max_ew) for w in edge_weights]

        # Layout
        if G.number_of_nodes() < 50:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(G, k=1.5, iterations=30, seed=42)

        # Draw
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                              edge_color=edge_colors, alpha=0.6)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                              node_color=node_colors, alpha=0.8, edgecolors="gray",
                              linewidths=0.5)

        # Labels for small graphs
        if G.number_of_nodes() <= 30:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        # Legend
        legend_elements = [
            mpatches.Patch(color=self.colors["io_account"], label="IO Account"),
            mpatches.Patch(color=self.colors["control_account"], label="Control Account"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        return self._save_fig(fig, filename)

    def plot_feature_distribution(self, df: pd.DataFrame,
                                  feature_col: str,
                                  group_col: str = "is_io",
                                  title: str = None,
                                  filename: str = None):
        """Plot distribution of a feature split by IO/control."""
        fig, ax = plt.subplots(figsize=(10, 6))

        groups = df[group_col].unique()
        for group in sorted(groups):
            subset = df[df[group_col] == group]
            label = "IO" if group else "Control"
            color = self.colors["io_account"] if group else self.colors["control_account"]
            ax.hist(subset[feature_col], bins=50, alpha=0.5, label=label,
                   color=color, density=True)

        ax.set_xlabel(feature_col)
        ax.set_ylabel("Density")
        ax.set_title(title or f"Distribution of {feature_col}")
        ax.legend()

        return self._save_fig(fig, filename or f"dist_{feature_col}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "Confusion Matrix",
                              filename: str = "confusion_matrix"):
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=["Control", "IO"],
                   yticklabels=["Control", "IO"], ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        return self._save_fig(fig, filename)

    def plot_roc_curve(self, y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                       title: str = "ROC Curves",
                       filename: str = "roc_curve"):
        """Plot ROC curves for multiple models."""
        from sklearn.metrics import roc_curve, auc

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(y_scores)))

        for (model_name, scores), color in zip(y_scores.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                   label=f"{model_name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return self._save_fig(fig, filename)

    def plot_pr_curve(self, y_true: np.ndarray, y_scores: Dict[str, np.ndarray],
                      title: str = "Precision-Recall Curves",
                      filename: str = "pr_curve"):
        """Plot Precision-Recall curves for multiple models."""
        from sklearn.metrics import precision_recall_curve, average_precision_score

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set1(np.linspace(0, 1, len(y_scores)))

        for (model_name, scores), color in zip(y_scores.items(), colors):
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            ax.plot(recall, precision, color=color, linewidth=2,
                   label=f"{model_name} (AP = {ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        return self._save_fig(fig, filename)

    def plot_model_comparison(self, results_df: pd.DataFrame,
                              filename: str = "model_comparison"):
        """Plot bar chart comparing model performance."""
        fig, ax = plt.subplots(figsize=(12, 6))

        metrics = [c for c in results_df.columns
                  if c not in ["model", "precision", "recall"]]
        x = np.arange(len(results_df))
        width = 0.8 / len(metrics)

        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, results_df[metric], width,
                  label=metric, alpha=0.8)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.set_xticks(x + width * len(metrics) / 2)
        ax.set_xticklabels(results_df["model"], rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        return self._save_fig(fig, filename)

    def plot_ablation_results(self, ablation_df: pd.DataFrame,
                              filename: str = "ablation_study"):
        """Plot ablation study results."""
        fig, ax = plt.subplots(figsize=(12, 6))

        metrics = [c for c in ablation_df.columns if c != "ablation"]
        x = np.arange(len(ablation_df))
        width = 0.8 / len(metrics)

        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i * width, ablation_df[metric], width,
                         label=metric, alpha=0.8)

        ax.set_xlabel("Ablation Configuration")
        ax.set_ylabel("Score")
        ax.set_title("Feature Ablation Study")
        ax.set_xticks(x + width * len(metrics) / 2)
        ax.set_xticklabels(ablation_df["ablation"], rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        return self._save_fig(fig, filename)

    def plot_community_detection(self, G: nx.Graph,
                                 communities: Dict[int, List[str]],
                                 title: str = "Detected Communities",
                                 filename: str = "communities"):
        """Plot communities detected in the graph."""
        if G.number_of_nodes() == 0:
            return

        # Subsample if needed
        if G.number_of_nodes() > self.max_nodes:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:self.max_nodes]
            G = G.subgraph(top_nodes).copy()

        fig, ax = plt.subplots(figsize=(14, 10))

        # Color by community
        node_community = {}
        for comm_id, members in communities.items():
            for member in members:
                if member in G.nodes():
                    node_community[member] = comm_id

        cmap = plt.cm.Set3
        unique_comms = sorted(set(node_community.values()))
        comm_colors = {c: cmap(i / max(len(unique_comms), 1))
                      for i, c in enumerate(unique_comms)}

        node_colors = [comm_colors.get(node_community.get(n, -1), (0.7, 0.7, 0.7, 1))
                      for n in G.nodes()]

        # Layout
        pos = nx.spring_layout(G, k=1.5, iterations=30, seed=42)

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=100,
                              node_color=node_colors, alpha=0.8)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        return self._save_fig(fig, filename)

    def plot_coordination_score_distribution(self, df: pd.DataFrame,
                                              score_col: str = "coordination_score",
                                              filename: str = "coordination_distribution"):
        """Plot the distribution of coordination scores."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(df[score_col], bins=50, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(x=self.config.get("coordination", {}).get("edge_threshold", 0.3),
                  color="red", linestyle="--", label="Edge Threshold")
        ax.set_xlabel("Coordination Score")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Coordination Scores")
        ax.legend()

        return self._save_fig(fig, filename)

    def plot_explanation_dashboard(self, group_explanation: dict,
                                   df: pd.DataFrame,
                                   filename: str = "group_explanation"):
        """Create a multi-panel explanation dashboard for a group."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Panel 1: Group summary
        ax = axes[0, 0]
        ax.axis("off")
        summary_text = (
            f"Group ID: {group_explanation['group_id']}\n"
            f"Accounts: {len(group_explanation['accounts'])}\n"
            f"Coordination Score: {group_explanation['coordination_score']:.3f}\n"
            f"Suspiciousness: {group_explanation['suspiciousness_score']:.3f}\n"
            f"IO Ratio: {group_explanation['io_ratio']:.1%}"
        )
        ax.text(0.1, 0.5, summary_text, fontsize=12, fontfamily="monospace",
               verticalalignment="center", transform=ax.transAxes,
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_title("Group Summary")

        # Panel 2: Evidence
        ax = axes[0, 1]
        if group_explanation["evidence"]:
            evidence_text = "\n".join(
                e["description"][:80] for e in group_explanation["evidence"][:5]
            )
        else:
            evidence_text = "No strong pairwise evidence found"
        ax.text(0.1, 0.5, evidence_text, fontsize=10, fontfamily="monospace",
               verticalalignment="center", transform=ax.transAxes,
               wrap=True, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
        ax.set_title("Evidence")

        # Panel 3: Feature importance (if available)
        ax = axes[1, 0]
        features = ["semantic", "temporal", "url", "hashtag", "mention", "repost"]
        weights = [
            self.config.get("coordination", {}).get("weights", {}).get(f, 0.1)
            for f in features
        ]
        ax.barh(features, weights, color="#2ecc71")
        ax.set_xlabel("Weight")
        ax.set_title("Feature Weights")

        # Panel 4: Account details
        ax = axes[1, 1]
        details = group_explanation.get("account_details", [])
        if details:
            detail_df = pd.DataFrame(details)
            detail_text = detail_df.to_string(index=False)
            ax.text(0.05, 0.5, detail_text, fontsize=8, fontfamily="monospace",
                   verticalalignment="center", transform=ax.transAxes)
        ax.set_title("Account Details")
        ax.axis("off")

        fig.suptitle(f"Suspicious Group Explanation: Group {group_explanation['group_id']}",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()

        return self._save_fig(fig, filename)
