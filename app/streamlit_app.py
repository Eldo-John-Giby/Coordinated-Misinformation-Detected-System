#!/usr/bin/env python3
"""
Streamlit Dashboard for Coordinated Misinformation Network Detection.

Interactive visualization of:
- Suspicious account groups
- Network graphs
- Coordination scores
- Feature explanations
- Example posts
"""

import json
import os
import sys

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Coordinated Network Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .suspicious-group {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .evidence-item {
        background-color: #f8f9fa;
        padding: 8px;
        margin: 3px 0;
        border-radius: 3px;
        font-family: monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


def load_results(results_dir: str = "results"):
    """Load pipeline results from disk."""
    results_path = os.path.join(results_dir, "pipeline_results.json")
    if not os.path.exists(results_path):
        return None

    with open(results_path, "r") as f:
        return json.load(f)


def load_data(processed_dir: str = "data/processed"):
    """Load processed data."""
    csv_path = os.path.join(processed_dir, "processed_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, low_memory=False)
    return None


def create_network_plot(G: nx.Graph, title: str = "Coordination Network"):
    """Create an interactive Plotly network visualization."""
    if G.number_of_nodes() == 0:
        return go.Figure()

    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_colors = []
    node_sizes = []
    node_text = []
    for node in G.nodes():
        is_io = G.nodes[node].get("is_io", False)
        node_colors.append("#e74c3c" if is_io else "#2ecc71")
        node_sizes.append(10 + G.degree(node) * 3)
        node_text.append(
            f"Account: {node}<br>"
            f"IO: {'Yes' if is_io else 'No'}<br>"
            f"Connections: {G.degree(node)}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="gray"),
            opacity=0.8
        ),
        name="Accounts"
    )

    # Edge trace
    edge_x = []
    edge_y = []
    edge_widths = []
    edge_colors = []
    edge_text = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        score = data.get("coordination_score", 0)
        edge_widths.append(0.5 + score * 3)
        edge_colors.append(score)
        edge_text.append(f"Score: {score:.3f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        hoverinfo="text",
        text=edge_text,
        line=dict(
            width=1,
            color="rgba(150,150,150,0.5)"
        ),
        name="Coordination Edges"
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            height=600,
        )
    )

    return fig


def main():
    st.title("🔍 Coordinated Misinformation Network Detection")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Dashboard Controls")

    # Load results
    results_dir = st.sidebar.text_input("Results Directory", "results")
    results = load_results(results_dir)

    if results is None:
        st.warning(
            "No results found. Please run the pipeline first:\n\n"
            "```bash\npython train.py --sample\n```"
        )
        st.info(
            "This will create a sample dataset and run the full pipeline."
        )
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🕵️ Suspicious Groups", "🌐 Network Graph",
        "📈 Model Comparison", "🔬 Explanations"
    ])

    # Tab 1: Overview
    with tab1:
        st.header("System Overview")

        col1, col2, col3, col4 = st.columns(4)

        data_stats = results.get("data_stats", {})
        with col1:
            st.metric("Total Posts", f"{data_stats.get('total_posts', 0):,}")
        with col2:
            st.metric("Unique Accounts", f"{data_stats.get('unique_accounts', 0):,}")
        with col3:
            graph_stats = results.get("graph_stats", {})
            st.metric("Graph Nodes", f"{graph_stats.get('num_nodes', 0):,}")
        with col4:
            st.metric("Graph Edges", f"{graph_stats.get('num_edges', 0):,}")

        # Model comparison table
        st.subheader("Model Performance Comparison")
        comparison = results.get("model_comparison", {})
        if comparison:
            comp_df = pd.DataFrame([
                {"Model": k, **{mk: f"{mv:.4f}" if isinstance(mv, float) else mv
                               for mk, mv in v.items() if isinstance(mv, (int, float))}}
                for k, v in comparison.items()
            ])
            st.dataframe(comp_df, use_container_width=True)

    # Tab 2: Suspicious Groups
    with tab2:
        st.header("Suspicious Account Groups")

        groups = results.get("suspicious_groups", [])
        if not groups:
            st.info("No suspicious groups detected")
            return

        st.write(f"**{len(groups)} suspicious groups detected**")

        for i, group in enumerate(groups):
            with st.expander(
                f"🚨 Group {group.get('group_id', i)}: "
                f"{group.get('num_accounts', 0)} accounts, "
                f"Score: {group.get('coordination_score', 0):.3f}",
                expanded=(i == 0)
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.write("**Accounts:**")
                    accounts = group.get("accounts", [])
                    for acc in accounts:
                        detail = next(
                            (d for d in group.get("account_details", [])
                             if d["accountid"] == acc),
                            {}
                        )
                        is_io = detail.get("is_io", False)
                        icon = "🔴" if is_io else "🟢"
                        st.write(f"{icon} {acc} "
                                f"(Posts: {detail.get('post_count', 'N/A')})")

                with col2:
                    st.write("**Metrics:**")
                    st.write(f"Coordination Score: "
                            f"{group.get('coordination_score', 0):.3f}")
                    st.write(f"Suspiciousness: "
                            f"{group.get('suspiciousness_score', 0):.3f}")
                    st.write(f"IO Ratio: "
                            f"{group.get('io_ratio', 0):.1%}")

                st.write("**Evidence:**")
                evidence = group.get("evidence", [])
                if evidence:
                    for ev in evidence:
                        st.markdown(
                            f"<div class='evidence-item'>"
                            f"📌 {ev.get('description', 'N/A')}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No strong pairwise evidence found")

    # Tab 3: Network Graph
    with tab3:
        st.header("Coordination Network Graph")

        # Check if graph data is available
        graph_path = os.path.join(results_dir, "figures", "network_graph.png")
        if os.path.exists(graph_path):
            st.image(graph_path, caption="Coordination Network")
        else:
            st.info("Network graph not generated yet")

        # Graph statistics
        graph_stats = results.get("graph_stats", {})
        if graph_stats:
            st.subheader("Graph Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Density:** {graph_stats.get('density', 0):.4f}")
                st.write(f"**Avg Degree:** {graph_stats.get('avg_degree', 0):.2f}")
            with col2:
                st.write(f"**Components:** {graph_stats.get('num_components', 0)}")
                st.write(f"**Avg Clustering:** {graph_stats.get('avg_clustering', 0):.4f}")
            with col3:
                st.write(f"**Max Degree:** {graph_stats.get('max_degree', 0)}")
                st.write(f"**Median Degree:** {graph_stats.get('median_degree', 0):.1f}")

    # Tab 4: Model Comparison
    with tab4:
        st.header("Model Comparison")

        comparison = results.get("model_comparison", {})
        if comparison:
            # ROC curves
            st.subheader("Performance Metrics")

            models = list(comparison.keys())
            metrics_to_show = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

            chart_data = []
            for model_name, metrics in comparison.items():
                for metric in metrics_to_show:
                    if metric in metrics:
                        chart_data.append({
                            "Model": model_name,
                            "Metric": metric.upper(),
                            "Value": metrics[metric]
                        })

            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                fig = px.bar(
                    chart_df, x="Model", y="Value", color="Metric",
                    barmode="group", title="Model Performance Comparison"
                )
                st.plotly_chart(fig, use_container_width=True, key="model_comparison_chart")
        else:
            st.info("No model comparison data available")

    # Tab 5: Explanations
    with tab5:
        st.header("Explanations")

        groups = results.get("suspicious_groups", [])
        if not groups:
            st.info("No groups to explain")
            return

        # Explanation for each group
        for group in groups:
            st.subheader(f"Group {group.get('group_id', 'N/A')}")

            st.write("**Summary:**")
            st.write(group.get("summary", "No summary available"))

            # Feature weights
            st.write("**Detection Weights:**")
            weights = {
                "Semantic": 0.35,
                "Temporal": 0.25,
                "URL": 0.15,
                "Hashtag": 0.10,
                "Mention": 0.05,
                "Repost": 0.10,
            }

            fig = go.Figure(go.Bar(
                x=list(weights.values()),
                y=list(weights.keys()),
                orientation="h",
                marker_color="#3498db"
            ))
            fig.update_layout(
                title="Feature Weights for Coordination Score",
                xaxis_title="Weight",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True, key=f"weights_chart_{group.get('group_id', 'N/A')}")

            st.markdown("---")


if __name__ == "__main__":
    main()
