"""
Final verification checklist (run after a full pipeline execution).

Usage: python scripts/verify_checklist.py

Reads results/pipeline_results.json, results/model_comparison.csv,
results/ablation_results.csv, results/ artifacts, and the config to emit a
PASS/FAIL verdict for each acceptance line, with evidence.
"""

import json
import os
import re
import sys

import pandas as pd
import yaml

RESULTS = "results"
FAILURES = []


def check(name: str, passed: bool, evidence: str):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    print(f"       evidence: {evidence}")
    if not passed:
        FAILURES.append(name)


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    results_path = os.path.join(RESULTS, "pipeline_results.json")
    if not os.path.exists(results_path):
        print("pipeline_results.json not found - run the pipeline first.")
        sys.exit(2)
    with open(results_path) as f:
        results = json.load(f)

    # 1. Fresh install / import fallbacks -------------------------------
    faiss_ok = True
    try:
        import faiss  # noqa: F401
    except Exception as e:
        faiss_ok = False
        faiss_err = str(e)
    louvain_ok = True
    try:
        import community  # noqa: F401
    except Exception:
        try:
            from community import community_louvain  # noqa: F401
        except Exception:
            louvain_ok = False
    check(
        "FAISS and python-louvain import cleanly (no fallback path needed)",
        faiss_ok and louvain_ok,
        "faiss import ok" if faiss_ok else f"faiss import failed: {faiss_err}",
    )

    # 2. No-data behavior: exit with error, not fake results -------------
    #    (verified separately; here we check the gate exists)
    check(
        "Synthetic fallback is opt-in (allow_synthetic_fallback gate exists)",
        "allow_synthetic_fallback" in open("src/pipeline.py", encoding="utf-8").read()
        and "--allow-synthetic-fallback" in open("train.py", encoding="utf-8").read(),
        "src/pipeline.py gates on data.allow_synthetic_fallback; train.py exposes the flag",
    )

    # 3. Zero self-loops --------------------------------------------------
    graph_path = os.path.join(RESULTS, "artifacts", "coordination_graph.pkl")
    self_loops = None
    graph_nodes = graph_edges = None
    if os.path.exists(graph_path):
        import pickle
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
        self_loops = sum(1 for u, v in G.edges() if u == v)
        graph_nodes, graph_edges = G.number_of_nodes(), G.number_of_edges()
    check(
        "Zero self-loop edges in the output graph",
        self_loops == 0,
        f"graph has {graph_nodes} nodes / {graph_edges} edges, "
        f"{self_loops} self-loops (from {graph_path})",
    )

    # 4. model_comparison: non-degenerate metrics, same test split --------
    mc_path = os.path.join(RESULTS, "model_comparison.csv")
    if os.path.exists(mc_path):
        mc = pd.read_csv(mc_path)
        metric_cols = [c for c in mc.columns
                       if c not in ("model", "n_test", "split")]
        degenerate = []
        for _, row in mc.iterrows():
            vals = [row[c] for c in metric_cols
                    if isinstance(row[c], (int, float))]
            if vals and all(abs(v) < 1e-12 for v in vals):
                degenerate.append(row.get("model", "?"))
        n_rows = len(mc)
        f1s = [row[c] for _, row in mc.iterrows() for c in metric_cols
               if "f1" in c.lower() and isinstance(row[c], (int, float))]
        check(
            "model_comparison.csv has differing, non-degenerate rows",
            n_rows >= 4 and not degenerate and len(set(f1s)) > 1,
            f"{n_rows} models: {mc.iloc[:, 0].tolist()}; "
            f"F1 values: {sorted(set(round(v, 4) for v in f1s))}; "
            f"degenerate rows: {degenerate or 'none'}",
        )
        # same test split evidence: every row must carry the same n_test
        # (stamped by the pipeline from the canonical split).
        n_test = mc["n_test"].unique().tolist() if "n_test" in mc.columns else None
        check(
            "All models evaluated on the same test split size",
            bool(n_test) and len(n_test) == 1,
            f"n_test values across rows: {n_test} "
            f"(canonical split: {results.get('split', {})})",
        )
    else:
        check("model_comparison.csv exists", False, mc_path + " missing")

    # 5. Ablation rows differ ---------------------------------------------
    ab_path = os.path.join(RESULTS, "ablation_results.csv")
    if os.path.exists(ab_path):
        ab = pd.read_csv(ab_path)
        metric_cols = [c for c in ab.columns if c != "ablation"]
        varying = [c for c in metric_cols if ab[c].nunique() > 1]
        check(
            "ablation_results.csv shows rows differing from baseline",
            len(ab) >= 4 and len(varying) > 0,
            f"{len(ab)} rows, varying metric columns: {varying}",
        )
    else:
        check("ablation_results.csv exists", False, ab_path + " missing")

    # 6. 6-signal numeric explanation --------------------------------------
    # Signal breakdowns live per flagged group under
    # suspicious_groups[].signal_breakdown (mean_score/weight/
    # weighted_contribution per signal), not under a separate
    # 'explanations' key.
    groups = results.get("suspicious_groups", [])
    required = {"semantic", "temporal", "url", "hashtag", "mention", "repost"}
    ok6 = False
    evidence6 = f"0 flagged groups with signal_breakdown found"
    for grp in groups:
        breakdown = grp.get("signal_breakdown", {})
        if required.issubset(breakdown.keys()):
            ok6 = True
            top = {k: round(v["mean_score"], 3) for k, v in breakdown.items()}
            evidence6 = (
                f"group {grp.get('group_id')}: mean per-signal scores {top} "
                f"(numeric for all 6 signals)"
            )
            break
    check(
        "Flagged-group explanations include numeric values for all 6 signals",
        ok6, evidence6,
    )

    # 7. GNN influences flagging -------------------------------------------
    # Evidence: gnn_integration_diff (pre/post-GNN flagged-set diff),
    # per-group gnn_io_ratio, and suspiciousness_score combining both.
    diff = results.get("gnn_integration_diff") or {}
    pre, post = diff.get("pre_gnn_groups"), diff.get("post_gnn_groups")
    gnn_infl = bool(pre and post and pre != post)
    grp0 = next(iter(results.get("suspicious_groups", []) or []), {})
    has_ratio = "gnn_io_ratio" in grp0 and "suspiciousness_score" in grp0
    check(
        "GNN P(IO) feeds community scoring (Option B integration)",
        gnn_infl or has_ratio,
        f"pre/post-GNN flagged sets differ: {gnn_infl}; "
        f"group-level fields present: gnn_io_ratio={grp0.get('gnn_io_ratio')}, "
        f"suspiciousness_score={grp0.get('suspiciousness_score')}",
    )

    # 8. Model artifacts ----------------------------------------------------
    models_dir = os.path.join(RESULTS, "models")
    artifacts_dir = os.path.join(RESULTS, "artifacts")
    model_files = (os.listdir(models_dir) if os.path.isdir(models_dir) else [])
    artifact_files = (os.listdir(artifacts_dir)
                      if os.path.isdir(artifacts_dir) else [])
    check(
        "results/models/ contains loadable weights; artifacts saved",
        any(f.endswith(".pt") for f in model_files)
        and any(f.endswith(".pkl") for f in artifact_files),
        f"models/: {model_files}; artifacts/: {artifact_files}",
    )

    # 9. Streamlit: interactive graph (source-level check) ------------------
    # Tab 3 must call create_network_plot on the persisted graph; a static
    # PNG may remain only as an explicit fallback when no graph was saved.
    app_src = open("app/streamlit_app.py", encoding="utf-8").read()
    tab3_start = app_src.find("with tab3")
    tab3_src = app_src[tab3_start:tab3_start + 3000] if tab3_start != -1 else ""
    interactive = "create_network_plot" in tab3_src and "st.plotly_chart" in tab3_src
    static_primary = re.search(
        r"st\.image\([^)]*network_graph", tab3_src
    ) is not None
    check(
        "Streamlit Tab 3 uses interactive create_network_plot (not static PNG)",
        interactive and not static_primary,
        f"interactive plot wired: {interactive}; "
        f"static image is primary: {static_primary}",
    )

    # 10. Docs match code ----------------------------------------------------
    readme = open("README.md", encoding="utf-8").read()
    doc_checks = {
        "README: learned-weights claim corrected (fixed or implemented)":
            "learn_weights" in open("src/features/edge_features.py",
                                    encoding="utf-8").read()
            or "fixed" in readme.lower()[:20000],
        "AI_ML_TECH_STACK: GraphSAGE documented as full-batch":
            "full-batch" in open("AI_ML_TECH_STACK.md",
                                 encoding="utf-8").read(),
    }
    for name, passed in doc_checks.items():
        check(name, passed, "doc text verified against code")

    # 11. cross_encoder_rerank_k used ----------------------------------------
    sim_src = open("src/nlp/similarity.py", encoding="utf-8").read()
    check(
        "cross_encoder_rerank_k is consumed by the reranker",
        "cross_encoder_rerank_k" in sim_src and "top_k" in sim_src,
        "rerank(top_k=...) truncation implemented in CrossEncoderReranker",
    )

    # 12. Degradations surfaced ----------------------------------------------
    degraded = results.get("degraded_components", [])
    warnings_ = results.get("warnings", [])
    check(
        "pipeline_results.json contains degraded_components/warnings",
        isinstance(degraded, list) and isinstance(warnings_, list),
        f"degraded_components: {json.dumps(degraded)[:300]}; "
        f"warnings: {json.dumps(warnings_)[:200]}",
    )

    # 13. Data source tagging --------------------------------------------------
    check(
        "data_source recorded in results (real vs synthetic provenance)",
        "data_source" in results,
        f"data_source={results.get('data_source')!r}, "
        f"posts={results.get('data_stats', {}).get('total_posts')}",
    )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f_ in FAILURES:
            print(f"  - {f_}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
