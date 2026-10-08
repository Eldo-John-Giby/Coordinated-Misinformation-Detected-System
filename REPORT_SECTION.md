# 4. Implementation, Evaluation, and Limitations

## 4.1 System Overview

The system detects coordinated information operations (IOs) in social media campaign data by combining paraphrase-sensitive content matching with behavioral evidence in a weighted account graph. Posts are embedded with a SentenceTransformer bi-encoder (`all-MiniLM-L6-v2`); FAISS retrieves top-k (k=20) similar post pairs above a low similarity floor (0.30) as a recall-oriented candidate stage, and a cross-encoder (`cross-encoder/nli-deberta-v3-base`) reranks the matched post texts of the top candidates to produce final semantic scores. Five behavioral signals — decayed temporal proximity, shared URLs, hashtag overlap, mention-pattern similarity, and repost linkage — are computed per account pair and combined with the semantic score into a weighted edge. The resulting account graph (nodes carry behavioral statistics and a PCA-reduced content embedding) is partitioned with Louvain community detection; GCN and GraphSAGE models then learn node representations over graph structure and node features, and their per-node IO probabilities are used to re-score communities before flagging. Every flagged group exposes a numeric per-signal breakdown and its matched post pairs for human audit.

Account-pair edge weights are computed as:

> C(A,B) = 0.35·Semantic + 0.25·Temporal + 0.15·URL + 0.10·Hashtag + 0.10·Repost + 0.05·Mention

with all six components normalized to [0,1] before combination. The pipeline stages are:

1. **Ingestion/preprocessing** — posts normalized into structured records (timestamp, URLs, hashtags, mentions, repost links, IO/control label).
2. **Bi-encoder embedding** — post-level embeddings (`all-MiniLM-L6-v2`).
3. **FAISS retrieval** — top-k similar post pairs per post (k=20, similarity floor 0.30), aggregated to account-pair candidates (top-3 post pairs retained per account pair).
4. **Cross-encoder rerank** — joint-attention scoring of the matched post texts for the highest-ranked candidates (capped at 20,000 pairs in the reported run).
5. **Behavioral signals** — the five account-pair signals above.
6. **Graph construction** — weighted account graph with the C(A,B) edge score.
7. **Community detection and GNN refinement** — Louvain partition; GCN/GraphSAGE node classification; P(IO)-re-weighted community scoring.
8. **Explainability and evaluation** — per-group 6-signal breakdown with evidence; model comparison on a canonical held-out split.

## 4.2 Experimental Setup and Results

**Dataset.** All reported results come from a real Honduras election-related IO campaign dataset (Zenodo record 13912659), ingested through the repository's converter: 32,000 posts (20,000 IO-labeled, 12,000 control) from 3,733 accounts. A synthetic data generator exists in the codebase for development and testing; it is gated behind an explicit `--allow-synthetic-fallback` flag, tagged `data_source: "synthetic"` when used, and was **not** used for any reported result (`data_source: "real"` on the final run).

**Evaluation protocol.** All six methods are evaluated on one canonical account-level 70/15/15 split (random seed 42; 2,613/560/560), so every row of the comparison table is scored on the identical 560-account test set (each row records `n_test=560`). This uniformity was itself a development fix: earlier runs mixed a full-set threshold baseline, an 80/20 RF/LR split, and a separate random 15% GNN mask, which made the original comparison table not meaningfully interpretable; the final report does not use those mixed splits.

**Results.** F1 on the shared test set, ascending:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| Louvain (community-membership baseline) | 0.409 | 0.347 | 0.832 | **0.490** | — | — |
| Threshold baseline (coordination-score threshold) | 0.341 | 0.341 | 0.995 | **0.507** | 0.126 | 0.208 |
| GCN | 0.959 | 0.959 | 0.959 | **0.959** | 0.989 | 0.986 |
| Logistic regression | 0.982 | 0.984 | 0.963 | **0.974** | 0.993 | 0.992 |
| GraphSAGE | 0.984 | 0.984 | 0.984 | **0.984** | 0.999 | 0.997 |
| Random forest | 0.996 | 0.990 | 1.000 | **0.995** | 0.998 | 0.992 |

The weakest performers are unsupervised — Louvain flags entire communities from edge scores alone (precision 0.35) and the threshold baseline predicts IO for nearly every account (recall 0.995, precision 0.34) — whereas the strongest performer (random forest, F1 0.995) is trained directly on labeled per-account behavioral features, so the gap primarily reflects access to labeled training signal rather than a failure of the graph-based evidence.

**Ablation.** Seven conditions were run (full model plus one for each dropped signal), each retraining the edge-feature-consuming classifier rather than — as in an earlier vacuous version of this experiment — a node-feature-only model unaffected by the ablation. F1 varies by condition from 0.926 (no hashtag) to 0.984 (no temporal) against a full-model baseline of 0.981, with semantic ablation producing the second-largest drop (0.964); per-row values are in `ablation_results.csv`.

**Worked explainability example.** Flagged group 121 (4 accounts) exposes mean per-signal scores of semantic 0.790, temporal 0.166, URL 0.0, hashtag 0.0, repost 0.0, mention 1.0. Its coordination score is computed transparently as 0.35×0.790 + 0.25×0.166 + 0.15×0 + 0.10×0 + 0.10×0 + 0.05×1.0 = 0.368, which is exactly the stored value (0.3678). The group additionally records a GNN-derived IO probability of 0.995 and a combined suspiciousness score of 0.689, together with the specific matched post pairs as evidence.

**Measured GNN effect.** The GNN is not decorative: comparing the flagged groups produced before and after GNN re-scoring (`gnn_integration_diff`, pre-GNN vs post-GNN), 3 of the 10 flagged groups changed in member composition (3 groups dropped, 3 added; the flagged count remained 10). Removing the GNN step therefore changes the final output, confirming its predictions influence which accounts are grouped and flagged.

## 4.3 Development Process: From Initial Implementation to Verified System

**Initial implementation.** The first version's core computational components were genuinely implemented rather than mocked: real SentenceTransformer embeddings, a real cross-encoder reranker, real FAISS retrieval, real Louvain detection (via `python-louvain`), and real GCN/GraphSAGE training. An earlier real-data run over 1.26M posts demonstrated the pipeline executed end to end.

**Issues found during code review.** The review identified five themes of defects. *Data integrity:* a pandas type-coercion bug cast large integer account IDs to floats when NaNs were present, collapsing distinct accounts and producing self-loop edges that inflated coordination scores in the initial real-data run. *Core method dilution:* semantic matching operated on account-averaged embeddings and reranked a single arbitrary representative post per account, rather than the specific matched post pairs — discarding precisely the post-level paraphrase evidence that motivated using a cross-encoder. *Evaluation validity:* the threshold baseline was scored on post volume instead of a coordination score (yielding a degenerate F1 of 0.0), three model families were evaluated on three different splits, and the ablation study retrained a model that did not depend on the ablated features, leaving all seven ablation rows identical. *Integration gaps:* the GNN trained correctly but its output never influenced which groups were flagged, and explanations covered only 4 of the 6 coordination signals with no numeric breakdown. *Environment/dependency issues:* `python-louvain` and `anthropic` were missing from `requirements.txt`, so fresh installs silently substituted a weaker community-detection algorithm.

**Remediation and final verification.** The fix pass replaced account-level averaging with post-level retrieve-then-rerank (FAISS over post embeddings, aggregation to account pairs, cross-encoder scoring of the actual matched texts, winning pairs stored as evidence); applied the single canonical split to every model; rewired the ablation to retrain the classifier that consumes the ablated features; integrated GNN IO probabilities into final community scoring; extended explanations to all six signals with numeric per-signal contributions; pinned the missing dependencies; and converted every silent fallback (FAISS→sklearn, Louvain→greedy modularity, cross-encoder bypass, synthetic-data substitution, missing optional columns, unparseable timestamps) into an explicitly recorded degradation with a WARNING-level log and a `degraded_components` section in the results file. A second-order defect was caught and fixed during this phase: unit tests wrote synthetic data and its provenance marker into the real data directory, mis-tagging an earlier real run as synthetic — the `data_source` field now derives from the files actually loaded (a stale marker is recorded as a degradation rather than overriding provenance). Final verification consisted of a 12-point acceptance checklist from the review — executed via an automated verification script with logged evidence — all passing on a fresh end-to-end run on the real dataset (3,294.8 s on CPU), including zero self-loops in the output graph, a fully populated comparison table on the shared split, and model checkpoints that reload and reproduce predictions on all 560 held-out accounts.

## 4.4 Limitations

- **Single-campaign evaluation.** All results come from one campaign (Honduras, 32k posts); generalization to other coordination campaigns, platforms, or paraphrase styles not represented in this data is untested.
- **Cross-encoder is a proxy.** The reranker is an NLI-entailment model (`nli-deberta-v3-base`) used as a paraphrase detector, not a model trained for paraphrase identification; this modeling choice may mismatch true paraphrase semantics.
- **Rerank coverage is capped.** Cost control restricted cross-encoder scoring to the top 20,000 of 362,427 candidate post pairs (16,833 account pairs carry cross-encoder scores); the remainder retain lower-confidence bi-encoder scores, an explicitly recorded degradation rather than a hidden one.
- **Full-batch GNN training.** Both GCN and GraphSAGE train full-batch without neighborhood sampling, a scalability constraint for datasets substantially larger than the evaluated one.
- **Fixed scoring weights.** The coordination weights (0.35/0.25/0.15/0.10/0.10/0.05) are manually set; a learned-weight mode (logistic regression on the six signals against labels) exists but was off by default and was **not** used to produce the reported results.
- **No cross-campaign generalization test.** All metrics reflect in-campaign generalization on a held-out account split; no held-out campaign or platform evaluation was performed.
