"""
Edge feature extraction for account pairs.

Computes all coordination evidence features for pairs of accounts:
- Shared URLs
- Shared hashtags
- Shared mentions
- Repost relationships
- Combined coordination score
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EdgeFeatureExtractor:
    """Extract edge-level features for account pairs."""

    def __init__(self, config: dict):
        self.config = config
        self.coord_config = config.get("coordination", {})
        self.weights = self.coord_config.get("weights", {
            "semantic": 0.35,
            "temporal": 0.25,
            "url": 0.15,
            "hashtag": 0.10,
            "mention": 0.05,
            "repost": 0.10,
        })
        self.edge_threshold = self.coord_config.get("edge_threshold", 0.3)

    def compute_shared_url_score(self, urls_a: set, urls_b: set) -> float:
        """
        Compute shared URL score between two accounts.

        Jaccard-like: |intersection| / max(|A|, |B|)
        """
        if not urls_a and not urls_b:
            return 0.0
        intersection = len(urls_a & urls_b)
        max_size = max(len(urls_a), len(urls_b))
        return intersection / max_size if max_size > 0 else 0.0

    def compute_shared_hashtag_score(self, hashtags_a: set,
                                     hashtags_b: set) -> float:
        """
        Compute shared hashtag score.

        Formula: |common hashtags| / |unique hashtags across both|
        """
        all_hashtags = hashtags_a | hashtags_b
        if not all_hashtags:
            return 0.0
        common = len(hashtags_a & hashtags_b)
        return common / len(all_hashtags)

    def compute_shared_mention_score(self, mentions_a: set,
                                     mentions_b: set) -> float:
        """
        Compute shared mention score.

        Accounts that mention the same set of other accounts
        may be coordinated.
        """
        all_mentions = mentions_a | mentions_b
        if not all_mentions:
            return 0.0
        common = len(mentions_a & mentions_b)
        return common / len(all_mentions)

    def compute_repost_score(self, account_a: str, account_b: str,
                             reposts_df: pd.DataFrame) -> float:
        """
        Compute repost similarity score.

        If A reposts B or B reposts A, or they repost the same accounts,
        this indicates coordination.
        """
        # Direct repost relationship
        direct_repost = (
            ((reposts_df["accountid"] == account_a) &
             (reposts_df["repost_of_account"] == account_b)).any() or
            ((reposts_df["accountid"] == account_b) &
             (reposts_df["repost_of_account"] == account_a)).any()
        )

        # Shared repost targets
        targets_a = set(
            reposts_df[reposts_df["accountid"] == account_a]["repost_of_account"].dropna()
        )
        targets_b = set(
            reposts_df[reposts_df["accountid"] == account_b]["repost_of_account"].dropna()
        )

        all_targets = targets_a | targets_b
        if not all_targets:
            return 1.0 if direct_repost else 0.0

        shared_targets = len(targets_a & targets_b)
        shared_ratio = shared_targets / len(all_targets)

        # Combine direct repost + shared targets
        score = 0.5 * float(direct_repost) + 0.5 * shared_ratio
        return score

    def compute_coordination_score(self, features: Dict[str, float],
                                   weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute weighted coordination score from individual feature scores.

        coordination_score = w_sem * sem + w_temp * temp + w_url * url
                          + w_ht * hashtag + w_rep * repost + w_ment * mention

        All six component scores are expected in [0, 1]:
        - semantic: sigmoid of cross-encoder logit, or cosine of L2-normalized
          bi-encoder embeddings (theoretically [-1, 1]; in practice > 0 for
          candidate pairs, which are floor-filtered at retrieval time)
        - temporal: exp(-delta/tau)
        - url: |intersection| / max(|A|, |B|); hashtag/mention: Jaccard
        - repost: 0.5 * direct + 0.5 * shared-target ratio
        With the default weights (sum = 1.0) the result is therefore in [0, 1].

        Args:
            features: dict of component scores (missing keys count as 0).
            weights: optional weight override; defaults to the configured
                fixed weights. Used by the learned-weight mode (item 19),
                where weights are fit by logistic regression on labeled data.
        """
        w = weights if weights is not None else self.weights
        score = (
            w["semantic"] * features.get("semantic_similarity", 0) +
            w["temporal"] * features.get("temporal_score", 0) +
            w["url"] * features.get("shared_url_score", 0) +
            w["hashtag"] * features.get("shared_hashtag_score", 0) +
            w["mention"] * features.get("shared_mention_score", 0) +
            w["repost"] * features.get("repost_score", 0)
        )
        return float(score)

    def learn_weights(self, account_df: pd.DataFrame,
                      edge_features_df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Learn the six coordination weights via logistic regression (item 19).

        Fits per-account aggregated coordination signals against IO-vs-control
        labels and returns non-negative weights normalized to sum to 1.0, so
        the weighted sum keeps the same [0, 1] scale as the fixed weights.
        Returns None when the data does not support fitting (missing labels,
        single class, or degenerate features) - callers should fall back to
        the configured fixed weights.
        """
        from sklearn.linear_model import LogisticRegression

        signal_cols = [
            ("semantic", "semantic_similarity"),
            ("temporal", "temporal_score"),
            ("url", "shared_url_score"),
            ("hashtag", "shared_hashtag_score"),
            ("mention", "shared_mention_score"),
            ("repost", "repost_score"),
        ]
        col_names = [c for _, c in signal_cols]

        if (edge_features_df is None or edge_features_df.empty
                or "is_io" not in account_df.columns
                or not all(c in edge_features_df.columns for c in col_names)):
            logger.warning("Cannot learn weights: missing signals or labels")
            return None

        stacked = pd.concat([
            edge_features_df[["account_a"] + col_names].rename(
                columns={"account_a": "account"}),
            edge_features_df[["account_b"] + col_names].rename(
                columns={"account_b": "account"}),
        ], ignore_index=True)
        per_account = stacked.groupby("account")[col_names].mean()

        X = (
            per_account.reindex(account_df["accountid"].values)
            .reindex(columns=col_names)
            .fillna(0.0)
            .values
        )
        y = account_df["is_io"].astype(int).values

        if len(np.unique(y)) < 2 or len(X) < 10:
            logger.warning("Cannot learn weights: need >=2 classes and >=10 accounts")
            return None

        try:
            lr = LogisticRegression(max_iter=1000, class_weight="balanced")
            lr.fit(X, y)
        except Exception as e:
            logger.warning(f"Weight learning failed: {e}")
            return None

        coefs = np.abs(lr.coef_[0]).astype(float)
        total = coefs.sum()
        if total <= 0:
            logger.warning("Learned weights are degenerate (all zero)")
            return None

        learned = {
            name: float(coef / total)
            for (name, _), coef in zip(signal_cols, coefs)
        }
        logger.info(
            "Learned coordination weights (sum=%.3f): %s",
            sum(learned.values()),
            {k: round(v, 3) for k, v in learned.items()},
        )
        return learned

    def compute_account_pair_features(
        self,
        account_a: str,
        account_b: str,
        df: pd.DataFrame,
        semantic_score: float = 0.0,
        temporal_score: float = 0.0,
    ) -> Dict[str, float]:
        """Compute all edge features for a single account pair."""
        # Get posts for each account
        posts_a = df[df["accountid"] == account_a]
        posts_b = df[df["accountid"] == account_b]

        # Extract sets of content
        urls_a = set()
        urls_b = set()
        hashtags_a = set()
        hashtags_b = set()
        mentions_a = set()
        mentions_b = set()

        for urls in posts_a["url_list"]:
            if isinstance(urls, list):
                urls_a.update(urls)
        for urls in posts_b["url_list"]:
            if isinstance(urls, list):
                urls_b.update(urls)

        for tags in posts_a["hashtag_list"]:
            if isinstance(tags, list):
                hashtags_a.update(tags)
        for tags in posts_b["hashtag_list"]:
            if isinstance(tags, list):
                hashtags_b.update(tags)

        for mentions in posts_a["mention_list"]:
            if isinstance(mentions, list):
                mentions_a.update(mentions)
        for mentions in posts_b["mention_list"]:
            if isinstance(mentions, list):
                mentions_b.update(mentions)

        # Compute individual scores
        shared_url_score = self.compute_shared_url_score(urls_a, urls_b)
        shared_hashtag_score = self.compute_shared_hashtag_score(hashtags_a, hashtags_b)
        shared_mention_score = self.compute_shared_mention_score(mentions_a, mentions_b)
        repost_score = self.compute_repost_score(
            account_a, account_b,
            df[["accountid", "repost_of_account"]].dropna()
        )

        features = {
            "account_a": account_a,
            "account_b": account_b,
            "semantic_similarity": semantic_score,
            "temporal_score": temporal_score,
            "shared_url_score": shared_url_score,
            "shared_url_count": len(urls_a & urls_b),
            "shared_hashtag_score": shared_hashtag_score,
            "shared_hashtag_count": len(hashtags_a & hashtags_b),
            "shared_mention_score": shared_mention_score,
            "shared_mention_count": len(mentions_a & mentions_b),
            "repost_score": repost_score,
            "common_content_ratio": (
                (len(urls_a & urls_b) + len(hashtags_a & hashtags_b) +
                 len(mentions_a & mentions_b)) /
                max(len(urls_a | urls_b) + len(hashtags_a | hashtags_b) +
                    len(mentions_a | mentions_b), 1)
            ),
        }

        # Compute overall coordination score
        features["coordination_score"] = self.compute_coordination_score(features)

        return features

    def compute_batch_edge_features(
        self,
        account_pairs: List[Tuple[str, str]],
        df: pd.DataFrame,
        semantic_scores: Optional[Dict[Tuple[str, str], float]] = None,
        temporal_scores: Optional[Dict[Tuple[str, str], float]] = None,
        fallback_pairs: Optional[set] = None,
        evidence_scores: Optional[Dict[Tuple[str, str], list]] = None,
        degradations=None,
    ) -> pd.DataFrame:
        """Compute edge features for a batch of account pairs.

        Pre-groups data by account for efficiency.

        Args:
            account_pairs: list of (acc_a, acc_b) tuples to score.
            df: posts DataFrame (needs url_list/hashtag_list/mention_list and
                repost_of_account columns).
            semantic_scores: pair_key -> active semantic similarity score.
            temporal_scores: pair_key -> temporal score.
            fallback_pairs: pair keys that came from the top-volume fallback
                rather than semantic retrieval (item 8d). These get
                semantic_fallback=True and must never look like a real
                "no semantic similarity found" result.
            evidence_scores: pair_key -> list of matched post-pair dicts
                (item 4d) attached to each edge as semantic evidence.
            degradations: optional DegradationTracker.
        """
        semantic_scores = semantic_scores or {}
        temporal_scores = temporal_scores or {}
        fallback_pairs = fallback_pairs or set()
        evidence_scores = evidence_scores or {}

        # Pre-group URLs, hashtags, mentions, and repost targets by account
        account_urls = {}
        account_hashtags = {}
        account_mentions = {}
        repost_df = df[["accountid", "repost_of_account"]].dropna()

        for acc, group in df.groupby("accountid"):
            urls = set()
            hashtags = set()
            mentions = set()
            for u in group["url_list"]:
                if isinstance(u, list):
                    urls.update(u)
            for h in group["hashtag_list"]:
                if isinstance(h, list):
                    hashtags.update(h)
            for m in group["mention_list"]:
                if isinstance(m, list):
                    mentions.update(m)
            account_urls[acc] = urls
            account_hashtags[acc] = hashtags
            account_mentions[acc] = mentions

        # Pre-group repost targets
        account_repost_targets = {}
        for acc, group in repost_df.groupby("accountid"):
            account_repost_targets[acc] = set(group["repost_of_account"])

        results = []
        for acc_a, acc_b in account_pairs:
            pair_key = tuple(sorted([acc_a, acc_b]))
            sem_score = semantic_scores.get(pair_key, 0.0)
            temp_score = temporal_scores.get(pair_key, 0.0)

            urls_a = account_urls.get(acc_a, set())
            urls_b = account_urls.get(acc_b, set())
            hashtags_a = account_hashtags.get(acc_a, set())
            hashtags_b = account_hashtags.get(acc_b, set())
            mentions_a = account_mentions.get(acc_a, set())
            mentions_b = account_mentions.get(acc_b, set())

            targets_a = account_repost_targets.get(acc_a, set())
            targets_b = account_repost_targets.get(acc_b, set())
            all_targets = targets_a | targets_b
            shared_targets = len(targets_a & targets_b)

            # Direct repost check
            direct_repost = (
                (acc_b in targets_a) or (acc_a in targets_b)
            )
            repost_score = 0.5 * float(direct_repost) + 0.5 * (shared_targets / len(all_targets)) if all_targets else float(direct_repost)

            shared_url_score = self.compute_shared_url_score(urls_a, urls_b)
            shared_hashtag_score = self.compute_shared_hashtag_score(hashtags_a, hashtags_b)
            shared_mention_score = self.compute_shared_mention_score(mentions_a, mentions_b)

            features = {
                "account_a": acc_a,
                "account_b": acc_b,
                "semantic_similarity": sem_score,
                "temporal_score": temp_score,
                "shared_url_score": shared_url_score,
                "shared_url_count": len(urls_a & urls_b),
                "shared_hashtag_score": shared_hashtag_score,
                "shared_hashtag_count": len(hashtags_a & hashtags_b),
                "shared_mention_score": shared_mention_score,
                "shared_mention_count": len(mentions_a & mentions_b),
                "repost_score": repost_score,
                # Item 8d: pairs from the top-volume fallback are tagged so
                # they can never be mistaken for genuinely-scored pairs.
                "semantic_fallback": pair_key in fallback_pairs,
                # Item 4d: matched post pairs that drove the semantic score,
                # kept on the edge for auditability/explanations.
                "semantic_evidence": evidence_scores.get(pair_key, []),
                "common_content_ratio": (
                    (len(urls_a & urls_b) + len(hashtags_a & hashtags_b) +
                     len(mentions_a & mentions_b)) /
                    max(len(urls_a | urls_b) + len(hashtags_a | hashtags_b) +
                        len(mentions_a | mentions_b), 1)
                ),
            }
            features["coordination_score"] = self.compute_coordination_score(features)
            results.append(features)

        result_df = pd.DataFrame(results)
        if degradations is not None and fallback_pairs:
            degradations.record(
                "semantic_retrieval",
                f"{len(fallback_pairs)} account pairs from top-volume fallback "
                "(semantic score 0, tagged semantic_fallback=True)",
            )
        return result_df
