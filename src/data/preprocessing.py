"""
Data preprocessing pipeline for the Labeled Information Operations dataset.

Handles:
- Malformed record removal
- Missing value handling
- Timestamp parsing
- Text normalization
- URL/hashtag/mention extraction
- Repost relationship identification
- Account-to-integer node ID mapping
"""

import re
import logging
from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocess raw IO dataset into clean, analysis-ready format."""

    def __init__(self, config: dict):
        self.config = config
        self.pp_config = config.get("preprocessing", {})
        self.lowercase = self.pp_config.get("lowercase", True)
        self.remove_urls = self.pp_config.get("remove_urls_from_text", True)
        self.remove_mentions_text = self.pp_config.get("remove_mentions_from_text", False)
        self.min_text_length = self.pp_config.get("min_text_length", 3)
        self.drop_malformed = self.pp_config.get("drop_malformed", True)

        # Max tolerated fraction of rows with unparseable timestamps (item 8f):
        # above this the data is considered badly parsed and we fail loudly
        # instead of continuing on corrupted time data.
        self.max_timestamp_drop_rate = self.pp_config.get("max_timestamp_drop_rate", 0.20)

        # Mappings built during preprocessing
        self.account_to_node_id: Dict[str, int] = {}
        self.node_id_to_account: Dict[int, str] = {}

        # Optional DegradationTracker (set in process()); a silent sink by
        # default so standalone use of the preprocessor still works.
        self._degradations = None

    def _note_degradation(self, component: str, reason: str, detail: str = ""):
        """Record a degradation if a tracker is attached; else just warn."""
        if self._degradations is not None:
            self._degradations.record(component, reason, detail)
        else:
            logger.warning("DEGRADED [%s]: %s%s", component, reason,
                           f" ({detail})" if detail else "")

    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and report schema issues."""
        required = ["postid", "post_text", "post_time", "accountid", "is_control"]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset. "
                               f"Available columns: {list(df.columns)}")

        # Report missing columns
        expected = set([
            "postid", "post_text", "application_name", "post_language",
            "in_reply_to_postid", "in_reply_to_accountid", "post_time",
            "accountid", "account_profile_description", "follower_count",
            "following_count", "account_creation_date", "is_repost",
            "reposted_accountid", "reposted_postid", "hashtags", "urls",
            "account_mentions", "is_control"
        ])
        missing_cols = expected - set(df.columns)
        if missing_cols:
            logger.warning(f"Missing optional columns: {missing_cols}")

        logger.info(f"Schema validation passed. Shape: {df.shape}")
        return df

    def remove_malformed_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with critical missing or invalid data."""
        initial_len = len(df)

        # Remove rows where critical fields are completely invalid
        if self.drop_malformed:
            # Remove rows with null post_text or accountid
            df = df.dropna(subset=["postid", "accountid"])

            # Remove rows where post_text is not a string
            df = df[df["post_text"].apply(lambda x: isinstance(x, str) or pd.isna(x))]

            # Remove rows with empty post_text
            df = df[df["post_text"].str.strip().str.len() >= self.min_text_length]

        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} malformed records "
                       f"({removed/initial_len*100:.1f}%)")
        return df

    def parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and validate timestamp columns."""
        if "post_time" not in df.columns:
            logger.warning("No 'post_time' column found")
            return df

        df["post_time"] = pd.to_datetime(df["post_time"], errors="coerce")
        na_count = int(df["post_time"].isna().sum())
        if na_count > 0:
            drop_rate = na_count / len(df)
            logger.warning(f"{na_count} timestamps could not be parsed "
                         f"({drop_rate*100:.1f}%)")
            self._note_degradation(
                "timestamps",
                "unparseable timestamps dropped",
                f"{na_count} rows ({drop_rate*100:.2f}%)",
            )
            if drop_rate > self.max_timestamp_drop_rate:
                raise ValueError(
                    f"{drop_rate*100:.1f}% of timestamps are unparseable "
                    f"(>{self.max_timestamp_drop_rate*100:.0f}% threshold). "
                    "Refusing to continue on badly parsed time data."
                )
            # Drop rows with invalid timestamps
            df = df.dropna(subset=["post_time"])

        # Parse account creation date if available
        if "account_creation_date" in df.columns:
            df["account_creation_date"] = pd.to_datetime(
                df["account_creation_date"], errors="coerce"
            )

        logger.info(f"Parsed timestamps. Range: {df['post_time'].min()} "
                   f"to {df['post_time'].max()}")
        return df

    def normalize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text content."""
        if "post_text" not in df.columns:
            return df

        # Fill missing text
        df["post_text"] = df["post_text"].fillna(
            self.pp_config.get("fill_missing_text", "")
        )

        if self.lowercase:
            df["post_text"] = df["post_text"].str.lower()

        # Remove extra whitespace
        df["post_text"] = df["post_text"].str.strip()
        df["post_text"] = df["post_text"].str.replace(r"\s+", " ", regex=True)

        # Optionally remove URLs from text (keep the extracted url features separately)
        if self.remove_urls:
            df["post_text"] = df["post_text"].str.replace(
                r"https?://\S+", "", regex=True
            )

        # Optionally remove mentions from text
        if self.remove_mentions_text:
            df["post_text"] = df["post_text"].str.replace(r"@\S+", "", regex=True)

        return df

    def extract_urls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse URL field into a list."""
        if "urls" not in df.columns:
            df["url_list"] = [[] for _ in range(len(df))]
            self._note_degradation(
                "url_signal",
                "'urls' column missing; shared-URL score will be 0 for all pairs",
            )
            return df

        df["url_list"] = df["urls"].apply(
            lambda x: [u.strip() for u in str(x).split(",") if u.strip() and u.strip() != "nan"]
            if pd.notna(x) else []
        )
        return df

    def extract_hashtags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse hashtags field into a list."""
        if "hashtags" not in df.columns:
            df["hashtag_list"] = [[] for _ in range(len(df))]
            self._note_degradation(
                "hashtag_signal",
                "'hashtags' column missing; shared-hashtag score will be 0 for all pairs",
            )
            return df

        df["hashtag_list"] = df["hashtags"].apply(
            lambda x: [h.strip().lower() for h in str(x).split(",") if h.strip() and h.strip() != "nan"]
            if pd.notna(x) else []
        )
        return df

    def extract_mentions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse account_mentions field into a list."""
        if "account_mentions" not in df.columns:
            df["mention_list"] = [[] for _ in range(len(df))]
            self._note_degradation(
                "mention_signal",
                "'account_mentions' column missing; shared-mention score will be 0 for all pairs",
            )
            return df

        df["mention_list"] = df["account_mentions"].apply(
            lambda x: [m.strip() for m in str(x).split(",") if m.strip() and m.strip() != "nan"]
            if pd.notna(x) else []
        )
        return df

    def identify_reposts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify repost/retweet relationships."""
        if "is_repost" not in df.columns:
            df["is_repost"] = False
            df["repost_of_account"] = None
            self._note_degradation(
                "repost_signal",
                "'is_repost' column missing; repost score will be 0 for all pairs",
            )
            return df

        df["is_repost"] = df["is_repost"].fillna(False).astype(bool)

        if "reposted_accountid" in df.columns:
            df["repost_of_account"] = df["reposted_accountid"]
        else:
            df["repost_of_account"] = None

        repost_count = df["is_repost"].sum()
        logger.info(f"Identified {repost_count} reposts "
                   f"({repost_count/len(df)*100:.1f}%)")
        return df

    def build_account_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map account IDs to integer node IDs."""
        unique_accounts = sorted(df["accountid"].unique())
        self.account_to_node_id = {acc: idx for idx, acc in enumerate(unique_accounts)}
        self.node_id_to_account = {idx: acc for acc, idx in self.account_to_node_id.items()}

        df["node_id"] = df["accountid"].map(self.account_to_node_id)

        logger.info(f"Built account mappings: {len(unique_accounts)} unique accounts")
        return df

    def compute_account_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-account ground truth labels.

        An account is labeled as IO (is_control=False) if ANY of its posts
        is labeled as IO. This is because IO accounts may have pre-campaign
        or post-campaign organic posts.
        """
        if "is_control" not in df.columns:
            logger.warning("No 'is_control' column found")
            return df

        # Aggregate: account is IO if any post is from IO
        account_labels = df.groupby("accountid")["is_control"].all().reset_index()
        account_labels.columns = ["accountid", "account_is_control"]

        # Merge back
        df = df.merge(account_labels, on="accountid", how="left", suffixes=("", "_account"))

        io_accounts = (~account_labels["account_is_control"]).sum()
        control_accounts = account_labels["account_is_control"].sum()
        logger.info(f"Account labels: {io_accounts} IO accounts, "
                   f"{control_accounts} control accounts")

        return df

    def process(self, df: pd.DataFrame, degradations=None) -> pd.DataFrame:
        """Run the full preprocessing pipeline.

        Args:
            df: Raw posts DataFrame.
            degradations: Optional DegradationTracker; when provided, missing
                optional columns and timestamp-drop events are recorded there
                so they surface in pipeline_results.json (item 8).
        """
        self._degradations = degradations
        logger.info(f"Starting preprocessing. Input shape: {df.shape}")

        df = self.validate_schema(df)
        df = self.remove_malformed_records(df)
        df = self.parse_timestamps(df)
        df = self.normalize_text(df)
        df = self.extract_urls(df)
        df = self.extract_hashtags(df)
        df = self.extract_mentions(df)
        df = self.identify_reposts(df)
        df = self.build_account_mappings(df)
        df = self.compute_account_labels(df)
        self._degradations = None

        logger.info(f"Preprocessing complete. Output shape: {df.shape}")
        return df
