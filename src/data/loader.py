"""
Data loader for the Labeled Information Operations dataset.

Dataset: "Labeled Datasets for Research on Information Operations"
Source: https://zenodo.org/records/14141550

Expected CSV columns:
    postid, post_text, application_name, post_language,
    in_reply_to_postid, in_reply_to_accountid, post_time,
    accountid, account_profile_description, follower_count,
    following_count, account_creation_date, is_repost,
    reposted_accountid, reposted_postid, hashtags, urls,
    account_mentions, is_control
"""

import os
import glob
import logging
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and inspect the Labeled Information Operations dataset."""

    REQUIRED_COLUMNS = [
        "postid", "post_text", "post_time", "accountid", "is_control"
    ]

    ALL_COLUMNS = [
        "postid", "post_text", "application_name", "post_language",
        "in_reply_to_postid", "in_reply_to_accountid", "post_time",
        "accountid", "account_profile_description", "follower_count",
        "following_count", "account_creation_date", "is_repost",
        "reposted_accountid", "reposted_postid", "hashtags", "urls",
        "account_mentions", "is_control"
    ]

    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = config["data"]["raw_dir"]
        self.processed_dir = config["data"]["processed_dir"]
        self.campaign = config["data"]["campaign"]
        self.max_files = config["data"].get("max_files_per_campaign", 10)
        self.max_posts = config["data"].get("max_posts", None)
        self.subsample = config["data"].get("subsample_fraction", None)
        self.seed = config["data"].get("random_seed", 42)

    def discover_files(self, campaign: Optional[str] = None) -> List[str]:
        """Discover CSV files for a given campaign."""
        campaign = campaign or self.campaign
        search_patterns = [
            os.path.join(self.raw_dir, f"{campaign}*", "*.csv"),
            os.path.join(self.raw_dir, f"{campaign}*", "*.CSV"),
            os.path.join(self.raw_dir, f"{campaign}*.csv"),
            os.path.join(self.raw_dir, "*.csv"),
        ]

        files = []
        for pattern in search_patterns:
            files = sorted(glob.glob(pattern))
            if files:
                break

        if not files:
            logger.warning(f"No CSV files found for campaign '{campaign}' "
                         f"in {self.raw_dir}")
        else:
            logger.info(f"Found {len(files)} CSV files for campaign '{campaign}'")

        # Limit files for memory management
        if self.max_files and len(files) > self.max_files:
            logger.info(f"Limiting to {self.max_files} files (out of {len(files)})")
            files = files[:self.max_files]

        return files

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Load a single CSV file with validation."""
        try:
            df = pd.read_csv(filepath, low_memory=False)
            logger.info(f"Loaded {filepath}: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return pd.DataFrame()

    def validate_columns(self, df: pd.DataFrame, filepath: str) -> bool:
        """Validate that a DataFrame has the required columns."""
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            logger.error(f"{filepath} is missing required columns: {missing}")
            return False
        return True

    def load_campaign(self, campaign: Optional[str] = None) -> pd.DataFrame:
        """Load all data for a campaign."""
        campaign = campaign or self.campaign
        files = self.discover_files(campaign)

        if not files:
            raise FileNotFoundError(
                f"No data files found for campaign '{campaign}' in {self.raw_dir}. "
                f"Please download the dataset from https://zenodo.org/records/14141550 "
                f"and place the CSV files in {self.raw_dir}."
            )

        dfs = []
        for f in files:
            df = self.load_csv(f)
            if df.empty:
                continue
            if not self.validate_columns(df, f):
                logger.warning(f"Skipping {f} due to missing columns")
                continue
            dfs.append(df)

        if not dfs:
            raise ValueError(f"No valid data loaded for campaign '{campaign}'")

        combined = pd.concat(dfs, ignore_index=True)

        # Apply max_posts limit
        if self.max_posts and len(combined) > self.max_posts:
            combined = combined.sample(n=self.max_posts, random_state=self.seed)
            logger.info(f"Subsampled to {self.max_posts} posts")

        # Apply subsample fraction
        if self.subsample and self.subsample < 1.0:
            combined = combined.sample(
                frac=self.subsample, random_state=self.seed
            )
            logger.info(f"Subsampled to fraction {self.subsample}: "
                       f"{len(combined)} posts")

        logger.info(f"Total loaded: {len(combined)} posts for campaign '{campaign}'")
        return combined

    def load_all_campaigns(self) -> pd.DataFrame:
        """Load data for all available campaigns."""
        campaign_dirs = glob.glob(os.path.join(self.raw_dir, "*"))
        campaign_dirs = [d for d in campaign_dirs if os.path.isdir(d)]

        if not campaign_dirs:
            # Try loading all CSVs directly from raw_dir
            return self.load_campaign("")

        dfs = []
        for d in campaign_dirs:
            campaign_name = os.path.basename(d)
            try:
                df = self.load_campaign(campaign_name)
                df["campaign"] = campaign_name
                dfs.append(df)
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping campaign {campaign_name}: {e}")

        if not dfs:
            raise ValueError("No campaigns loaded successfully")

        return pd.concat(dfs, ignore_index=True)

    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics about the dataset."""
        info = {
            "total_posts": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
        }

        if "accountid" in df.columns:
            info["unique_accounts"] = df["accountid"].nunique()

        if "is_control" in df.columns:
            control_counts = df["is_control"].value_counts()
            info["control_distribution"] = control_counts.to_dict()

        if "post_time" in df.columns:
            try:
                times = pd.to_datetime(df["post_time"], errors="coerce")
                info["time_range"] = {
                    "start": str(times.min()),
                    "end": str(times.max()),
                }
            except Exception:
                pass

        if "hashtags" in df.columns:
            non_null = df["hashtags"].dropna()
            info["hashtag_coverage"] = len(non_null) / len(df) if len(df) > 0 else 0

        if "urls" in df.columns:
            non_null = df["urls"].dropna()
            info["url_coverage"] = len(non_null) / len(df) if len(df) > 0 else 0

        return info

    def save_processed(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """Save processed data to disk."""
        os.makedirs(self.processed_dir, exist_ok=True)
        path = os.path.join(self.processed_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved processed data to {path}: {len(df)} rows")
        return path

    def load_processed(self, filename: str = "processed_data.csv") -> pd.DataFrame:
        """Load processed data from disk."""
        path = os.path.join(self.processed_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed data not found at {path}")
        return pd.read_csv(path, low_memory=False)


def create_sample_dataset(output_dir: str = "data/raw", num_accounts: int = 50,
                         posts_per_account: int = 20):
    """
    Create a small synthetic sample dataset for testing the pipeline.
    Uses the same schema as the real dataset.
    """
    np.random.seed(42)
    records = []

    # Create IO accounts (is_control = False)
    io_accounts = [f"IO_{i:04d}" for i in range(num_accounts // 2)]
    # Create control accounts (is_control = True)
    control_accounts = [f"CTRL_{i:04d}" for i in range(num_accounts // 2)]

    hashtags_pool = [
        "#politics", "#election", "#news", "#breaking", "#russia",
        "#ukraine", "#war", "#propaganda", "#fakenews", "#media",
        "#corruption", "#protest", "#freedom", "#democracy", "#government"
    ]
    urls_pool = [f"url_hash_{i}" for i in range(20)]
    mentions_pool = [f"mention_{i}" for i in range(30)]

    # IO accounts tend to share similar hashtags/urls in coordinated fashion
    io_hashtags = np.random.choice(hashtags_pool, 5, replace=False).tolist()
    io_urls = np.random.choice(urls_pool, 3, replace=False).tolist()

    base_time = pd.Timestamp("2019-01-01")

    for account in io_accounts:
        # Coordinated posting: similar timestamps
        offset_minutes = np.random.randint(0, 60)
        for j in range(posts_per_account):
            post_time = base_time + pd.Timedelta(
                days=np.random.randint(0, 365),
                minutes=offset_minutes + np.random.randint(-5, 5)
            )
            text = f"IO post about {np.random.choice(io_hashtags)} topic"
            hashtags = np.random.choice(
                io_hashtags, size=min(3, len(io_hashtags)), replace=False
            ).tolist()
            urls = np.random.choice(
                io_urls, size=min(2, len(io_urls)), replace=False
            ).tolist()
            mentions = np.random.choice(
                mentions_pool, size=np.random.randint(0, 3), replace=False
            ).tolist()

            records.append({
                "postid": f"post_{account}_{j}",
                "post_text": text,
                "application_name": "app_hash_1",
                "post_language": "en",
                "in_reply_to_postid": None,
                "in_reply_to_accountid": None,
                "post_time": post_time.isoformat(),
                "accountid": account,
                "account_profile_description": "IO profile",
                "follower_count": np.random.randint(10, 500),
                "following_count": np.random.randint(50, 1000),
                "account_creation_date": "2018-01-01",
                "is_repost": np.random.random() < 0.3,
                "reposted_accountid": np.random.choice(io_accounts) if np.random.random() < 0.3 else None,
                "reposted_postid": None,
                "hashtags": ",".join(hashtags) if hashtags else None,
                "urls": ",".join(urls) if urls else None,
                "account_mentions": ",".join(mentions) if mentions else None,
                "is_control": False,
            })

    for account in control_accounts:
        for j in range(posts_per_account):
            post_time = base_time + pd.Timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(0, 24)
            )
            text = f"Control post about {np.random.choice(hashtags_pool)} topic"
            hashtags = np.random.choice(
                hashtags_pool, size=np.random.randint(0, 4), replace=False
            ).tolist()

            records.append({
                "postid": f"post_{account}_{j}",
                "post_text": text,
                "application_name": "app_hash_2",
                "post_language": "en",
                "in_reply_to_postid": None,
                "in_reply_to_accountid": None,
                "post_time": post_time.isoformat(),
                "accountid": account,
                "account_profile_description": "Control profile",
                "follower_count": np.random.randint(50, 5000),
                "following_count": np.random.randint(100, 2000),
                "account_creation_date": "2015-01-01",
                "is_repost": np.random.random() < 0.1,
                "reposted_accountid": None,
                "reposted_postid": None,
                "hashtags": ",".join(hashtags) if hashtags else None,
                "urls": None,
                "account_mentions": None,
                "is_control": True,
            })

    df = pd.DataFrame(records)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sample_campaign.csv")
    df.to_csv(path, index=False)
    logger.info(f"Created sample dataset with {len(df)} posts at {path}")
    return path
