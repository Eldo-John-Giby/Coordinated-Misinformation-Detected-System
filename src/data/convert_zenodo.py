"""
Converter for the Zenodo Information Operations dataset.

Dataset: "Twitter dataset about Information Operations in Honduras and UAE"
Source: https://zenodo.org/records/13912659
Paper: Cima et al., "Coordinated Behavior in Information Operations on Twitter", IEEE Access 2024

Converts JSONL format to the CSV schema expected by the pipeline:
    postid, post_text, application_name, post_language,
    in_reply_to_postid, in_reply_to_accountid, post_time,
    accountid, account_profile_description, follower_count,
    following_count, account_creation_date, is_repost,
    reposted_accountid, reposted_postid, hashtags, urls,
    account_mentions, is_control
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Zenodo record and file URLs
ZENODO_RECORD = "13912659"
ZENODO_BASE = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

# Available campaigns
CAMPAIGNS = {
    "Honduras": {
        "bad": f"{ZENODO_BASE}/honduras-bad-anonymized/content",
        "good": f"{ZENODO_BASE}/honduras-good-anonymized/content",
    },
    "UAE": {
        "bad": f"{ZENODO_BASE}/uae-bad-anonymized/content",
        "good": f"{ZENODO_BASE}/uae-good-anonymized/content",
    },
}

# Expected pipeline CSV columns
PIPELINE_COLUMNS = [
    "postid", "post_text", "application_name", "post_language",
    "in_reply_to_postid", "in_reply_to_accountid", "post_time",
    "accountid", "account_profile_description", "follower_count",
    "following_count", "account_creation_date", "is_repost",
    "reposted_accountid", "reposted_postid", "hashtags", "urls",
    "account_mentions", "is_control",
]


def parse_json_array(val) -> list:
    """Parse a JSON array string or return empty list."""
    if val is None or val == "[]" or val == "":
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def epoch_ms_to_iso(epoch_ms) -> str:
    """Convert epoch milliseconds to ISO 8601 string."""
    try:
        ts = float(epoch_ms)
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except (ValueError, TypeError, OSError):
        return ""


def convert_jsonl_to_pipeline_df(jsonl_path: str, max_posts: Optional[int] = None) -> pd.DataFrame:
    """
    Convert a single Zenodo JSONL file to pipeline-compatible DataFrame.

    Args:
        jsonl_path: Path to the JSONL file (or URL via pandas).
        max_posts: Maximum number of posts to load (None = all).

    Returns:
        DataFrame with pipeline-compatible columns.
    """
    logger.info(f"Converting {jsonl_path}...")

    # Read JSONL - each line is a JSON object
    records = []
    count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                continue

            # Map fields
            hashtags = parse_json_array(obj.get("hashtags"))
            urls = parse_json_array(obj.get("urls"))
            mentions = parse_json_array(obj.get("user_mentions"))

            retweet_userid = obj.get("retweet_userid")
            is_repost = retweet_userid is not None and retweet_userid != 0

            record = {
                "postid": str(obj.get("tweetid", "")),
                "post_text": obj.get("tweet_text", ""),
                "application_name": "twitter",
                "post_language": obj.get("tweet_language", ""),
                "in_reply_to_postid": str(obj["in_reply_to_tweetid"]) if obj.get("in_reply_to_tweetid") else None,
                "in_reply_to_accountid": str(obj["in_reply_to_userid"]) if obj.get("in_reply_to_userid") else None,
                "post_time": epoch_ms_to_iso(obj.get("tweet_time")),
                "accountid": str(obj.get("userid", "")),
                "account_profile_description": "",
                "follower_count": int(obj.get("follower_count", 0) or 0),
                "following_count": int(obj.get("following_count", 0) or 0),
                "account_creation_date": obj.get("account_creation_date", ""),
                "is_repost": is_repost,
                "reposted_accountid": str(int(retweet_userid)) if retweet_userid and retweet_userid != 0 else None,
                "reposted_postid": str(int(obj["retweet_tweetid"])) if obj.get("retweet_tweetid") else None,
                "hashtags": ",".join(str(h) for h in hashtags) if hashtags else None,
                "urls": ",".join(str(u) for u in urls) if urls else None,
                "account_mentions": ",".join(str(m) for m in mentions) if mentions else None,
                # good=1 means genuine/control, good=0 means IO/malicious
                "is_control": bool(obj.get("good", 1)),
            }
            records.append(record)
            count += 1

            if max_posts and count >= max_posts:
                logger.info(f"Reached max_posts limit ({max_posts})")
                break

    df = pd.DataFrame(records)
    logger.info(f"Converted {len(df)} records from {jsonl_path}")
    return df


def stream_jsonl_from_url(url: str, max_posts: Optional[int] = None) -> list:
    """
    Stream a JSONL file from a URL, reading line by line.
    Only loads up to max_posts lines into memory.
    """
    import urllib.request
    import ssl

    records = []
    count = 0

    logger.info(f"Streaming from {url}...")
    ssl_ctx = ssl.create_default_context()

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as response:
        for line in response:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
                count += 1
                if count % 10000 == 0:
                    logger.info(f"  ... streamed {count} records")
                if max_posts and count >= max_posts:
                    logger.info(f"  Reached max_posts limit ({max_posts})")
                    break
            except json.JSONDecodeError:
                continue

    logger.info(f"Streamed {len(records)} records total")
    return records


def download_and_convert(
    campaign: str,
    output_dir: str = "data/raw",
    max_posts: Optional[int] = None,
    include_good: bool = True,
) -> str:
    """
    Download a campaign from Zenodo and convert to pipeline CSV.
    Streams data line-by-line to handle large files efficiently.

    Args:
        campaign: Campaign name ("Honduras" or "UAE").
        output_dir: Directory to save the CSV.
        max_posts: Max posts per file (None = all).
        include_good: Whether to include genuine (good) tweets.

    Returns:
        Path to the saved CSV file.
    """
    if campaign not in CAMPAIGNS:
        raise ValueError(f"Unknown campaign '{campaign}'. Available: {list(CAMPAIGNS.keys())}")

    campaign_urls = CAMPAIGNS[campaign]
    os.makedirs(output_dir, exist_ok=True)

    # Stream and convert bad (IO) tweets
    logger.info(f"Downloading {campaign} IO tweets...")
    bad_records = stream_jsonl_from_url(campaign_urls["bad"], max_posts=max_posts)
    for r in bad_records:
        r["good"] = 0  # Ensure label is set

    # Save temporary JSONL, then convert
    tmp_bad = os.path.join(output_dir, f"_{campaign}_bad_tmp.jsonl")
    with open(tmp_bad, "w", encoding="utf-8") as f:
        for r in bad_records:
            f.write(json.dumps(r) + "\n")
    bad_converted = convert_jsonl_to_pipeline_df(tmp_bad)
    os.remove(tmp_bad)

    if include_good:
        # Stream and convert good (genuine) tweets
        logger.info(f"Downloading {campaign} genuine tweets...")
        good_records = stream_jsonl_from_url(campaign_urls["good"], max_posts=max_posts)
        for r in good_records:
            r["good"] = 1  # Ensure label is set

        tmp_good = os.path.join(output_dir, f"_{campaign}_good_tmp.jsonl")
        with open(tmp_good, "w", encoding="utf-8") as f:
            for r in good_records:
                f.write(json.dumps(r) + "\n")
        good_converted = convert_jsonl_to_pipeline_df(tmp_good)
        os.remove(tmp_good)

        # Combine
        combined = pd.concat([bad_converted, good_converted], ignore_index=True)
    else:
        combined = bad_converted

    # Ensure all pipeline columns exist
    for col in PIPELINE_COLUMNS:
        if col not in combined.columns:
            combined[col] = None

    # Reorder columns
    combined = combined[PIPELINE_COLUMNS]

    # Save
    campaign_dir = os.path.join(output_dir, campaign)
    os.makedirs(campaign_dir, exist_ok=True)
    output_path = os.path.join(campaign_dir, f"{campaign.lower()}_io.csv")
    combined.to_csv(output_path, index=False)

    logger.info(
        f"Saved {len(combined)} posts to {output_path} "
        f"(IO: {(~combined['is_control']).sum()}, Control: {combined['is_control'].sum()})"
    )
    return output_path


def main():
    """CLI entry point for the converter."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and convert Zenodo IO dataset to pipeline format"
    )
    parser.add_argument(
        "--campaign",
        choices=["Honduras", "UAE", "all"],
        default="Honduras",
        help="Campaign to download (default: Honduras)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help="Max posts per file (for quick testing)",
    )
    parser.add_argument(
        "--bad-only",
        action="store_true",
        help="Only download IO (bad) tweets, skip genuine tweets",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    campaigns = ["Honduras", "UAE"] if args.campaign == "all" else [args.campaign]

    for campaign in campaigns:
        try:
            path = download_and_convert(
                campaign=campaign,
                output_dir=args.output_dir,
                max_posts=args.max_posts,
                include_good=not args.bad_only,
            )
            print(f"[OK] {campaign}: {path}")
        except Exception as e:
            print(f"[FAIL] {campaign}: {e}")
            logger.exception(f"Failed to convert {campaign}")


if __name__ == "__main__":
    main()
