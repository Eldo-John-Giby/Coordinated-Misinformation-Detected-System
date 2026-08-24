"""
Temporal coordination features.

Detects temporal proximity and synchronized posting patterns
between accounts.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalFeatures:
    """Extract temporal coordination features."""

    def __init__(self, config: dict):
        self.config = config
        self.temp_config = config.get("temporal", {})
        self.tau = self.temp_config.get("tau", 3600)  # 1 hour default
        self.burst_window = self.temp_config.get("burst_window", 300)  # 5 min
        self.burst_min_posts = self.temp_config.get("burst_min_posts", 3)

    def compute_temporal_proximity(self, times_a: pd.Series,
                                   times_b: pd.Series) -> Dict[str, float]:
        """
        Compute temporal coordination between two accounts' posting times.

        Returns:
            temporal_score: exp(-mean_min_delta / tau) — exponential decay
            mean_min_delta: average minimum time gap between posts
            min_delta: minimum time gap found
            synchronized_ratio: fraction of posts within burst_window
        """
        if len(times_a) == 0 or len(times_b) == 0:
            return {
                "temporal_score": 0.0,
                "temporal_mean_min_delta": float("inf"),
                "temporal_min_delta": float("inf"),
                "temporal_synchronized_ratio": 0.0,
            }

        # Convert to numpy datetime64 for vectorized operations
        a_times = times_a.values.astype(np.int64)  # nanoseconds
        b_times = times_b.values.astype(np.int64)

        # For each post in A, find the nearest post in B
        min_deltas_a = np.array([
            np.min(np.abs(b_times - t)) for t in a_times
        ])

        # For each post in B, find the nearest post in A
        min_deltas_b = np.array([
            np.min(np.abs(a_times - t)) for t in b_times
        ])

        # Average minimum delta across both directions (in seconds)
        mean_min_delta = (min_deltas_a.mean() + min_deltas_b.mean()) / 2 / 1e9

        # Minimum delta
        min_delta = min(min_deltas_a.min(), min_deltas_b.min()) / 1e9

        # Temporal score: exponential decay
        temporal_score = float(np.exp(-mean_min_delta / self.tau))

        # Synchronized posting ratio
        all_min_deltas = np.concatenate([min_deltas_a, min_deltas_b]) / 1e9
        synchronized_ratio = float(np.mean(all_min_deltas <= self.burst_window))

        return {
            "temporal_score": temporal_score,
            "temporal_mean_min_delta": mean_min_delta,
            "temporal_min_delta": min_delta,
            "temporal_synchronized_ratio": synchronized_ratio,
        }

    def detect_bursts(self, times: pd.Series,
                      window_seconds: Optional[int] = None,
                      min_posts: Optional[int] = None
                      ) -> List[Dict]:
        """
        Detect posting bursts from a single account.

        A burst is a period where posts cluster within a short time window.
        """
        window_seconds = window_seconds or self.burst_window
        min_posts = min_posts or self.burst_min_posts

        if len(times) < min_posts:
            return []

        sorted_times = times.sort_values().values
        bursts = []

        i = 0
        while i < len(sorted_times):
            window_end = sorted_times[i] + np.timedelta64(window_seconds, "s")
            in_window = sorted_times[sorted_times <= window_end]

            if len(in_window) >= min_posts:
                bursts.append({
                    "start_time": pd.Timestamp(sorted_times[i]),
                    "end_time": pd.Timestamp(in_window[-1]),
                    "post_count": len(in_window),
                    "duration_seconds": (
                        in_window[-1] - sorted_times[i]
                    ) / np.timedelta64(1, "s"),
                })
                i += len(in_window)  # Skip past this burst
            else:
                i += 1

        return bursts

    def compute_synchronization_score(self, times_a: pd.Series,
                                      times_b: pd.Series) -> float:
        """
        Compute synchronization score using efficient nearest-neighbor alignment.

        Higher score means more synchronized posting behavior.
        Uses sorted merge instead of brute-force window iteration.
        """
        if len(times_a) < 2 or len(times_b) < 2:
            return 0.0

        a_ns = np.sort(times_a.values.astype(np.int64))
        b_ns = np.sort(times_b.values.astype(np.int64))
        window_ns = self.burst_window * 1_000_000_000  # seconds to nanoseconds

        # For each post in A, check if there's a post in B within the burst window
        aligned_count = 0
        j = 0
        for t_a in a_ns:
            # Advance j to the first B post within window
            while j < len(b_ns) - 1 and b_ns[j] < t_a - window_ns:
                j += 1
            # Count B posts within window [t_a - window, t_a + window]
            k = j
            while k < len(b_ns) and b_ns[k] <= t_a + window_ns:
                aligned_count += 1
                k += 1

        return aligned_count / max(len(a_ns), len(b_ns))

    def compute_account_pair_temporal_features(
        self,
        df: pd.DataFrame,
        account_a: str,
        account_b: str
    ) -> Dict[str, float]:
        """Compute all temporal features for an account pair."""
        posts_a = df[df["accountid"] == account_a]["post_time"]
        posts_b = df[df["accountid"] == account_b]["post_time"]

        proximity = self.compute_temporal_proximity(posts_a, posts_b)
        synchronization = self.compute_synchronization_score(posts_a, posts_b)

        proximity["temporal_synchronization"] = synchronization
        return proximity

    def compute_batch_temporal_features(
        self,
        account_pairs: List[Tuple[str, str]],
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute temporal features for a batch of account pairs.

        Pre-groups post times by account for efficiency.
        """
        # Pre-group timestamps by account to avoid repeated DataFrame filtering
        account_times = {}
        for acc, group in df.groupby("accountid"):
            account_times[acc] = group["post_time"]

        results = []
        for acc_a, acc_b in account_pairs:
            times_a = account_times.get(acc_a, pd.Series(dtype="datetime64[ns]"))
            times_b = account_times.get(acc_b, pd.Series(dtype="datetime64[ns]"))

            proximity = self.compute_temporal_proximity(times_a, times_b)
            synchronization = self.compute_synchronization_score(times_a, times_b)
            proximity["temporal_synchronization"] = synchronization
            proximity["account_a"] = acc_a
            proximity["account_b"] = acc_b
            results.append(proximity)

        return pd.DataFrame(results)
