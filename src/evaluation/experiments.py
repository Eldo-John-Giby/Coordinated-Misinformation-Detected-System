"""
Experiment runner for reproducible experiments and ablation studies.

Runs:
1. Feature ablation experiments
2. Model comparison experiments
3. Cross-campaign evaluation
"""

import json
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run and manage experiments."""

    def __init__(self, config: dict):
        self.config = config
        self.exp_config = config.get("experiments", {})
        self.output_dir = self.exp_config.get("output_dir", "results")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_model_comparison(self,
                             model_results: Dict[str, Dict[str, float]]
                             ) -> pd.DataFrame:
        """
        Run model comparison experiment.

        Args:
            model_results: Dict mapping model names to their metrics

        Returns:
            Comparison DataFrame
        """
        rows = []
        for model_name, metrics in model_results.items():
            row = {"model": model_name}
            row.update(metrics)
            rows.append(row)

        comparison_df = pd.DataFrame(rows)

        # Save results
        output_path = os.path.join(self.output_dir, "model_comparison.csv")
        comparison_df.to_csv(output_path, index=False)
        logger.info(f"Model comparison saved to {output_path}")

        return comparison_df

    def save_experiment_config(self, config: dict):
        """Save experiment configuration for reproducibility."""
        output_path = os.path.join(self.output_dir, "experiment_config.json")
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        logger.info(f"Experiment config saved to {output_path}")
