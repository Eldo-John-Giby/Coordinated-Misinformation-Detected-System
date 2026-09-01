#!/usr/bin/env python3
"""
Main entry point for the Coordinated Misinformation Network Detection System.

Usage:
    python train.py                         # Run with sample data
    python train.py --config configs/config.yaml
    python train.py --campaign Armenia
    python train.py --sample                # Force sample data creation
"""

import argparse
import logging
import os
import sys
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join("configs", "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded config from {config_path}")
    else:
        logging.warning(f"Config not found at {config_path}, using defaults")
        config = get_default_config()

    return config


def get_default_config() -> dict:
    """Default configuration."""
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "campaign": "Armenia",
            "max_files_per_campaign": 10,
            "max_posts": None,
            "subsample_fraction": None,
            "random_seed": 42,
        },
        "preprocessing": {
            "lowercase": True,
            "remove_urls_from_text": True,
            "remove_mentions_from_text": False,
            "min_text_length": 3,
            "drop_malformed": True,
        },
        "nlp": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "batch_size": 256,
            "device": None,
        },
        "similarity": {
            "method": "faiss",
            "k_neighbors": 20,
            "semantic_threshold": 0.7,
            "faiss_index_type": "flat",
        },
        "temporal": {
            "tau": 3600,
            "burst_window": 300,
            "burst_min_posts": 3,
        },
        "coordination": {
            "edge_threshold": 0.3,
            "weights": {
                "semantic": 0.35,
                "temporal": 0.25,
                "url": 0.15,
                "hashtag": 0.10,
                "mention": 0.05,
                "repost": 0.10,
            },
            "max_edges_per_node": 50,
            "min_edge_score": 0.1,
        },
        "graph": {
            "use_node_features": True,
            "node_features": [
                "post_count", "avg_follower_count", "avg_following_count",
                "repost_ratio", "avg_hashtag_count", "avg_url_count",
                "avg_mention_count"
            ],
            "edge_features": [
                "semantic_similarity", "temporal_score",
                "shared_url_score", "shared_hashtag_score",
                "shared_mention_score", "repost_score"
            ],
        },
        "models": {
            "gcn": {
                "hidden_channels": 64,
                "num_layers": 2,
                "dropout": 0.5,
                "learning_rate": 0.001,
                "epochs": 100,
                "patience": 10,
            },
            "graphsage": {
                "hidden_channels": 64,
                "num_layers": 2,
                "dropout": 0.5,
                "learning_rate": 0.001,
                "epochs": 100,
                "patience": 10,
                "aggregator": "mean",
            },
            "baseline": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 20,
                    "random_state": 42,
                },
                "logistic_regression": {
                    "max_iter": 1000,
                    "random_state": 42,
                },
            },
        },
        "splitting": {
            "strategy": "account",
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        },
        "evaluation": {
            "classification_metrics": [
                "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
            ],
            "community_metrics": ["adjusted_rand_index", "normalized_mutual_info"],
        },
        "explainability": {
            "methods": ["feature_importance", "permutation_importance"],
            "permutation_runs": 10,
            "use_shap": False,
            "use_gnnexplainer": True,
        },
        "visualization": {
            "output_dir": "results/figures",
            "format": "png",
            "dpi": 150,
            "max_network_nodes": 200,
            "color_scheme": {
                "io_account": "#e74c3c",
                "control_account": "#2ecc71",
                "suspicious_group": "#e67e22",
                "background": "#ffffff",
            },
        },
        "experiments": {
            "ablation": True,
            "ablation_features": [
                "semantic", "temporal", "url", "hashtag", "mention", "repost"
            ],
            "cross_campaign": False,
            "output_dir": "results",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Coordinated Misinformation Network Detection System"
    )
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config YAML file")
    parser.add_argument("--campaign", type=str, default=None,
                       help="Campaign to analyze (e.g., Armenia, China_1)")
    parser.add_argument("--sample", action="store_true",
                       help="Use/create sample data for testing")
    parser.add_argument("--convert", action="store_true",
                       help="Download and convert Zenodo IO dataset to pipeline format")
    parser.add_argument("--convert-max-posts", type=int, default=None,
                       help="Max posts to convert per file (for quick testing)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load config
    config = load_config(args.config)

    # Override campaign if specified
    if args.campaign:
        config["data"]["campaign"] = args.campaign

    logger = logging.getLogger(__name__)

    # Convert Zenodo dataset if requested
    if args.convert:
        from src.data.convert_zenodo import download_and_convert, CAMPAIGNS
        campaign = config["data"]["campaign"]
        if campaign not in CAMPAIGNS and campaign != "all":
            logger.error(
                f"Campaign '{campaign}' not found in Zenodo dataset. "
                f"Available: {list(CAMPAIGNS.keys())}"
            )
            sys.exit(1)
        campaigns_to_convert = (
            list(CAMPAIGNS.keys()) if campaign == "all" else [campaign]
        )
        for c in campaigns_to_convert:
            logger.info(f"Converting {c} from Zenodo...")
            download_and_convert(
                campaign=c,
                output_dir=config["data"]["raw_dir"],
                max_posts=args.convert_max_posts,
            )
        logger.info("Conversion complete!")

    # Import and run pipeline
    from src.pipeline import CoordinationDetectionPipeline

    logger.info("Starting Coordinated Misinformation Network Detection System")
    logger.info(f"Campaign: {config['data']['campaign']}")
    logger.info(f"Using sample data: {args.sample}")

    pipeline = CoordinationDetectionPipeline(config)
    results = pipeline.run(use_sample=args.sample)

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {config.get('experiments', {}).get('output_dir', 'results')}/")
    logger.info("Run 'streamlit run app/streamlit_app.py' to view the dashboard")


if __name__ == "__main__":
    main()
