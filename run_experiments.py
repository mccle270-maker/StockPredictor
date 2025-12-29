#!/usr/bin/env python3
"""
Main entry point for running experiments from JSON configuration.

Usage:
    python run_experiments.py --config experiments_phase2b.json
    python run_experiments.py --config experiments_phase2b.json --top_k 20
    python run_experiments.py --config experiments_phase2b.json --sort_by accuracy
"""

import argparse
import json
import logging
from pathlib import Path
import sys

from experiment_runner import ExperimentRunner, ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('RunExperiments')


def load_config_file(config_path: str) -> dict:
    """Load experiment config from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run stock predictor experiments from JSON config"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiments JSON config file'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Show top K results in leaderboard'
    )
    parser.add_argument(
        '--sort_by',
        type=str,
        default='sharpe_ratio',
        help='Sort results by this metric'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--max_experiments',
        type=int,
        default=None,
        help='Limit number of experiments to run (useful for testing)'
    )
    
    args = parser.parse_args()
    
    # Load config
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    config_data = load_config_file(args.config)
    experiments_list = config_data.get('experiments', [])
    
    if not experiments_list:
        logger.error("No experiments found in config")
        sys.exit(1)
    
    logger.info(f"Loaded {len(experiments_list)} experiments from {args.config}")
    
    # Limit experiments if requested
    if args.max_experiments:
        experiments_list = experiments_list[:args.max_experiments]
        logger.info(f"Limiting to {args.max_experiments} experiments")
    
    # Create runner
    runner = ExperimentRunner(results_dir=args.results_dir)
    
    # Add experiments from config
    logger.info(f"Loading {len(experiments_list)} experiments...")
    runner.add_experiments_from_dict(experiments_list)
    
    # Run experiments
    logger.info("\n" + "="*100)
    logger.info("STARTING EXPERIMENT RUN")
    logger.info("="*100)
    runner.run_all_experiments()
    
    # Display results
    logger.info("\n" + "="*100)
    logger.info("RESULTS")
    logger.info("="*100)
    runner.print_leaderboard(top_k=args.top_k, sort_by=args.sort_by)
    
    # Save results
    results_file = runner.save_results()
    leaderboard_file = runner.save_leaderboard_csv()
    
    logger.info(f"\n✅ Results saved:")
    logger.info(f"   - JSON: {results_file}")
    logger.info(f"   - CSV: {leaderboard_file}")
    logger.info(f"   - Log: experiment_runner.log")


if __name__ == "__main__":
    main()
