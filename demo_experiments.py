#!/usr/bin/env python3
"""
Quick Start Demo for Experiment Framework

This script demonstrates all major features:
1. Basic experiment runner
2. Grid search
3. Feature ablation
4. Results analysis

Run with: python demo_experiments.py
"""

import logging
from pathlib import Path

from experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ModelConfig,
    BacktestConfig,
    FeatureConfig,
    create_baseline_experiments,
    create_feature_ablation_experiments,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('Demo')


def demo_basic_experiments():
    """Demo 1: Run a few basic experiments."""
    logger.info("\n" + "="*100)
    logger.info("DEMO 1: Basic Experiments")
    logger.info("="*100)
    
    runner = ExperimentRunner(results_dir="demo_results")
    
    # Test on two tickers with two models each
    for ticker in ["GLD", "SPY"]:
        exps = create_baseline_experiments([ticker])
        for exp in exps:
            runner.add_experiment(exp)
    
    logger.info(f"Running {len(runner.experiment_configs)} baseline experiments...")
    runner.run_all_experiments()
    
    # Display results
    runner.print_leaderboard(top_k=10, sort_by="sharpe_ratio")
    
    return runner


def demo_feature_ablation():
    """Demo 2: Feature ablation to understand contribution."""
    logger.info("\n" + "="*100)
    logger.info("DEMO 2: Feature Ablation (What features help?)")
    logger.info("="*100)
    
    runner = ExperimentRunner(results_dir="demo_results")
    
    # Test different feature combinations on GLD
    exps = create_feature_ablation_experiments(ticker="GLD", base_model="rf")
    for exp in exps:
        runner.add_experiment(exp)
    
    logger.info(f"Running {len(runner.experiment_configs)} feature ablation experiments...")
    runner.run_all_experiments()
    
    # Display results
    runner.print_leaderboard(top_k=10, sort_by="sharpe_ratio")
    
    # Compare metrics
    df = runner.leaderboard(top_k=10)
    logger.info("\nFeature Impact Analysis:")
    logger.info(df[['experiment_id', 'accuracy', 'sharpe_ratio', 'max_drawdown']].to_string(index=False))
    
    return runner


def demo_hyperparameter_sweep():
    """Demo 3: Simple hyperparameter search."""
    logger.info("\n" + "="*100)
    logger.info("DEMO 3: Hyperparameter Sweep")
    logger.info("="*100)
    
    runner = ExperimentRunner(results_dir="demo_results")
    
    # Test different RF max_depth values
    for max_depth in [None, 10, 20, 30]:
        config = ExperimentConfig(
            experiment_id=f"gld_rf_depth_{max_depth}",
            ticker="GLD",
            model=ModelConfig(model_type="rf", max_depth=max_depth, n_estimators=100),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description=f"RF with max_depth={max_depth}",
        )
        runner.add_experiment(config)
    
    logger.info(f"Running {len(runner.experiment_configs)} hyperparameter sweep experiments...")
    runner.run_all_experiments()
    
    # Display results
    runner.print_leaderboard(top_k=10, sort_by="sharpe_ratio")
    
    return runner


def demo_results_analysis():
    """Demo 4: Analyze and compare results."""
    logger.info("\n" + "="*100)
    logger.info("DEMO 4: Results Analysis")
    logger.info("="*100)
    
    runner = ExperimentRunner(results_dir="demo_results")
    
    # Run a few experiments
    exps = create_baseline_experiments(["GLD", "QQQ"])
    for exp in exps:
        runner.add_experiment(exp)
    
    runner.run_all_experiments()
    
    # Get results as dataframe
    df = runner.leaderboard(top_k=20)
    
    # Print statistics
    print("\n" + "-"*100)
    print("STATISTICS BY MODEL TYPE")
    print("-"*100)
    
    for model_type in df['model_type'].unique():
        model_df = df[df['model_type'] == model_type]
        print(f"\n{model_type.upper()}:")
        print(f"  Count: {len(model_df)}")
        print(f"  Avg Accuracy: {model_df['accuracy'].mean():.3f}")
        print(f"  Avg Sharpe: {model_df['sharpe_ratio'].mean():.3f}")
        print(f"  Avg Max DD: {model_df['max_drawdown'].mean():.3f}")
    
    # Statistics by ticker
    print("\n" + "-"*100)
    print("STATISTICS BY TICKER")
    print("-"*100)
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        print(f"\n{ticker}:")
        print(f"  Count: {len(ticker_df)}")
        print(f"  Best Sharpe: {ticker_df['sharpe_ratio'].max():.3f}")
        print(f"  Best Accuracy: {ticker_df['accuracy'].max():.3f}")
        print(f"  Worst Max DD: {ticker_df['max_drawdown'].min():.3f}")


def main():
    """Run all demos."""
    logger.info("="*100)
    logger.info("EXPERIMENT FRAMEWORK DEMONSTRATION")
    logger.info("="*100)
    
    # Create results directory
    Path("demo_results").mkdir(exist_ok=True)
    
    try:
        # Run demos
        runner1 = demo_basic_experiments()
        runner2 = demo_feature_ablation()
        runner3 = demo_hyperparameter_sweep()
        demo_results_analysis()
        
        # Save all results
        logger.info("\n" + "="*100)
        logger.info("SAVING RESULTS")
        logger.info("="*100)
        
        runner1.save_results("demo_basic_results.json")
        runner1.save_leaderboard_csv("demo_basic_leaderboard.csv")
        
        runner2.save_results("demo_ablation_results.json")
        runner2.save_leaderboard_csv("demo_ablation_leaderboard.csv")
        
        runner3.save_results("demo_hyperparameter_results.json")
        runner3.save_leaderboard_csv("demo_hyperparameter_leaderboard.csv")
        
        logger.info("\n✅ Demo complete! Check demo_results/ directory for outputs.")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python run_experiments.py --config experiments_phase2b.json")
        logger.info("  2. Run: python grid_search.py --ticker GLD --model xgb")
        logger.info("  3. Review: results/*.csv")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
