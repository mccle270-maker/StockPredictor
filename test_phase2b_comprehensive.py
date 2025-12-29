#!/usr/bin/env python3
"""
Phase 2b Testing & Analysis Script
Run multiple tests, analyze results, and make adjustments
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from experiment_runner import ExperimentRunner, ExperimentConfig, ModelConfig, BacktestConfig, FeatureConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def run_test_round(round_num, configs):
    """Run a round of experiments and return results."""
    logger.info(f"\n{'='*80}")
    logger.info(f"ROUND {round_num}: Running {len(configs)} experiments")
    logger.info(f"{'='*80}\n")
    
    runner = ExperimentRunner(results_dir="results")
    
    for config in configs:
        runner.add_experiment(config)
    
    runner.run_all_experiments()
    runner.print_leaderboard(top_k=10)
    runner.save_results()
    
    # Get results
    results_files = sorted(Path("results").glob("experiment_results_*.json"))
    if results_files:
        with open(results_files[-1]) as f:
            results = json.load(f)
        return pd.DataFrame(results)
    return None

def create_round1_configs():
    """Round 1: Test basic models and feature combinations on GLD."""
    return [
        ExperimentConfig(
            experiment_id="r1_gld_rf_baseline",
            ticker="GLD",
            model=ModelConfig(model_type="rf", n_estimators=100),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description="GLD baseline RF"
        ),
        ExperimentConfig(
            experiment_id="r1_gld_rf_macro",
            ticker="GLD",
            model=ModelConfig(model_type="rf", n_estimators=100),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="with_macro", include_macro=True),
            description="GLD RF with macro features"
        ),
        ExperimentConfig(
            experiment_id="r1_gld_xgb_baseline",
            ticker="GLD",
            model=ModelConfig(model_type="xgb", n_estimators=100, max_depth=5, learning_rate=0.05),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description="GLD baseline XGB with regularization"
        ),
        ExperimentConfig(
            experiment_id="r1_gld_xgb_macro",
            ticker="GLD",
            model=ModelConfig(model_type="xgb", n_estimators=100, max_depth=5, learning_rate=0.05, reg_lambda=2.0),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="with_macro", include_macro=True),
            description="GLD XGB with macro + regularization"
        ),
    ]

def create_round2_configs():
    """Round 2: Test different tickers with best-performing config."""
    best_config = ModelConfig(model_type="rf", n_estimators=100)
    
    return [
        ExperimentConfig(
            experiment_id="r2_spy_rf",
            ticker="SPY",
            model=best_config,
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description="SPY baseline RF"
        ),
        ExperimentConfig(
            experiment_id="r2_spy_xgb",
            ticker="SPY",
            model=ModelConfig(model_type="xgb", n_estimators=100, max_depth=5),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description="SPY baseline XGB"
        ),
        ExperimentConfig(
            experiment_id="r2_qqq_rf",
            ticker="QQQ",
            model=best_config,
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description="QQQ baseline RF"
        ),
        ExperimentConfig(
            experiment_id="r2_qqq_xgb",
            ticker="QQQ",
            model=ModelConfig(model_type="xgb", n_estimators=100, max_depth=5),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description="QQQ baseline XGB"
        ),
    ]

def create_round3_configs():
    """Round 3: Test regularization and hyperparameter tuning."""
    return [
        ExperimentConfig(
            experiment_id="r3_gld_xgb_shallow",
            ticker="GLD",
            model=ModelConfig(model_type="xgb", n_estimators=150, max_depth=3, learning_rate=0.05, reg_lambda=3.0),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description="GLD XGB shallow tree with strong regularization"
        ),
        ExperimentConfig(
            experiment_id="r3_gld_xgb_deep",
            ticker="GLD",
            model=ModelConfig(model_type="xgb", n_estimators=50, max_depth=7, learning_rate=0.02, reg_lambda=1.0),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description="GLD XGB deeper tree with less regularization"
        ),
        ExperimentConfig(
            experiment_id="r3_gld_rf_deep",
            ticker="GLD",
            model=ModelConfig(model_type="rf", n_estimators=200, max_depth=20),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description="GLD RF with deeper trees"
        ),
    ]

def analyze_results(df):
    """Analyze results and provide recommendations."""
    if df is None or df.empty:
        logger.warning("No results to analyze")
        return
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS")
    logger.info("="*80)
    
    # Group by ticker
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker]
        logger.info(f"\n{ticker}:")
        logger.info(f"  Count: {len(ticker_df)}")
        logger.info(f"  Avg Sharpe: {ticker_df['sharpe_ratio'].mean():.3f}")
        logger.info(f"  Max Sharpe: {ticker_df['sharpe_ratio'].max():.3f} ({ticker_df.loc[ticker_df['sharpe_ratio'].idxmax(), 'experiment_id']})")
        logger.info(f"  Avg Accuracy: {ticker_df['accuracy'].mean():.1%}")
        logger.info(f"  Avg Max DD: {ticker_df['max_drawdown'].mean():.1%}")
    
    # Model comparison
    logger.info("\nModel Type Comparison:")
    for model in df['model_type'].unique():
        model_df = df[df['model_type'] == model]
        logger.info(f"  {model.upper()}: Avg Sharpe={model_df['sharpe_ratio'].mean():.3f}, Count={len(model_df)}")
    
    # Top 5 overall
    logger.info("\nTop 5 Experiments Overall:")
    top5 = df.nlargest(5, 'sharpe_ratio')[['experiment_id', 'ticker', 'model_type', 'sharpe_ratio', 'accuracy']]
    for idx, row in top5.iterrows():
        logger.info(f"  {row['experiment_id']:30} | {row['ticker']:5} | {row['model_type']:5} | Sharpe={row['sharpe_ratio']:.3f} | Acc={row['accuracy']:.1%}")

def main():
    """Run all test rounds."""
    logger.info("PHASE 2B COMPREHENSIVE TESTING & ANALYSIS")
    logger.info("="*80)
    
    all_results = []
    
    # Round 1: Basic models and features (GLD focus)
    logger.info("\n🔍 ROUND 1: Feature & Model Testing on GLD")
    df_r1 = run_test_round(1, create_round1_configs())
    if df_r1 is not None:
        all_results.append(df_r1)
        analyze_results(df_r1)
    
    # Round 2: Cross-ticker validation
    logger.info("\n🔍 ROUND 2: Cross-Ticker Testing")
    df_r2 = run_test_round(2, create_round2_configs())
    if df_r2 is not None:
        all_results.append(df_r2)
        analyze_results(df_r2)
    
    # Round 3: Hyperparameter tuning based on round 1/2 results
    logger.info("\n🔍 ROUND 3: Hyperparameter Tuning")
    df_r3 = run_test_round(3, create_round3_configs())
    if df_r3 is not None:
        all_results.append(df_r3)
        analyze_results(df_r3)
    
    # Combined analysis
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("FINAL SUMMARY - ALL ROUNDS COMBINED")
        logger.info("="*80)
        df_all = pd.concat(all_results, ignore_index=True)
        analyze_results(df_all)
        
        # Recommendations
        logger.info("\n" + "="*80)
        logger.info("RECOMMENDATIONS FOR PRODUCTION")
        logger.info("="*80)
        
        top_by_ticker = df_all.loc[df_all.groupby('ticker')['sharpe_ratio'].idxmax()]
        for idx, row in top_by_ticker.iterrows():
            logger.info(f"\n{row['ticker']}:")
            logger.info(f"  Best Experiment: {row['experiment_id']}")
            logger.info(f"  Model: {row['model_type'].upper()}")
            logger.info(f"  Sharpe: {row['sharpe_ratio']:.3f}")
            logger.info(f"  Accuracy: {row['accuracy']:.1%}")
            logger.info(f"  Max Drawdown: {row['max_drawdown']:.1%}")

if __name__ == "__main__":
    main()
