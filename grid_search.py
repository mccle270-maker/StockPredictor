#!/usr/bin/env python3
"""
Grid Search and Optimization Framework

Systematically tests hyperparameter combinations and proposes improvements
based on out-of-sample performance, not just in-sample Sharpe.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple, Any
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ModelConfig,
    BacktestConfig,
    FeatureConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('GridSearch')


@dataclass
class GridSearchConfig:
    """Configuration for grid search."""
    ticker: str
    model_type: str
    param_grid: Dict[str, List[Any]]
    backtest_config: BacktestConfig
    feature_config: FeatureConfig
    description: str = ""
    
    def generate_experiments(self) -> List[ExperimentConfig]:
        """Generate all experiment combinations."""
        param_names = list(self.param_grid.keys())
        param_values_list = [self.param_grid[pn] for pn in param_names]
        
        experiments = []
        
        for i, param_values in enumerate(itertools.product(*param_values_list)):
            param_dict = dict(zip(param_names, param_values))
            
            # Create model config with params
            model_cfg = ModelConfig(
                model_type=self.model_type,
                **param_dict
            )
            
            # Create experiment ID
            param_str = '_'.join([f"{k}={v}" for k, v in param_dict.items()])
            exp_id = f"{self.ticker}_{self.model_type}_{param_str}"
            
            config = ExperimentConfig(
                experiment_id=exp_id,
                ticker=self.ticker,
                model=model_cfg,
                backtest=self.backtest_config,
                features=self.feature_config,
                description=f"{self.description} | {param_dict}",
            )
            experiments.append(config)
        
        return experiments


class GridSearchOptimizer:
    """Orchestrates grid search and optimization."""
    
    def __init__(self, results_dir: str = "results"):
        self.runner = ExperimentRunner(results_dir=results_dir)
        self.search_configs: List[GridSearchConfig] = []
        self.history: List[Dict[str, Any]] = []
    
    def add_grid_search(self, config: GridSearchConfig) -> None:
        """Add a grid search configuration."""
        self.search_configs.append(config)
        logger.info(f"Added grid search: {config.ticker} / {config.model_type}")
        
        # Log parameter grid
        total_combinations = 1
        for param_name, values in config.param_grid.items():
            total_combinations *= len(values)
            logger.info(f"  {param_name}: {values}")
        logger.info(f"  Total combinations: {total_combinations}")
    
    def run_grid_searches(self) -> pd.DataFrame:
        """Execute all grid searches."""
        for i, config in enumerate(self.search_configs, 1):
            logger.info(f"\n[{i}/{len(self.search_configs)}] Running grid search: {config.ticker}/{config.model_type}")
            
            experiments = config.generate_experiments()
            logger.info(f"Generated {len(experiments)} experiments")
            
            for exp in experiments:
                self.runner.add_experiment(exp)
        
        # Run all experiments
        logger.info(f"\n{'='*100}")
        logger.info(f"RUNNING {len(self.runner.experiment_configs)} EXPERIMENTS")
        logger.info(f"{'='*100}")
        
        self.runner.run_all_experiments()
        
        return pd.DataFrame([r.to_dict() for r in self.runner.results])
    
    def analyze_results(self, df: pd.DataFrame, sort_by: str = "sharpe_ratio") -> Dict[str, Any]:
        """Analyze grid search results and propose improvements."""
        analysis = {}
        
        # Overall statistics
        analysis['total_experiments'] = len(df)
        analysis['successful_experiments'] = len(df[df['status'] == 'success'])
        analysis['success_rate'] = analysis['successful_experiments'] / analysis['total_experiments']
        
        # Metrics summary
        success_df = df[df['status'] == 'success']
        
        for metric in ['sharpe_ratio', 'accuracy', 'max_drawdown', 'profit_factor']:
            if metric in success_df.columns:
                analysis[f'{metric}_mean'] = float(success_df[metric].mean())
                analysis[f'{metric}_std'] = float(success_df[metric].std())
                analysis[f'{metric}_max'] = float(success_df[metric].max())
                analysis[f'{metric}_min'] = float(success_df[metric].min())
        
        # Best experiments
        analysis['best_by_sharpe'] = success_df.nlargest(3, 'sharpe_ratio')[
            ['experiment_id', 'model_type', 'sharpe_ratio', 'accuracy']
        ].to_dict('records')
        
        analysis['best_by_accuracy'] = success_df.nlargest(3, 'accuracy')[
            ['experiment_id', 'model_type', 'accuracy', 'sharpe_ratio']
        ].to_dict('records')
        
        # Worst experiments (for debugging)
        analysis['worst_by_sharpe'] = success_df.nsmallest(3, 'sharpe_ratio')[
            ['experiment_id', 'model_type', 'sharpe_ratio', 'accuracy']
        ].to_dict('records')
        
        return analysis
    
    def propose_improvements(self, df: pd.DataFrame) -> List[str]:
        """
        Propose minimal code changes based on grid search results.
        """
        proposals = []
        
        success_df = df[df['status'] == 'success']
        if len(success_df) == 0:
            return ["No successful experiments to analyze"]
        
        # Analyze accuracy patterns
        avg_accuracy = success_df['accuracy'].mean()
        if avg_accuracy < 0.55:
            proposals.append(
                "Accuracy is below 55%. Consider:\n"
                "  1. Adding more features (macro, sentiment)\n"
                "  2. Increasing training window (train_years: 2 → 3-4)\n"
                "  3. Testing longer lookback periods (5y → 10y)"
            )
        
        # Analyze overfitting
        overfitting_models = df[df['status'] == 'success'].copy()
        if 'rmse_ratio' in overfitting_models.columns:
            high_overfit = overfitting_models[overfitting_models['rmse_ratio'] > 2.0]
            if len(high_overfit) > 0:
                proposals.append(
                    f"Found {len(high_overfit)} models with severe overfitting (RMSE ratio > 2.0).\n"
                    "  Recommendations:\n"
                    "  1. Increase regularization (reg_lambda: 1→5, reg_alpha: 0→0.1)\n"
                    "  2. Reduce max_depth (depth: 6 → 3-4)\n"
                    "  3. Enable feature selection (use_feature_selection: true)"
                )
        
        # Model-specific analysis
        models = success_df['model_type'].unique()
        for model in models:
            model_df = success_df[success_df['model_type'] == model]
            avg_sharpe = model_df['sharpe_ratio'].mean()
            
            if avg_sharpe < 1.0:
                proposals.append(
                    f"{model.upper()} average Sharpe ({avg_sharpe:.3f}) is weak.\n"
                    "  Options:\n"
                    "  1. Swap to better model (RF often more stable than XGB)\n"
                    "  2. Ensemble RF + XGB predictions\n"
                    "  3. Add regime detection (don't trade in low-vol periods)"
                )
        
        # Deflated Sharpe warning
        if len(success_df) > 20:
            proposals.append(
                f"Ran {len(success_df)} experiments. Risk of Sharpe inflation.\n"
                "  Best practice:\n"
                "  1. Use deflated Sharpe ratio (multiply by 1/sqrt(tests))\n"
                "  2. Reserve 20% of experiments for final out-of-sample test\n"
                "  3. Focus on stability metrics (max_drawdown, win_rate) not just Sharpe"
            )
        
        return proposals


def create_xgb_depth_sweep(
    ticker: str,
    backtest_cfg: BacktestConfig = None,
) -> GridSearchConfig:
    """Create XGBoost max_depth grid search."""
    if backtest_cfg is None:
        backtest_cfg = BacktestConfig(period="5y", horizon=1)
    
    return GridSearchConfig(
        ticker=ticker,
        model_type="xgb",
        param_grid={
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "reg_lambda": [0.0, 1.0, 5.0],
        },
        backtest_config=backtest_cfg,
        feature_config=FeatureConfig(name="baseline"),
        description=f"XGBoost hyperparameter tuning for {ticker}",
    )


def create_rf_leaf_sweep(
    ticker: str,
    backtest_cfg: BacktestConfig = None,
) -> GridSearchConfig:
    """Create Random Forest min_samples_leaf grid search."""
    if backtest_cfg is None:
        backtest_cfg = BacktestConfig(period="5y", horizon=1)
    
    return GridSearchConfig(
        ticker=ticker,
        model_type="rf",
        param_grid={
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [2, 5, 10],
        },
        backtest_config=backtest_cfg,
        feature_config=FeatureConfig(name="baseline"),
        description=f"Random Forest tuning for {ticker}",
    )


def main():
    parser = argparse.ArgumentParser(description="Grid search and optimization")
    parser.add_argument('--ticker', type=str, default='GLD', help='Stock ticker')
    parser.add_argument('--model', type=str, default='xgb', 
                       choices=['rf', 'xgb'], help='Model type')
    parser.add_argument('--search_type', type=str, default='depth',
                       choices=['depth', 'leaf', 'custom'], help='Grid search type')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--top_k', type=int, default=15, help='Top K to display')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = GridSearchOptimizer(results_dir=args.results_dir)
    
    # Create grid search config
    if args.model == 'xgb' and args.search_type == 'depth':
        config = create_xgb_depth_sweep(args.ticker)
    elif args.model == 'rf' and args.search_type == 'leaf':
        config = create_rf_leaf_sweep(args.ticker)
    else:
        logger.error(f"Unknown search type: {args.search_type} for {args.model}")
        return
    
    optimizer.add_grid_search(config)
    
    # Run grid search
    logger.info("="*100)
    logger.info("GRID SEARCH: START")
    logger.info("="*100)
    
    results_df = optimizer.run_grid_searches()
    
    # Analyze results
    logger.info("\n" + "="*100)
    logger.info("GRID SEARCH: ANALYSIS")
    logger.info("="*100)
    
    analysis = optimizer.analyze_results(results_df)
    
    # Print analysis
    print("\n" + "="*100)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*100)
    print(f"Total Experiments: {analysis['total_experiments']}")
    print(f"Successful: {analysis['successful_experiments']} ({analysis['success_rate']*100:.1f}%)")
    
    for metric in ['sharpe_ratio', 'accuracy', 'max_drawdown']:
        if f'{metric}_mean' in analysis:
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {analysis[f'{metric}_mean']:.4f}")
            print(f"  Std: {analysis[f'{metric}_std']:.4f}")
            print(f"  Range: [{analysis[f'{metric}_min']:.4f}, {analysis[f'{metric}_max']:.4f}]")
    
    print("\n" + "-"*100)
    print("TOP 3 BY SHARPE RATIO")
    print("-"*100)
    for exp in analysis['best_by_sharpe']:
        print(f"  {exp['experiment_id']}: Sharpe={exp['sharpe_ratio']:.3f}, Acc={exp['accuracy']:.3f}")
    
    # Print proposals
    print("\n" + "="*100)
    print("IMPROVEMENT PROPOSALS")
    print("="*100)
    proposals = optimizer.propose_improvements(results_df)
    for i, proposal in enumerate(proposals, 1):
        print(f"\n[{i}] {proposal}")
    
    # Save results
    optimizer.runner.save_results()
    optimizer.runner.save_leaderboard_csv()
    
    # Save analysis
    analysis_file = Path(args.results_dir) / f"grid_search_analysis_{args.ticker}_{args.model}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Analysis saved to {analysis_file}")
    
    print("\n✅ Grid search complete!")


if __name__ == "__main__":
    main()
