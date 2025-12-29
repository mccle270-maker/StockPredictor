"""
Phase 2b Improved: Confidence Filtering + Regularization Tests

Run improved models with:
1. Confidence filtering (skip low-confidence predictions)
2. Regularized hyperparameters (max_depth=7, min_samples_leaf=20)
3. Compare vs Phase 2b baseline

Target: 55%+ accuracy, Sharpe 2.5+, reduced overfitting
"""

import os
import sys
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_runner import ExperimentRunner, ExperimentConfig, ModelConfig, BacktestConfig, FeatureConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def run_improved_tests():
    """Run multiple rounds of improved experiments"""
    
    runner = ExperimentRunner()
    
    all_results = []
    
    # Define rounds
    rounds = [
        {
            "name": "ROUND 1: GLD Regularization Testing",
            "experiments": [
                ExperimentConfig(
                    experiment_id="improved_r1_gld_rf_reg",
                    ticker="GLD",
                    model=ModelConfig(
                        model_type="rf",
                        n_estimators=100,
                        max_depth=7,
                        min_samples_leaf=20,
                        min_samples_split=50,
                    ),
                    features=FeatureConfig(name="baseline"),
                    backtest=BacktestConfig(),
                ),
                ExperimentConfig(
                    experiment_id="improved_r1_gld_xgb_reg",
                    ticker="GLD",
                    model=ModelConfig(
                        model_type="xgb",
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.05,
                        reg_lambda=2.0,
                        reg_alpha=0.5,
                    ),
                    features=FeatureConfig(name="baseline"),
                    backtest=BacktestConfig(),
                ),
                ExperimentConfig(
                    experiment_id="improved_r1_gld_gb_reg",
                    ticker="GLD",
                    model=ModelConfig(
                        model_type="gbrt",
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                    ),
                    features=FeatureConfig(name="baseline"),
                    backtest=BacktestConfig(),
                ),
            ]
        },
        {
            "name": "ROUND 2: Cross-Ticker Validation (Improved)",
            "experiments": [
                ExperimentConfig(
                    experiment_id="improved_r2_spy_rf_reg",
                    ticker="SPY",
                    model=ModelConfig(
                        model_type="rf",
                        n_estimators=100,
                        max_depth=7,
                        min_samples_leaf=20,
                        min_samples_split=50,
                    ),
                    features=FeatureConfig(name="baseline"),
                    backtest=BacktestConfig(),
                ),
                ExperimentConfig(
                    experiment_id="improved_r2_spy_xgb_reg",
                    ticker="SPY",
                    model=ModelConfig(
                        model_type="xgb",
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.05,
                        reg_lambda=2.0,
                    ),
                    features=FeatureConfig(name="baseline"),
                    backtest=BacktestConfig(),
                ),
                ExperimentConfig(
                    experiment_id="improved_r2_qqq_rf_reg",
                    ticker="QQQ",
                    model=ModelConfig(
                        model_type="rf",
                        n_estimators=100,
                        max_depth=7,
                        min_samples_leaf=20,
                    ),
                    features=FeatureConfig(name="baseline"),
                    backtest=BacktestConfig(),
                ),
            ]
        },
        {
            "name": "ROUND 3: Confirmation Test (Same configs)",
            "experiments": [
                ExperimentConfig(
                    experiment_id="improved_r3_gld_rf_reg_confirm",
                    ticker="GLD",
                    model=ModelConfig(
                        model_type="rf",
                        n_estimators=100,
                        max_depth=7,
                        min_samples_leaf=20,
                        min_samples_split=50,
                    ),
                    features=FeatureConfig(name="baseline"),
                    backtest=BacktestConfig(),
                ),
                ExperimentConfig(
                    experiment_id="improved_r3_spy_rf_reg_confirm",
                    ticker="SPY",
                    model=ModelConfig(
                        model_type="rf",
                        n_estimators=100,
                        max_depth=7,
                        min_samples_leaf=20,
                        min_samples_split=50,
                    ),
                    features=FeatureConfig(name="baseline"),
                    backtest=BacktestConfig(),
                ),
            ]
        }
    ]
    
    # Execute each round
    for round_info in rounds:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# {round_info['name']}")
        logger.info(f"{'#'*80}\n")
        
        round_results = []
        for exp_config in round_info["experiments"]:
            result = runner.run_experiment(exp_config)
            round_results.append(result)
            all_results.append(result)
            
            # Log result
            status_emoji = "✅" if result.status == "success" else "⚠️"
            logger.info(f"\n{status_emoji} {result.experiment_id}")
            if result.status == "success":
                logger.info(f"   Accuracy: {result.accuracy*100:.1f}%")
                logger.info(f"   Sharpe: {result.sharpe_ratio:.3f}" if result.sharpe_ratio else "   Sharpe: N/A")
                logger.info(f"   Max DD: {result.max_drawdown*100:.1f}%" if result.max_drawdown else "")
        
        # Round summary
        logger.info(f"\n{'='*80}")
        logger.info(f"ROUND SUMMARY: {round_info['name']}")
        logger.info(f"{'='*80}")
        
        successful = [r for r in round_results if r.status == "success"]
        if successful:
            avg_acc = sum(r.accuracy for r in successful) / len(successful)
            avg_sharpe = sum(r.sharpe_ratio for r in successful if r.sharpe_ratio) / len([r for r in successful if r.sharpe_ratio])
            
            logger.info(f"Experiments: {len(successful)}/{len(round_results)} successful")
            logger.info(f"Avg Accuracy: {avg_acc*100:.1f}%")
            logger.info(f"Avg Sharpe: {avg_sharpe:.3f}")
            
            # Find best
            best_acc = max(successful, key=lambda r: r.accuracy)
            best_sharpe = max((r for r in successful if r.sharpe_ratio), key=lambda r: r.sharpe_ratio, default=None)
            logger.info(f"\nBest Accuracy: {best_acc.experiment_id} ({best_acc.accuracy*100:.1f}%)")
            if best_sharpe:
                logger.info(f"Best Sharpe: {best_sharpe.experiment_id} ({best_sharpe.sharpe_ratio:.3f})")
    
    # Final summary
    logger.info(f"\n\n{'='*80}")
    logger.info("FINAL RESULTS: All Rounds Combined")
    logger.info(f"{'='*80}")
    
    successful = [r for r in all_results if r.status == "success"]
    logger.info(f"\nTotal: {len(successful)}/{len(all_results)} successful")
    
    if successful:
        # By ticker
        by_ticker = {}
        for r in successful:
            if r.ticker not in by_ticker:
                by_ticker[r.ticker] = []
            by_ticker[r.ticker].append(r)
        
        logger.info(f"\nBy Ticker:")
        for ticker, results in sorted(by_ticker.items()):
            avg_acc = sum(r.accuracy for r in results) / len(results)
            avg_sharpe = sum(r.sharpe_ratio for r in results if r.sharpe_ratio) / len([r for r in results if r.sharpe_ratio])
            logger.info(f"\n  {ticker}:")
            logger.info(f"    Avg Accuracy: {avg_acc*100:.1f}%")
            logger.info(f"    Avg Sharpe: {avg_sharpe:.3f}")
            logger.info(f"    Experiments: {len(results)}")
        
        # Top experiments
        logger.info(f"\nTop 5 by Sharpe:")
        top = sorted(successful, key=lambda r: r.sharpe_ratio or 0, reverse=True)[:5]
        for i, r in enumerate(top, 1):
            logger.info(f"  {i}. {r.experiment_id}: Sharpe={r.sharpe_ratio:.3f}, Acc={r.accuracy*100:.1f}%")
    
    # Save results
    output_file = f"improved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_dict = [r.to_dict() for r in all_results]
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"\n✅ Results saved to {output_file}")
    
    return all_results


if __name__ == "__main__":
    logger.info("\n" + "="*80)
    logger.info("PHASE 2B IMPROVED: Confidence Filtering + Regularization")
    logger.info("="*80)
    run_improved_tests()
