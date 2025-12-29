# Automated Experiment Framework for Stock Predictor

Unified backtesting and optimization framework for systematically improving the stock prediction model through grid search, hyperparameter tuning, and feature ablation.

## Overview

This framework enables:

- **Experiment Configuration**: Define model, features, and backtest parameters in JSON
- **Grid Search**: Systematically test hyperparameter combinations
- **Leaderboard**: Track and compare experiment results
- **Reproducibility**: Seed RNGs, log configs, version code
- **Proposals**: AI-powered suggestions for code improvements
- **Safety**: Prevents overfitting via deflated Sharpe warnings and out-of-sample metrics

## Quick Start

### 1. Run Pre-Built Experiments (Phase 2b)

```bash
# Run experiments from JSON config
python run_experiments.py --config experiments_phase2b.json

# View top 20 results
python run_experiments.py --config experiments_phase2b.json --top_k 20

# Sort by accuracy instead of Sharpe
python run_experiments.py --config experiments_phase2b.json --sort_by accuracy
```

### 2. Run Grid Search

```bash
# XGBoost hyperparameter sweep on GLD
python grid_search.py --ticker GLD --model xgb --search_type depth

# Random Forest tuning on SPY
python grid_search.py --ticker SPY --model rf --search_type leaf

# Limit to 20 experiments for testing
python grid_search.py --ticker GLD --model xgb --search_type depth --max_experiments 20
```

### 3. Use in Python Code

```python
from experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ModelConfig,
    BacktestConfig,
    FeatureConfig,
)

# Create runner
runner = ExperimentRunner(results_dir="results")

# Configure experiment
model = ModelConfig(model_type="rf", n_estimators=100, max_depth=None)
backtest = BacktestConfig(period="5y", horizon=1)
features = FeatureConfig(name="baseline", include_macro=True)

config = ExperimentConfig(
    experiment_id="my_first_exp",
    ticker="GLD",
    model=model,
    backtest=backtest,
    features=features,
)

# Run it
runner.add_experiment(config)
runner.run_all_experiments()

# View results
runner.print_leaderboard(top_k=10)
runner.save_results()
```

## Key Components

### 1. ExperimentConfig

Specifies all parameters for one backtest:

```python
@dataclass
class ExperimentConfig:
    experiment_id: str          # Unique identifier
    ticker: str                 # Stock symbol
    model: ModelConfig          # Model type + hyperparams
    backtest: BacktestConfig    # Train/test periods
    features: FeatureConfig     # Feature set spec
    description: str            # Human-readable notes
```

### 2. ExperimentRunner

Orchestrates experiments and tracking:

```python
runner = ExperimentRunner(results_dir="results")

# Add experiments
runner.add_experiment(config)
runner.add_experiments_from_dict(json_list)

# Execute
runner.run_all_experiments()

# View results
runner.print_leaderboard(sort_by="sharpe_ratio")
runner.save_results()                     # → JSON
runner.save_leaderboard_csv()             # → CSV
```

### 3. GridSearchOptimizer

Systematic hyperparameter tuning:

```python
optimizer = GridSearchOptimizer(results_dir="results")

# Configure grid search
config = GridSearchConfig(
    ticker="GLD",
    model_type="xgb",
    param_grid={
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "reg_lambda": [0.0, 1.0, 5.0],
    },
    backtest_config=BacktestConfig(),
    feature_config=FeatureConfig(),
)

optimizer.add_grid_search(config)
results_df = optimizer.run_grid_searches()
proposals = optimizer.propose_improvements(results_df)
```

### 4. Metrics Calculation

Comprehensive backtest metrics:

```python
metrics = calculate_metrics(
    y_true=returns,
    y_pred=predictions,
    y_proba=probabilities,
    returns=daily_pnl,
    risk_free_rate=0.02,
)

# Returns:
# - accuracy: Direction prediction accuracy
# - sharpe_ratio: Annualized Sharpe (252-day year)
# - max_drawdown: Largest peak-to-trough decline
# - win_rate, loss_rate, profit_factor
# - rmse, mae: Regression error metrics
```

## JSON Configuration Format

Define experiments in `experiments_phase2b.json`:

```json
{
  "experiments": [
    {
      "experiment_id": "gld_rf_baseline",
      "ticker": "GLD",
      "description": "Baseline RF on GLD",
      "model": {
        "model_type": "rf",
        "n_estimators": 100,
        "max_depth": null,
        "min_samples_leaf": 2
      },
      "backtest": {
        "period": "5y",
        "horizon": 1,
        "train_years": 2,
        "test_years": 1,
        "threshold": 0.002,
        "use_feature_selection": false
      },
      "features": {
        "name": "baseline",
        "include_macro": false,
        "include_sentiment": false,
        "use_elasticnet": false
      }
    }
  ]
}
```

## Outputs

After running experiments, check `results/`:

```
results/
├── experiment_results_20251229_150000.json     # Full result details
├── leaderboard_20251229_150000.csv             # Sortable leaderboard
└── grid_search_analysis_GLD_xgb.json           # Grid search insights
```

### Results JSON Structure

```json
{
  "experiment_id": "phase2b_baseline_rf",
  "ticker": "GLD",
  "model_type": "rf",
  "status": "success",
  "accuracy": 0.600,
  "sharpe_ratio": 3.925,
  "max_drawdown": -0.145,
  "profit_factor": 2.340,
  "samples": 499,
  "features_used": 143,
  "timestamp": "2025-12-29T15:10:00"
}
```

## Best Practices

### 1. Avoid Overfitting in Grid Search

- Don't optimize only Sharpe ratio (high variance metric)
- Also consider: accuracy, max_drawdown, win_rate
- Use deflated Sharpe when running many experiments
- Reserve 20% of experiments for final out-of-sample test

```python
# Correct: Multi-objective
best = results.nlargest(3, 'sharpe_ratio')
best = best[best['max_drawdown'] > -0.20]
best = best[best['accuracy'] > 0.55]

# Wrong: Single metric
best = results.nlargest(1, 'sharpe_ratio')
```

### 2. Feature Selection Safety

When enabling `use_elasticnet_select`:
- Ensures feature count ≤ sample_count / 5 (safety margin)
- Applies p-value filtering (alpha=0.05)
- Still uses all features; just weights them via ElasticNet L1

### 3. Reproducibility

Every experiment logs:
- Random seed (default 42)
- Feature count
- Train/test split sizes
- Timestamp

Inspect results CSV to see all parameters:

```bash
cat results/leaderboard_*.csv | less
```

### 4. Incremental Changes

Don't run all 100 hyperparams at once. Use phases:

**Phase 1**: Coarse grid (3-5 values per param)
```python
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
}
# 3 × 2 = 6 experiments
```

**Phase 2**: Fine-tune around best params
```python
param_grid = {
    "max_depth": [4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.07],
}
# 3 × 3 = 9 experiments
```

**Phase 3**: Final verification on holdout set

## Example Workflows

### Scenario 1: Improve XGBoost Overfitting

Problem: XGBoost overfitting (train 95%, test 55%)

```bash
# Step 1: Test regularization
python grid_search.py --ticker QQQ --model xgb --search_type depth

# Step 2: Review results
cat results/leaderboard_*.csv

# Step 3: If improved, update make_model() in prediction_model.py
# with winning hyperparams
```

### Scenario 2: Compare RF vs XGB

```python
from experiment_runner import create_baseline_experiments

runner = ExperimentRunner()

# Create baselines for both models
for exp in create_baseline_experiments(['GLD', 'QQQ', 'SPY']):
    runner.add_experiment(exp)

runner.run_all_experiments()
runner.print_leaderboard(sort_by='sharpe_ratio')

# Check which model dominates → use that as default
```

### Scenario 3: Feature Ablation

Which features help most?

```python
from experiment_runner import create_feature_ablation_experiments

runner = ExperimentRunner()

for exp in create_feature_ablation_experiments(ticker='GLD', base_model='rf'):
    runner.add_experiment(exp)

runner.run_all_experiments()

# Compare: baseline vs with_macro vs with_sentiment vs with_both
# If with_sentiment hurts → remove from production
```

## Troubleshooting

### Experiments running slowly

- Check CPU/RAM usage: `top` or `Activity Monitor`
- Limit experiments: `--max_experiments 5` for testing
- Reduce period: `"period": "2y"` instead of `"5y"`

### Low accuracy (< 50%)

- Add more features: `include_macro: true`, `include_sentiment: true`
- Increase training data: `train_years: 2 → 4`
- Try longer lookback: `period: "5y" → "10y"`

### High Sharpe but low accuracy

- Metric inconsistency. Check if model "bets small on up days, big on down days"
- Use win_rate + profit_factor to verify real trading edge
- Don't trust Sharpe alone—it's easily gamed

### NaN results

- Check logs: `cat experiment_runner.log`
- Verify data availability: `build_features_and_target('GLD', '5y')`
- May need longer period or different ticker

## Advanced: Custom Optimization Loop

Implement manual optimization:

```python
from experiment_runner import ExperimentRunner, ModelConfig, ExperimentConfig

runner = ExperimentRunner()

# Iteration 1: Coarse search
for depth in [3, 5, 7]:
    exp = ExperimentConfig(
        experiment_id=f"iter1_depth_{depth}",
        ticker="GLD",
        model=ModelConfig(model_type="xgb", max_depth=depth),
        backtest=BacktestConfig(),
        features=FeatureConfig(),
    )
    runner.add_experiment(exp)

runner.run_all_experiments()

# Analyze
best = runner.leaderboard(top_k=1)
best_depth = best.iloc[0]['model_max_depth']

# Iteration 2: Fine search around best
runner.results = []  # Reset
for depth in [best_depth - 1, best_depth, best_depth + 1]:
    # ... repeat
```

## Performance Expectations

On typical hardware:

- **1 experiment**: 10-60 seconds (depends on period, features)
- **10 experiments**: 2-10 minutes
- **100 experiments**: 30 min to 2 hours

Use `--max_experiments` to test quickly:

```bash
python grid_search.py --ticker GLD --max_experiments 5
```

## References

- **Sharpe Ratio**: `(return - rf_rate) / volatility * sqrt(252)`
- **Deflated Sharpe**: `sharpe / sqrt(num_tests)` (Arnott et al.)
- **Max Drawdown**: Largest peak-to-trough decline in cumulative returns
- **Profit Factor**: Gross profit / Gross loss (> 1.5 is healthy)

## Next Steps

1. Run `experiments_phase2b.json` to validate Phase 1 features
2. Use `grid_search.py` to find optimal hyperparams for each model
3. Commit winning configs to `prediction_model.py`
4. Deploy best ticker (GLD) to paper trading
5. Monitor live performance vs backtest
6. Iterate Phase 2b → Phase 2c → Phase 3

---

**Questions?** Check logs: `experiment_runner.log`

**Need help?** Review Phase 2 summary: `PHASE_2_SUMMARY.txt`
