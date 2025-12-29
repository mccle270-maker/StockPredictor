# Quick Reference: Experiment Framework Commands

## Instant Execution

### Phase 2b Experiments (10 total, ~5-10 min)
```bash
cd ~/Desktop/Stock\ Predictor
python run_experiments.py --config experiments_phase2b.json
```

### Grid Search XGBoost (12-27 variations, ~15-30 min)
```bash
python grid_search.py --ticker GLD --model xgb --search_type depth
python grid_search.py --ticker QQQ --model xgb --search_type depth
```

### Grid Search Random Forest (12-27 variations, ~10-20 min)
```bash
python grid_search.py --ticker GLD --model rf --search_type leaf
python grid_search.py --ticker SPY --model rf --search_type leaf
```

### Full Framework Demo (4 demo scenarios, ~3-5 min)
```bash
python demo_experiments.py
```

## View Results

```bash
# CSV leaderboard (sortable in Excel)
open results/leaderboard_*.csv

# Detailed JSON (metrics per experiment)
cat results/experiment_results_*.json | python -m json.tool | less

# Grid search analysis (proposals and improvements)
cat grid_search_analysis_*.json | python -m json.tool | less
```

## Python API (For Custom Experiments)

```python
# Single experiment
from experiment_runner import ExperimentRunner, ExperimentConfig, ModelConfig, BacktestConfig

runner = ExperimentRunner(results_dir="my_results")
config = ExperimentConfig(
    experiment_id="my_exp_1",
    ticker="GLD",
    model=ModelConfig(model_type="xgb", max_depth=5, reg_lambda=2.0),
    backtest=BacktestConfig(period="5y"),
)
runner.add_experiment(config)
runner.run_all_experiments()
runner.print_leaderboard(top_k=5)

# Grid search
from grid_search import GridSearchOptimizer, create_xgb_depth_sweep

optimizer = GridSearchOptimizer()
config = create_xgb_depth_sweep(ticker="GLD")  # Preset sweep
optimizer.add_grid_search(config)
results = optimizer.run_grid_searches()
proposals = optimizer.propose_improvements(results)
for p in proposals:
    print(p)
```

## Expected Output Formats

### Leaderboard (CSV)
```
experiment_id,ticker,model_type,accuracy,sharpe_ratio,max_drawdown,profit_factor,rmse,samples
phase2b_elasticnet_rf,GLD,rf,0.620,4.150,-0.185,2.340,0.0185,400
phase2b_xgb_d3,QQQ,xgb,0.590,2.850,-0.220,1.920,0.0212,380
```

### Detailed Results (JSON)
```json
{
  "experiment_id": "phase2b_baseline_rf",
  "ticker": "GLD",
  "model_type": "rf",
  "status": "success",
  "accuracy": 0.615,
  "sharpe_ratio": 3.820,
  "max_drawdown": -0.195,
  "profit_factor": 2.180,
  "win_rate": 0.612,
  "loss_rate": 0.388,
  "rmse": 0.0189,
  "mae": 0.0145,
  "samples": 405,
  "features_used": 143,
  "timestamp": "2025-12-29T15:30:45",
  "notes": "Baseline RF with default hyperparameters"
}
```

### Grid Search Analysis (JSON)
```json
{
  "ticker": "GLD",
  "model_type": "xgb",
  "search_type": "depth",
  "experiments_run": 27,
  "best_experiment": {
    "max_depth": 5,
    "learning_rate": 0.05,
    "sharpe_ratio": 4.250
  },
  "proposals": [
    "Sharpe ratio improves with max_depth=5. Decrease further?",
    "Optimal learning_rate appears to be 0.05. Test 0.03?",
    "No overfitting detected. Can increase regularization..."
  ]
}
```

## Configuration Templates

### Minimal Config (baseline)
```json
{
  "experiments": [
    {
      "experiment_id": "test_1",
      "ticker": "GLD",
      "model": {"model_type": "rf"},
      "backtest": {"period": "5y"}
    }
  ]
}
```

### Advanced Config (with all options)
```json
{
  "experiments": [
    {
      "experiment_id": "advanced_1",
      "ticker": "GLD",
      "description": "XGBoost with regularization",
      "model": {
        "model_type": "xgb",
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "reg_lambda": 5.0,
        "subsample": 0.8
      },
      "backtest": {
        "period": "10y",
        "horizon": 1,
        "train_years": 2,
        "test_years": 1,
        "use_feature_selection": true
      },
      "features": {
        "name": "full_features",
        "include_price": true,
        "include_macro": true,
        "include_sentiment": true
      }
    }
  ]
}
```

## Common Workflows

### Workflow 1: Quick Comparison (3 minutes)
```bash
# Test RF vs XGB on GLD
python run_experiments.py --config experiments_phase2b.json \
  --experiment_ids phase2b_baseline_rf,phase2b_baseline_xgb \
  --top_k 2
```

### Workflow 2: Feature Impact (5 minutes)
```bash
# Run demo feature ablation
python demo_experiments.py --demo_type ablation
# Output shows impact of macro, sentiment features
```

### Workflow 3: Hyperparameter Optimization (30 minutes)
```bash
# 1. Run depth sweep
python grid_search.py --ticker GLD --model xgb --search_type depth

# 2. Review proposals
cat grid_search_analysis_*.json | python -m json.tool

# 3. Create new config with winning params
# Edit experiments_phase2b.json with best values
# Re-run with: python run_experiments.py --config experiments_phase2b.json
```

### Workflow 4: Production Deployment (10 minutes)
```bash
# 1. Find best experiment
python run_experiments.py --config experiments_phase2b.json --top_k 5

# 2. Extract winning params from leaderboard
cat results/leaderboard_*.csv | head -5

# 3. Hardcode into prediction_model.py make_model()
# Edit make_model() function

# 4. Test on paper trading
python auto_paper_trade.py
```

## Metrics Quick Lookup

| Metric | Target | Interpretation |
|--------|--------|-----------------|
| Accuracy | > 55% | Directional prediction hit rate |
| Sharpe | > 2.0 | Risk-adjusted return (annual) |
| Max Drawdown | < -20% | Worst peak-to-trough decline |
| Profit Factor | > 1.5 | Total wins / total losses |
| Win Rate | > 50% | Profitable days % |
| RMSE | < 0.025 | Prediction error (returns) |

**Green Zone** (Good): Sharpe > 2.5, Accuracy > 56%, Max DD < -18%  
**Yellow Zone** (Okay): Sharpe 1.5-2.5, Accuracy 52-56%, Max DD -18% to -25%  
**Red Zone** (Poor): Sharpe < 1.5, Accuracy < 52%, Max DD < -25%

## Debugging

### Check data availability
```python
from prediction_model import build_features_and_target
X, y, _, _, _, _, _ = build_features_and_target("GLD", "5y", 1)
print(f"Samples: {len(X)}, Features: {X.shape[1]}")
# Need at least 60 samples
```

### Verify one experiment manually
```python
from experiment_runner import ExperimentRunner, ExperimentConfig, ModelConfig

runner = ExperimentRunner()
config = ExperimentConfig(
    experiment_id="test",
    ticker="GLD",
    model=ModelConfig(model_type="rf")
)
runner.add_experiment(config)
try:
    runner.run_all_experiments()
except Exception as e:
    print(f"Error: {e}")
```

### Profile execution time
```bash
time python run_experiments.py --config experiments_phase2b.json \
  --max_experiments 1
```

## Environment Setup

```bash
# Ensure virtual environment active
source ~/Desktop/Stock\ Predictor/tf-env/bin/activate

# Verify Python version
python --version
# Expected: Python 3.11.x

# Verify key packages
python -c "import xgboost, sklearn, pandas, yfinance; print('✅ All packages available')"

# Run syntax check
python3 -m py_compile experiment_runner.py run_experiments.py grid_search.py
# Expected: ✅ (no output)
```

## File Locations

```
~/Desktop/Stock Predictor/
├── experiment_runner.py          ← Core framework
├── run_experiments.py            ← JSON runner
├── grid_search.py                ← Hyperparameter search
├── experiments_phase2b.json       ← Pre-built experiments
├── demo_experiments.py            ← Interactive demo
├── EXPERIMENT_FRAMEWORK_README.md ← Full docs
├── AUTOMATION_GUIDE.md            ← This file
├── prediction_model.py            ← Model library (existing)
├── results/                       ← Output directory (auto-created)
│   ├── leaderboard_*.csv
│   ├── experiment_results_*.json
│   └── grid_search_analysis_*.json
└── demo_results/                  ← Demo output (auto-created)
    └── ...
```

## Success Checklist

- ✅ All 6 framework files created
- ✅ `python3 -m py_compile` passes (no syntax errors)
- ✅ `demo_experiments.py` runs (basic sanity check)
- ✅ Phase 2b experiments execute (main test)
- ✅ Results saved to CSV/JSON (output validation)
- ✅ Leaderboard shows > 5 experiments (execution proof)
- ✅ Metrics > 0 and < 10 for Sharpe (reasonable values)
- ✅ No errors in results/ files (data quality)

---

**Ready to execute**: Pick a command above and run it!  
**Questions?** See `EXPERIMENT_FRAMEWORK_README.md` for details
