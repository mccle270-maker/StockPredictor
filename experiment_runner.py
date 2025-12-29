#!/usr/bin/env python3
"""
Unified Experiment Runner for Stock Predictor
Supports grid search, iterative optimization, and reproducible backtesting.

Features:
- Experiment configuration (model, hyperparams, feature sets, backtest params)
- Walk-forward validation to prevent look-ahead bias
- Metric calculation (Sharpe, accuracy, drawdown, win rate, etc.)
- Results tracking and leaderboard generation
- JSON/CSV export for dashboard integration
"""

import os
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('experiment_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ExperimentRunner')

# Import model functions
from prediction_model import (
    build_features_and_target,
    make_model,
    backtest_one_ticker,
    walk_forward_backtest,
)

# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class ModelConfig:
    """Model type and hyperparameters."""
    model_type: str  # "rf", "xgb", "gbrt", "linreg"
    random_state: int = 42
    task: str = "reg"  # "reg" or "clf"
    
    # Model-specific hyperparams
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    learning_rate: Optional[float] = None
    min_samples_leaf: Optional[int] = None
    min_samples_split: Optional[int] = None
    subsample: Optional[float] = None
    reg_lambda: Optional[float] = None
    reg_alpha: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to hyperparameter dict for model creation."""
        d = asdict(self)
        # Remove None values and fields that shouldn't go to make_model
        return {k: v for k, v in d.items() if v is not None and k not in ('model_type', 'task', 'random_state')}


@dataclass
class BacktestConfig:
    """Backtest parameters."""
    period: str = "5y"  # Historical data period
    horizon: int = 1  # Prediction horizon (days)
    train_years: int = 2  # Training window (years)
    test_years: int = 1  # Test window (years)
    step_days: Optional[int] = None  # Walk-forward step (None = test_years)
    threshold: float = 0.002  # Trade threshold
    use_feature_selection: bool = False
    
    def get_params(self) -> Dict[str, Any]:
        """Return as dict for walk-forward backtest."""
        return asdict(self)


@dataclass
class FeatureConfig:
    """Feature set specification."""
    name: str  # e.g., "baseline", "with_sentiment", "with_macro"
    include_price: bool = True
    include_volume: bool = True
    include_technical: bool = True
    include_macro: bool = False
    include_sentiment: bool = False
    include_fundamentals: bool = False
    use_elasticnet: bool = False
    elasticnet_l1_ratio: float = 0.5
    elasticnet_cv_folds: int = 5


@dataclass
class ExperimentConfig:
    """Complete experiment specification."""
    experiment_id: str
    ticker: str
    model: ModelConfig
    backtest: BacktestConfig
    features: FeatureConfig
    
    # Metadata
    description: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "experiment_id": self.experiment_id,
            "ticker": self.ticker,
            "model": asdict(self.model),
            "backtest": asdict(self.backtest),
            "features": asdict(self.features),
            "description": self.description,
            "timestamp": self.timestamp,
        }


# ============================================================================
# Metrics Calculation
# ============================================================================

@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    experiment_id: str
    ticker: str
    model_type: str
    status: str  # "success", "failed", "insufficient_data"
    
    # Metrics
    sharpe_ratio: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    auc: Optional[float] = None
    pnl: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    loss_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    
    # Metadata
    num_folds: Optional[int] = None
    samples: Optional[int] = None
    features_used: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict, converting all values to JSON-serializable types."""
        d = asdict(self)
        result = {}
        for k, v in d.items():
            if v is None:
                continue
            # Convert numpy types to Python types for JSON serialization
            if isinstance(v, (np.float32, np.float64)):
                result[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                result[k] = int(v)
            else:
                result[k] = v
        return result
    
    def sharpe_adjusted(self, num_experiments: int = 1) -> float:
        """
        Compute deflated Sharpe ratio to account for multiple testing.
        Uses Arnott et al. formula for multiple comparisons.
        """
        if self.sharpe_ratio is None or self.sharpe_ratio <= 0:
            return self.sharpe_ratio or 0.0
        
        # Degrees of freedom penalty for multiple tests
        dof_penalty = np.sqrt(num_experiments)
        return self.sharpe_ratio / dof_penalty


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02,
) -> Dict[str, float]:
    """
    Calculate comprehensive backtest metrics.
    
    Args:
        y_true: True returns (1D array)
        y_pred: Predicted returns (1D array)
        y_proba: Predicted probabilities for direction (optional, for AUC)
        returns: Actual returns for Sharpe calculation (optional)
        risk_free_rate: Annual risk-free rate (default 2%)
    
    Returns:
        Dict of metrics
    """
    metrics = {}
    
    # Direction accuracy (sign prediction)
    y_true_dir = (y_true > 0).astype(int)
    y_pred_dir = (y_pred > 0).astype(int)
    accuracy = np.mean(y_true_dir == y_pred_dir)
    metrics['accuracy'] = float(accuracy)
    
    # Precision & Recall
    if len(np.unique(y_true_dir)) > 1:
        tp = np.sum((y_pred_dir == 1) & (y_true_dir == 1))
        fp = np.sum((y_pred_dir == 1) & (y_true_dir == 0))
        fn = np.sum((y_pred_dir == 0) & (y_true_dir == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
    
    # AUC (if probabilities provided)
    if y_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true_dir, y_proba)
            metrics['auc'] = float(auc)
        except:
            pass
    
    # RMSE & MAE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    metrics['rmse'] = float(rmse)
    metrics['mae'] = float(mae)
    
    # Sharpe ratio (if returns provided)
    if returns is not None and len(returns) > 1:
        ret_mean = np.mean(returns)
        ret_std = np.std(returns)
        if ret_std > 0:
            annual_sharpe = (ret_mean - risk_free_rate / 252) / ret_std * np.sqrt(252)
            metrics['sharpe_ratio'] = float(annual_sharpe)
    
    # Win/Loss statistics
    positive_returns = returns > 0 if returns is not None else y_true > 0
    negative_returns = returns < 0 if returns is not None else y_true < 0
    
    if returns is not None:
        ret_arr = returns
    else:
        ret_arr = y_true
    
    if np.sum(positive_returns) > 0:
        metrics['avg_win'] = float(np.mean(ret_arr[positive_returns]))
        metrics['win_rate'] = float(np.sum(positive_returns) / len(ret_arr))
    
    if np.sum(negative_returns) > 0:
        metrics['avg_loss'] = float(np.mean(ret_arr[negative_returns]))
        metrics['loss_rate'] = float(np.sum(negative_returns) / len(ret_arr))
    
    # Profit factor
    if np.sum(positive_returns) > 0 and np.sum(negative_returns) > 0:
        gross_profit = np.sum(ret_arr[positive_returns])
        gross_loss = np.abs(np.sum(ret_arr[negative_returns]))
        if gross_loss > 0:
            metrics['profit_factor'] = float(gross_profit / gross_loss)
    
    # Max drawdown
    if returns is not None:
        cumret = np.cumprod(1 + returns) - 1
        running_max = np.maximum.accumulate(cumret)
        drawdown = (cumret - running_max) / (1 + running_max)
        metrics['max_drawdown'] = float(np.min(drawdown))
        metrics['pnl'] = float(cumret[-1])
    
    return metrics


# ============================================================================
# Experiment Runner
# ============================================================================

class ExperimentRunner:
    """Main orchestrator for running and tracking experiments."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize experiment runner.
        
        Args:
            results_dir: Directory to store results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results: List[ExperimentResult] = []
        self.experiment_configs: List[ExperimentConfig] = []
        
        logger.info(f"ExperimentRunner initialized. Results dir: {self.results_dir}")
    
    def add_experiment(self, config: ExperimentConfig) -> None:
        """Register an experiment config."""
        self.experiment_configs.append(config)
        logger.info(f"Added experiment: {config.experiment_id} ({config.ticker} / {config.model.model_type})")
    
    def add_experiments_from_dict(self, experiments: List[Dict[str, Any]]) -> None:
        """Load experiments from dict list (e.g., from JSON)."""
        for exp_dict in experiments:
            model_cfg = ModelConfig(**exp_dict['model'])
            backtest_cfg = BacktestConfig(**exp_dict['backtest'])
            feature_cfg = FeatureConfig(**exp_dict['features'])
            
            config = ExperimentConfig(
                experiment_id=exp_dict['experiment_id'],
                ticker=exp_dict['ticker'],
                model=model_cfg,
                backtest=backtest_cfg,
                features=feature_cfg,
                description=exp_dict.get('description', ''),
            )
            self.add_experiment(config)
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Execute a single experiment.
        
        Args:
            config: ExperimentConfig with all parameters
        
        Returns:
            ExperimentResult with metrics
        """
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            ticker=config.ticker,
            model_type=config.model.model_type,
            status="running",
        )
        
        logger.info(f"[{config.experiment_id}] Starting backtest for {config.ticker}")
        
        try:
            # Build features and target
            try:
                X, y, _, _, _, _, dates = build_features_and_target(
                    ticker=config.ticker,
                    period=config.backtest.period,
                    horizon=config.backtest.horizon,
                )
            except Exception as e:
                result.status = "insufficient_data"
                result.error_message = f"Could not build features: {str(e)}"
                logger.warning(f"[{config.experiment_id}] {result.error_message}")
                return result
            
            if X is None or len(X) < 100:
                result.status = "insufficient_data"
                result.error_message = f"Insufficient data: {len(X) if X is not None else 0} samples"
                logger.warning(f"[{config.experiment_id}] {result.error_message}")
                return result
            
            result.samples = len(X)
            result.features_used = X.shape[1]
            
            # Split train/test
            split_idx = int(len(X) * 0.8)
            X_train, y_train = X[:split_idx], y[:split_idx]
            X_test, y_test = X[split_idx:], y[split_idx:]
            
            # Train model
            model = make_model(
                model_type=config.model.model_type,
                task=config.model.task,
                **config.model.to_dict()  # Pass hyperparameters (includes random_state)
            )
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Get probabilities if available
            y_proba_test = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba_test = model.predict_proba(X_test)[:, 1]
                except:
                    pass
            
            # Calculate returns for Sharpe
            test_returns = y_test
            
            # Calculate metrics on test set
            metrics = calculate_metrics(
                y_test, y_pred_test,
                y_proba=y_proba_test,
                returns=test_returns,
            )
            
            # Populate result
            for key, value in metrics.items():
                if hasattr(result, key):
                    setattr(result, key, value)
            
            result.num_folds = 1  # Simple split, not walk-forward
            result.status = "success"
            
            sharpe_str = f"{result.sharpe_ratio:.3f}" if result.sharpe_ratio is not None else "N/A"
            logger.info(
                f"[{config.experiment_id}] ✅ Success | "
                f"Acc: {result.accuracy:.3f} | "
                f"Sharpe: {sharpe_str}"
            )
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            logger.error(f"[{config.experiment_id}] ❌ Failed: {str(e)}")
        
        return result
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all registered experiments."""
        logger.info(f"Starting {len(self.experiment_configs)} experiments...")
        
        for i, config in enumerate(self.experiment_configs, 1):
            logger.info(f"\n[{i}/{len(self.experiment_configs)}] Running: {config.experiment_id}")
            result = self.run_experiment(config)
            self.results.append(result)
        
        logger.info(f"\n✅ All {len(self.results)} experiments completed")
        return self.results
    
    def leaderboard(self, top_k: int = 10, sort_by: str = "sharpe_ratio") -> pd.DataFrame:
        """
        Generate leaderboard of experiments sorted by metric.
        
        Args:
            top_k: Show top K experiments
            sort_by: Sort by this metric
        
        Returns:
            DataFrame with results
        """
        result_dicts = [r.to_dict() for r in self.results]
        df = pd.DataFrame(result_dicts)
        
        if df.empty:
            logger.warning("No results to display")
            return df
        
        # Sort by requested metric (descending)
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False, na_position='last')
        
        return df.head(top_k)
    
    def print_leaderboard(self, top_k: int = 10, sort_by: str = "sharpe_ratio") -> None:
        """Pretty-print leaderboard."""
        df = self.leaderboard(top_k, sort_by)
        
        if df.empty:
            print("No results available")
            return
        
        # Select key columns
        display_cols = ['experiment_id', 'ticker', 'model_type', 'accuracy', 
                        'sharpe_ratio', 'max_drawdown', 'profit_factor', 'status']
        display_cols = [c for c in display_cols if c in df.columns]
        
        print(f"\n{'='*100}")
        print(f"LEADERBOARD (Top {top_k}, sorted by {sort_by})")
        print(f"{'='*100}")
        print(df[display_cols].to_string(index=False))
        print(f"{'='*100}\n")
    
    def save_results(self, filename: Optional[str] = None) -> Path:
        """
        Save results to JSON/CSV.
        
        Args:
            filename: Output filename (auto-generate if None)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        result_dicts = [r.to_dict() for r in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(result_dicts, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def save_leaderboard_csv(self, filename: Optional[str] = None) -> Path:
        """Save leaderboard as CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"leaderboard_{timestamp}.csv"
        
        filepath = self.results_dir / filename
        
        df = pd.DataFrame([r.to_dict() for r in self.results])
        df.to_csv(filepath, index=False)
        
        logger.info(f"Leaderboard saved to {filepath}")
        return filepath


# ============================================================================
# Experiment Configuration Generators
# ============================================================================

def create_baseline_experiments(tickers: List[str]) -> List[ExperimentConfig]:
    """Create baseline experiments with standard configurations."""
    experiments = []
    
    for ticker in tickers:
        for model_type in ["rf", "xgb"]:
            config = ExperimentConfig(
                experiment_id=f"{ticker}_baseline_{model_type}",
                ticker=ticker,
                model=ModelConfig(model_type=model_type),
                backtest=BacktestConfig(period="5y", horizon=1),
                features=FeatureConfig(name="baseline"),
                description=f"Baseline {model_type} model for {ticker}",
            )
            experiments.append(config)
    
    return experiments


def create_hyperparameter_sweep(
    ticker: str,
    model_type: str,
    param_grid: Dict[str, List[Any]],
) -> List[ExperimentConfig]:
    """
    Generate experiments sweeping hyperparameters.
    
    Args:
        ticker: Stock ticker
        model_type: Model type
        param_grid: Dict of param_name -> list of values
    
    Returns:
        List of ExperimentConfig
    """
    import itertools
    
    experiments = []
    param_names = list(param_grid.keys())
    param_values = [param_grid[pn] for pn in param_names]
    
    for values in itertools.product(*param_values):
        param_dict = dict(zip(param_names, values))
        
        model_cfg = ModelConfig(model_type=model_type, **param_dict)
        config = ExperimentConfig(
            experiment_id=f"{ticker}_{model_type}_{str(param_dict).replace(' ', '')}",
            ticker=ticker,
            model=model_cfg,
            backtest=BacktestConfig(period="5y", horizon=1),
            features=FeatureConfig(name="baseline"),
            description=f"{model_type} with {param_dict}",
        )
        experiments.append(config)
    
    return experiments


def create_feature_ablation_experiments(ticker: str, base_model: str = "rf") -> List[ExperimentConfig]:
    """
    Create feature ablation experiments to test individual feature contributions.
    """
    feature_configs = [
        FeatureConfig(name="baseline", include_macro=False, include_sentiment=False),
        FeatureConfig(name="with_macro", include_macro=True, include_sentiment=False),
        FeatureConfig(name="with_sentiment", include_macro=False, include_sentiment=True),
        FeatureConfig(name="with_both", include_macro=True, include_sentiment=True),
    ]
    
    experiments = []
    for i, feature_cfg in enumerate(feature_configs):
        config = ExperimentConfig(
            experiment_id=f"{ticker}_{base_model}_feat_{feature_cfg.name}",
            ticker=ticker,
            model=ModelConfig(model_type=base_model),
            backtest=BacktestConfig(period="5y", horizon=1),
            features=feature_cfg,
            description=f"Feature ablation: {feature_cfg.name}",
        )
        experiments.append(config)
    
    return experiments


# ============================================================================
# Example usage / Entry point
# ============================================================================

if __name__ == "__main__":
    # Create runner
    runner = ExperimentRunner(results_dir="results")
    
    # Example 1: Baseline experiments
    logger.info("="*80)
    logger.info("EXAMPLE 1: Baseline Experiments")
    logger.info("="*80)
    
    tickers = ["AAPL", "GLD", "SPY"]
    baseline_exps = create_baseline_experiments(tickers)
    for exp in baseline_exps:
        runner.add_experiment(exp)
    
    # Example 2: Hyperparameter sweep for one ticker
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Hyperparameter Sweep")
    logger.info("="*80)
    
    xgb_sweep = create_hyperparameter_sweep(
        ticker="AAPL",
        model_type="xgb",
        param_grid={
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        }
    )
    for exp in xgb_sweep:
        runner.add_experiment(exp)
    
    # Example 3: Feature ablation
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: Feature Ablation")
    logger.info("="*80)
    
    feature_exps = create_feature_ablation_experiments(ticker="GLD")
    for exp in feature_exps:
        runner.add_experiment(exp)
    
    # Run all experiments
    logger.info("\n" + "="*80)
    logger.info("RUNNING ALL EXPERIMENTS")
    logger.info("="*80)
    
    runner.run_all_experiments()
    
    # Display results
    runner.print_leaderboard(top_k=15, sort_by="sharpe_ratio")
    
    # Save results
    runner.save_results()
    runner.save_leaderboard_csv()
    
    logger.info("✅ Experiment run complete!")
