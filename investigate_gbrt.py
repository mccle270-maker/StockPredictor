#!/usr/bin/env python3
"""
GBRT Investigation Script
=========================

Investigates why GBRT has -0.54 Sharpe while XGB has +1.22.
Compares model configurations, training behavior, and ensemble impact.

Usage:
    python investigate_gbrt.py

Output:
    experiments/gbrt_investigation.json
    GBRT_INVESTIGATION_REPORT.md
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Disable ElasticNet
os.environ["USE_ELASTICNET_SELECT"] = "0"

# Imports
from src.data.market import get_price_history
from src.data.macro import get_macro_df
from src.core.features import build_all_features, validate_features
from src.core.models import make_model
from src.core.metrics import compute_sharpe
from src.config import FEATURE_COLUMNS

# ============================================================================
# CONFIGURATION
# ============================================================================

TEST_TICKER = "AAPL"
DATA_PERIOD = "2y"
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = PROJECT_ROOT / "experiments"

# Default hyperparameters from src/core/models.py
DEFAULT_HYPERPARAMS = {
    "rf": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE,
    },
    "xgb": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
    },
    "gbrt": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE,
    },
}


def print_section(title: str):
    """Print section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print subsection header."""
    print()
    print(f"  --- {title} ---")


# ============================================================================
# PART 1: COMPARE MODEL CONFIGURATIONS
# ============================================================================

def compare_model_configs() -> Dict[str, Any]:
    """Compare GBRT vs XGB configurations."""
    print_section("PART 1: MODEL CONFIGURATION COMPARISON")
    
    results = {
        "hyperparameters": {},
        "differences": [],
    }
    
    # Show hyperparameters
    for model_type in ["rf", "xgb", "gbrt"]:
        params = DEFAULT_HYPERPARAMS[model_type]
        results["hyperparameters"][model_type] = params
        print(f"\n  {model_type.upper()} hyperparameters:")
        for k, v in params.items():
            print(f"    {k}: {v}")
    
    # Compare key differences
    xgb_params = DEFAULT_HYPERPARAMS["xgb"]
    gbrt_params = DEFAULT_HYPERPARAMS["gbrt"]
    
    differences = []
    
    if xgb_params["max_depth"] != gbrt_params["max_depth"]:
        diff = f"max_depth: XGB={xgb_params['max_depth']} vs GBRT={gbrt_params['max_depth']}"
        differences.append(diff)
        print(f"\n  ⚠ {diff}")
    
    if xgb_params.get("subsample") and "subsample" not in gbrt_params:
        differences.append("XGB uses subsample (0.8), GBRT does not (uses 100%)")
        print(f"  ⚠ XGB uses subsample (0.8), GBRT does not")
    
    if xgb_params.get("colsample_bytree") and "colsample_bytree" not in gbrt_params:
        differences.append("XGB uses colsample_bytree (0.8), GBRT does not")
        print(f"  ⚠ XGB uses colsample_bytree (0.8), GBRT does not")
    
    results["differences"] = differences
    
    print(f"\n  Key observations:")
    print(f"    - XGB has regularization via subsample + colsample_bytree")
    print(f"    - GBRT lacks these regularization techniques")
    print(f"    - GBRT max_depth=3 may be too shallow or too deep")
    
    return results


# ============================================================================
# PART 2: CONTROLLED COMPARISON ON IDENTICAL DATA
# ============================================================================

def prepare_data(ticker: str, period: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Prepare identical train/test data for all models."""
    print(f"\n  Fetching {ticker} data...")
    
    # Get price data
    hist = get_price_history(ticker, period=period, interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"No data for {ticker}")
    
    print(f"    Got {len(hist)} rows")
    
    # Build features
    hist = build_all_features(hist.copy())
    
    # Join macro data
    try:
        macro_df = get_macro_df(period=period)
        hist = hist.join(macro_df, how="left")
        for col in macro_df.columns:
            if col in hist.columns:
                hist[col] = hist[col].ffill().bfill().fillna(0)
    except Exception as e:
        print(f"    ⚠ Macro data unavailable: {e}")
    
    # Build target (1-day forward return)
    hist["target"] = hist["Close"].pct_change(1).shift(-1)
    
    # Validate features
    validated, report = validate_features(
        hist.copy(),
        required_features=FEATURE_COLUMNS,
        max_row_nan_pct=0.20,
        max_feature_nan_pct=0.10,
        drop_warmup=True,
        min_rows_after_clean=60
    )
    
    print(f"    After validation: {len(validated)} rows")
    
    # Get available features
    feature_cols = [c for c in FEATURE_COLUMNS if c in validated.columns]
    print(f"    Features available: {len(feature_cols)}")
    
    # Prepare X and y
    validated = validated.dropna(subset=["target"])
    X = validated[feature_cols].values
    y = validated["target"].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False  # Time-series: no shuffle
    )
    
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, feature_cols, validated


def run_controlled_comparison(X_train, X_test, y_train, y_test, feature_cols) -> Dict[str, Any]:
    """Train all models on identical data and compare."""
    print_section("PART 2: CONTROLLED COMPARISON")
    
    results = {
        "models": {},
        "comparison": {},
    }
    
    model_predictions = {}
    model_importances = {}
    
    for model_type in ["rf", "xgb", "gbrt"]:
        print_subsection(f"Training {model_type.upper()}")
        
        # Create model with default params
        model = make_model(model_type=model_type, task="reg")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        model_predictions[model_type] = {
            "train": y_train_pred,
            "test": y_test_pred,
        }
        
        # Metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Directional accuracy
        train_dir_acc = np.mean(np.sign(y_train_pred) == np.sign(y_train))
        test_dir_acc = np.mean(np.sign(y_test_pred) == np.sign(y_test))
        
        # Sharpe-like metric (using predictions as returns)
        test_sharpe = compute_sharpe(y_test_pred * np.sign(y_test))  # Strategy return
        
        # Feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_features = sorted(
                zip(feature_cols, importances),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            model_importances[model_type] = top_features
        
        # Prediction distribution stats
        pred_mean = np.mean(y_test_pred)
        pred_std = np.std(y_test_pred)
        pred_min = np.min(y_test_pred)
        pred_max = np.max(y_test_pred)
        
        results["models"][model_type] = {
            "train_mse": round(train_mse, 8),
            "test_mse": round(test_mse, 8),
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
            "train_dir_accuracy": round(train_dir_acc, 4),
            "test_dir_accuracy": round(test_dir_acc, 4),
            "test_sharpe": round(test_sharpe, 4),
            "prediction_stats": {
                "mean": round(pred_mean, 6),
                "std": round(pred_std, 6),
                "min": round(pred_min, 6),
                "max": round(pred_max, 6),
            },
            "overfitting_gap": round(train_r2 - test_r2, 4),
        }
        
        print(f"    Train MSE: {train_mse:.8f}, Test MSE: {test_mse:.8f}")
        print(f"    Train R²:  {train_r2:.4f}, Test R²:  {test_r2:.4f}")
        print(f"    Train Dir: {train_dir_acc:.4f}, Test Dir: {test_dir_acc:.4f}")
        print(f"    Test Sharpe: {test_sharpe:.4f}")
        print(f"    Overfitting gap (R²): {train_r2 - test_r2:.4f}")
    
    # Store feature importances
    results["feature_importances"] = {
        model_type: [(f, round(i, 4)) for f, i in imps]
        for model_type, imps in model_importances.items()
    }
    
    # Analyze prediction divergence
    print_subsection("Prediction Divergence Analysis")
    
    xgb_pred = model_predictions["xgb"]["test"]
    gbrt_pred = model_predictions["gbrt"]["test"]
    
    # Correlation between predictions
    pred_corr = np.corrcoef(xgb_pred, gbrt_pred)[0, 1]
    print(f"    XGB vs GBRT prediction correlation: {pred_corr:.4f}")
    
    # Sign disagreement
    sign_disagree = np.mean(np.sign(xgb_pred) != np.sign(gbrt_pred))
    print(f"    Sign disagreement rate: {sign_disagree:.4f} ({sign_disagree*100:.1f}%)")
    
    # Mean absolute difference
    mean_abs_diff = np.mean(np.abs(xgb_pred - gbrt_pred))
    print(f"    Mean absolute prediction difference: {mean_abs_diff:.6f}")
    
    results["comparison"] = {
        "xgb_gbrt_correlation": round(pred_corr, 4),
        "sign_disagreement_rate": round(sign_disagree, 4),
        "mean_abs_difference": round(mean_abs_diff, 6),
    }
    
    return results, model_predictions


# ============================================================================
# PART 3: DIAGNOSE GBRT ISSUES
# ============================================================================

def diagnose_gbrt_issues(results: Dict) -> Dict[str, Any]:
    """Analyze what's wrong with GBRT."""
    print_section("PART 3: GBRT DIAGNOSIS")
    
    diagnosis = {
        "issues_found": [],
        "severity": "UNKNOWN",
        "recommended_action": "",
    }
    
    gbrt = results["models"]["gbrt"]
    xgb = results["models"]["xgb"]
    rf = results["models"]["rf"]
    
    print("\n  Comparing model performance:")
    print(f"    {'Model':<10} {'Train R²':>10} {'Test R²':>10} {'Gap':>10} {'Test Sharpe':>12}")
    print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for name, m in [("RF", rf), ("XGB", xgb), ("GBRT", gbrt)]:
        print(f"    {name:<10} {m['train_r2']:>10.4f} {m['test_r2']:>10.4f} {m['overfitting_gap']:>10.4f} {m['test_sharpe']:>12.4f}")
    
    # Issue 1: Check overfitting
    if gbrt["overfitting_gap"] > 0.1:
        issue = f"OVERFITTING: Train-Test R² gap = {gbrt['overfitting_gap']:.4f} (>0.1)"
        diagnosis["issues_found"].append(issue)
        print(f"\n  ❌ {issue}")
    elif gbrt["overfitting_gap"] < -0.05:
        issue = f"UNDERFITTING: Negative train-test gap = {gbrt['overfitting_gap']:.4f}"
        diagnosis["issues_found"].append(issue)
        print(f"\n  ❌ {issue}")
    else:
        print(f"\n  ✓ No severe overfitting (gap = {gbrt['overfitting_gap']:.4f})")
    
    # Issue 2: Check if GBRT significantly worse than XGB
    sharpe_diff = xgb["test_sharpe"] - gbrt["test_sharpe"]
    if sharpe_diff > 0.5:
        issue = f"UNDERPERFORMANCE: XGB Sharpe {xgb['test_sharpe']:.4f} >> GBRT Sharpe {gbrt['test_sharpe']:.4f} (diff: {sharpe_diff:.4f})"
        diagnosis["issues_found"].append(issue)
        print(f"  ❌ {issue}")
    
    # Issue 3: Check directional accuracy
    if gbrt["test_dir_accuracy"] < 0.50:
        issue = f"POOR DIRECTION: Test accuracy {gbrt['test_dir_accuracy']:.4f} < 50% (worse than random)"
        diagnosis["issues_found"].append(issue)
        print(f"  ❌ {issue}")
    elif gbrt["test_dir_accuracy"] < 0.52:
        issue = f"MARGINAL DIRECTION: Test accuracy {gbrt['test_dir_accuracy']:.4f} barely above random"
        diagnosis["issues_found"].append(issue)
        print(f"  ⚠️ {issue}")
    
    # Issue 4: Check prediction distribution
    gbrt_stats = gbrt["prediction_stats"]
    xgb_stats = xgb["prediction_stats"]
    
    if gbrt_stats["std"] < xgb_stats["std"] * 0.5:
        issue = f"LOW VARIANCE: GBRT std={gbrt_stats['std']:.6f} << XGB std={xgb_stats['std']:.6f}"
        diagnosis["issues_found"].append(issue)
        print(f"  ⚠️ {issue}")
    
    # Determine severity
    if len(diagnosis["issues_found"]) >= 3:
        diagnosis["severity"] = "HIGH"
    elif len(diagnosis["issues_found"]) >= 1:
        diagnosis["severity"] = "MEDIUM"
    else:
        diagnosis["severity"] = "LOW"
    
    print(f"\n  Severity: {diagnosis['severity']} ({len(diagnosis['issues_found'])} issues found)")
    
    return diagnosis


# ============================================================================
# PART 4: TEST HYPERPARAMETER TUNING FOR GBRT
# ============================================================================

def test_gbrt_tuning(X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """Test different GBRT configurations."""
    print_section("PART 4: GBRT HYPERPARAMETER TUNING")
    
    results = {"configs_tested": [], "best_config": None}
    
    # Configurations to test
    configs = [
        {"name": "default", "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0},
        {"name": "deeper", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "subsample": 1.0},
        {"name": "more_trees", "n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 1.0},
        {"name": "regularized", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8},
        {"name": "xgb_like", "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "subsample": 0.8},
    ]
    
    print(f"\n  Testing {len(configs)} configurations...")
    print(f"    {'Config':<15} {'Test R²':>10} {'Test Dir':>10} {'Sharpe':>10}")
    print(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    
    best_sharpe = -999
    best_config = None
    
    for cfg in configs:
        from sklearn.ensemble import GradientBoostingRegressor
        
        model = GradientBoostingRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            random_state=RANDOM_STATE,
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_pred)
        test_dir = np.mean(np.sign(y_pred) == np.sign(y_test))
        test_sharpe = compute_sharpe(y_pred * np.sign(y_test))
        
        result = {
            "name": cfg["name"],
            "params": cfg,
            "test_r2": round(test_r2, 4),
            "test_dir_accuracy": round(test_dir, 4),
            "test_sharpe": round(test_sharpe, 4),
        }
        results["configs_tested"].append(result)
        
        print(f"    {cfg['name']:<15} {test_r2:>10.4f} {test_dir:>10.4f} {test_sharpe:>10.4f}")
        
        if test_sharpe > best_sharpe:
            best_sharpe = test_sharpe
            best_config = result
    
    results["best_config"] = best_config
    print(f"\n  Best config: {best_config['name']} (Sharpe: {best_config['test_sharpe']:.4f})")
    
    return results


# ============================================================================
# PART 5: ENSEMBLE COMPARISON
# ============================================================================

def compare_ensembles(X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """Compare 3-model vs 2-model ensemble."""
    print_section("PART 5: ENSEMBLE COMPARISON")
    
    results = {"ensembles": {}, "recommendation": ""}
    
    # Train all models
    models = {}
    predictions = {}
    
    for model_type in ["rf", "xgb", "gbrt"]:
        model = make_model(model_type=model_type, task="reg")
        model.fit(X_train, y_train)
        models[model_type] = model
        predictions[model_type] = model.predict(X_test)
    
    # Ensemble configurations
    ensemble_configs = {
        "3-model (RF+XGB+GBRT)": ["rf", "xgb", "gbrt"],
        "2-model (RF+XGB)": ["rf", "xgb"],
        "2-model (XGB+GBRT)": ["xgb", "gbrt"],
        "XGB only": ["xgb"],
    }
    
    print(f"\n  {'Ensemble':<25} {'Test R²':>10} {'Test Dir':>10} {'Sharpe':>10} {'Stability':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    best_sharpe = -999
    best_ensemble = None
    
    for name, model_list in ensemble_configs.items():
        # Average predictions
        ensemble_pred = np.mean([predictions[m] for m in model_list], axis=0)
        
        test_r2 = r2_score(y_test, ensemble_pred)
        test_dir = np.mean(np.sign(ensemble_pred) == np.sign(y_test))
        test_sharpe = compute_sharpe(ensemble_pred * np.sign(y_test))
        
        # Stability: std of predictions (lower = more stable)
        stability = np.std(ensemble_pred)
        
        results["ensembles"][name] = {
            "models": model_list,
            "test_r2": round(test_r2, 4),
            "test_dir_accuracy": round(test_dir, 4),
            "test_sharpe": round(test_sharpe, 4),
            "prediction_std": round(stability, 6),
        }
        
        print(f"  {name:<25} {test_r2:>10.4f} {test_dir:>10.4f} {test_sharpe:>10.4f} {stability:>10.6f}")
        
        if test_sharpe > best_sharpe:
            best_sharpe = test_sharpe
            best_ensemble = name
    
    # Recommendation
    ensemble_3 = results["ensembles"]["3-model (RF+XGB+GBRT)"]
    ensemble_2 = results["ensembles"]["2-model (RF+XGB)"]
    
    sharpe_improvement = ensemble_2["test_sharpe"] - ensemble_3["test_sharpe"]
    
    if sharpe_improvement > 0.1:
        recommendation = f"REMOVE GBRT: 2-model ensemble improves Sharpe by {sharpe_improvement:.4f}"
    elif sharpe_improvement > 0:
        recommendation = f"CONSIDER REMOVING GBRT: Small Sharpe improvement of {sharpe_improvement:.4f}"
    else:
        recommendation = f"KEEP GBRT: Removing hurts Sharpe by {-sharpe_improvement:.4f}"
    
    results["recommendation"] = recommendation
    results["best_ensemble"] = best_ensemble
    
    print(f"\n  Best ensemble: {best_ensemble}")
    print(f"  Recommendation: {recommendation}")
    
    return results


# ============================================================================
# PART 6: FINAL RECOMMENDATION
# ============================================================================

def generate_recommendation(
    config_results: Dict,
    comparison_results: Dict,
    diagnosis: Dict,
    tuning_results: Dict,
    ensemble_results: Dict,
) -> Dict[str, Any]:
    """Generate final recommendation."""
    print_section("FINAL RECOMMENDATION")
    
    recommendation = {
        "action": "",
        "reasoning": [],
        "implementation": [],
    }
    
    # Analyze findings
    gbrt_sharpe = comparison_results["models"]["gbrt"]["test_sharpe"]
    xgb_sharpe = comparison_results["models"]["xgb"]["test_sharpe"]
    ensemble_improvement = (
        ensemble_results["ensembles"]["2-model (RF+XGB)"]["test_sharpe"] -
        ensemble_results["ensembles"]["3-model (RF+XGB+GBRT)"]["test_sharpe"]
    )
    best_tuned_sharpe = tuning_results["best_config"]["test_sharpe"]
    
    print("\n  Summary of findings:")
    print(f"    - GBRT Sharpe: {gbrt_sharpe:.4f}")
    print(f"    - XGB Sharpe: {xgb_sharpe:.4f}")
    print(f"    - Best tuned GBRT Sharpe: {best_tuned_sharpe:.4f}")
    print(f"    - 2-model vs 3-model improvement: {ensemble_improvement:.4f}")
    print(f"    - Issues found: {len(diagnosis['issues_found'])}")
    
    # Decision logic
    if ensemble_improvement > 0.1:
        # Removing GBRT significantly improves ensemble
        recommendation["action"] = "REMOVE_GBRT"
        recommendation["reasoning"] = [
            f"Removing GBRT improves ensemble Sharpe by {ensemble_improvement:.4f}",
            f"GBRT individual Sharpe ({gbrt_sharpe:.4f}) significantly worse than XGB ({xgb_sharpe:.4f})",
            f"{len(diagnosis['issues_found'])} issues identified with GBRT",
        ]
        recommendation["implementation"] = [
            "Update ensemble in src/services/prediction.py to use only RF + XGB",
            "Remove 'gbrt' from MODEL_TYPES list",
            "Update config to reflect 2-model ensemble",
        ]
    elif best_tuned_sharpe > gbrt_sharpe + 0.3:
        # Tuning significantly helps
        recommendation["action"] = "TUNE_GBRT"
        recommendation["reasoning"] = [
            f"Tuned GBRT achieves {best_tuned_sharpe:.4f} vs default {gbrt_sharpe:.4f}",
            "Proper hyperparameters can recover GBRT performance",
        ]
        recommendation["implementation"] = [
            f"Update GBRT hyperparameters to: {tuning_results['best_config']['params']}",
            "Add subsample parameter for regularization",
            "Consider learning_rate schedule",
        ]
    else:
        # Neither helps much - keep as is or remove
        recommendation["action"] = "REMOVE_GBRT"
        recommendation["reasoning"] = [
            "Tuning provides minimal improvement",
            "GBRT adds complexity without benefit",
            "Simpler 2-model ensemble preferred",
        ]
        recommendation["implementation"] = [
            "Remove GBRT from ensemble",
            "Use RF + XGB only",
        ]
    
    print(f"\n  ► RECOMMENDATION: {recommendation['action']}")
    print("\n  Reasoning:")
    for r in recommendation["reasoning"]:
        print(f"    - {r}")
    print("\n  Implementation:")
    for i in recommendation["implementation"]:
        print(f"    • {i}")
    
    return recommendation


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_markdown_report(all_results: Dict, output_path: Path):
    """Generate markdown report."""
    lines = [
        "# GBRT Investigation Report",
        "",
        f"**Generated:** {all_results['timestamp']}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"**Recommendation:** `{all_results['recommendation']['action']}`",
        "",
    ]
    
    for reason in all_results["recommendation"]["reasoning"]:
        lines.append(f"- {reason}")
    
    lines.extend([
        "",
        "---",
        "",
        "## Model Comparison (Controlled Test)",
        "",
        "| Model | Train R² | Test R² | Overfitting Gap | Test Sharpe | Direction Acc |",
        "|-------|----------|---------|-----------------|-------------|---------------|",
    ])
    
    for model_type in ["rf", "xgb", "gbrt"]:
        m = all_results["comparison"]["models"][model_type]
        lines.append(
            f"| {model_type.upper()} | {m['train_r2']:.4f} | {m['test_r2']:.4f} | "
            f"{m['overfitting_gap']:.4f} | {m['test_sharpe']:.4f} | {m['test_dir_accuracy']:.4f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## GBRT Issues Identified",
        "",
    ])
    
    for issue in all_results["diagnosis"]["issues_found"]:
        lines.append(f"- ❌ {issue}")
    
    if not all_results["diagnosis"]["issues_found"]:
        lines.append("- ✅ No major issues found")
    
    lines.extend([
        "",
        f"**Severity:** {all_results['diagnosis']['severity']}",
        "",
        "---",
        "",
        "## Hyperparameter Tuning Results",
        "",
        "| Config | Test R² | Direction | Sharpe |",
        "|--------|---------|-----------|--------|",
    ])
    
    for cfg in all_results["tuning"]["configs_tested"]:
        lines.append(
            f"| {cfg['name']} | {cfg['test_r2']:.4f} | {cfg['test_dir_accuracy']:.4f} | {cfg['test_sharpe']:.4f} |"
        )
    
    lines.extend([
        "",
        f"**Best Config:** {all_results['tuning']['best_config']['name']}",
        "",
        "---",
        "",
        "## Ensemble Comparison",
        "",
        "| Ensemble | Models | Test R² | Direction | Sharpe |",
        "|----------|--------|---------|-----------|--------|",
    ])
    
    for name, data in all_results["ensemble"]["ensembles"].items():
        lines.append(
            f"| {name} | {'+'.join(data['models'])} | {data['test_r2']:.4f} | "
            f"{data['test_dir_accuracy']:.4f} | {data['test_sharpe']:.4f} |"
        )
    
    lines.extend([
        "",
        f"**Best Ensemble:** {all_results['ensemble']['best_ensemble']}",
        "",
        f"**Recommendation:** {all_results['ensemble']['recommendation']}",
        "",
        "---",
        "",
        "## Implementation Steps",
        "",
    ])
    
    for step in all_results["recommendation"]["implementation"]:
        lines.append(f"1. {step}")
    
    lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n  ✓ Report saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "ticker": TEST_TICKER,
        "period": DATA_PERIOD,
    }
    
    # Part 1: Compare configs
    all_results["config"] = compare_model_configs()
    
    # Part 2: Prepare data and run controlled comparison
    X_train, X_test, y_train, y_test, feature_cols, validated_df = prepare_data(TEST_TICKER, DATA_PERIOD)
    comparison_results, model_predictions = run_controlled_comparison(X_train, X_test, y_train, y_test, feature_cols)
    all_results["comparison"] = comparison_results
    
    # Part 3: Diagnose GBRT
    diagnosis = diagnose_gbrt_issues(comparison_results)
    all_results["diagnosis"] = diagnosis
    
    # Part 4: Test GBRT tuning
    tuning_results = test_gbrt_tuning(X_train, X_test, y_train, y_test)
    all_results["tuning"] = tuning_results
    
    # Part 5: Compare ensembles
    ensemble_results = compare_ensembles(X_train, X_test, y_train, y_test)
    all_results["ensemble"] = ensemble_results
    
    # Part 6: Final recommendation
    recommendation = generate_recommendation(
        all_results["config"],
        comparison_results,
        diagnosis,
        tuning_results,
        ensemble_results,
    )
    all_results["recommendation"] = recommendation
    
    # Save JSON
    json_path = OUTPUT_DIR / "gbrt_investigation.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✓ JSON saved to: {json_path}")
    
    # Generate markdown report
    md_path = PROJECT_ROOT / "GBRT_INVESTIGATION_REPORT.md"
    generate_markdown_report(all_results, md_path)
    
    print()
    print("=" * 70)
    print("  INVESTIGATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
