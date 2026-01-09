"""
ML model creation and training.
Pure functions - no data fetching, no caching.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any, Dict, List
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ============================================================================
# MODEL FACTORY
# ============================================================================

# Try to import ModelEnsemble from model_improvements
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from model_improvements import ModelEnsemble
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False

def make_model(
    model_type: str = "rf",
    task: str = "reg",
    random_state: int = 42,
    use_optimized: bool = True,
    **kwargs
) -> Any:
    """
    Create a model instance.
    
    Args:
        model_type: One of 'rf', 'xgb', 'gbrt', 'linreg', 'ensemble'
        task: 'reg' for regression, 'clf' for classification
        random_state: Random seed
        use_optimized: If True, use OPTIMIZED_MODEL_CONFIG for XGB (default True)
        **kwargs: Override default hyperparameters
    
    Returns:
        Model instance (unfitted)
    """
    from ..config import MODEL_DEFAULTS, OPTIMIZED_MODEL_CONFIG, is_optimized_mode
    import logging
    logger = logging.getLogger("model_factory")
    
    # Handle ensemble model type
    if model_type == "ensemble":
        if HAS_ENSEMBLE:
            print("🔧 Creating ModelEnsemble (RF + XGB)")
            return ModelEnsemble(include_xgb=HAS_XGB)
        else:
            print("⚠️ ModelEnsemble not available, falling back to RF")
            model_type = "rf"
    
    # Determine if we should use optimized config
    use_opt = use_optimized and is_optimized_mode()
    
    # Get defaults for this model type
    if model_type == "xgb" and use_opt:
        # Use optimized hyperparameters from Model Improvement Report
        defaults = {
            "n_estimators": OPTIMIZED_MODEL_CONFIG["n_estimators"],
            "max_depth": OPTIMIZED_MODEL_CONFIG["max_depth"],
            "learning_rate": OPTIMIZED_MODEL_CONFIG["learning_rate"],
            "subsample": OPTIMIZED_MODEL_CONFIG["subsample"],
            "colsample_bytree": OPTIMIZED_MODEL_CONFIG["colsample_bytree"],
            "min_child_weight": OPTIMIZED_MODEL_CONFIG["min_child_weight"],
            "reg_alpha": OPTIMIZED_MODEL_CONFIG["reg_alpha"],
            "reg_lambda": OPTIMIZED_MODEL_CONFIG["reg_lambda"],
            "random_state": random_state,
        }
        logger.info("🚀 Using OPTIMIZED XGBoost config (Sharpe +2.77)")
        print("🚀 Using OPTIMIZED XGBoost config")
    elif model_type == "rf" and use_opt:
        # Use optimized RF hyperparameters from Optuna optimization (2026-01-08)
        from ..config import get_optimized_rf_config
        rf_config = get_optimized_rf_config()
        defaults = {
            "n_estimators": rf_config.get("n_estimators", 100),
            "max_depth": rf_config.get("max_depth", None),
            "min_samples_split": rf_config.get("min_samples_split", 2),
            "min_samples_leaf": rf_config.get("min_samples_leaf", 4),
            "max_features": rf_config.get("max_features", 0.7),
            "bootstrap": rf_config.get("bootstrap", True),
            "random_state": random_state,
            "n_jobs": -1,
        }
        logger.info("🚀 Using OPTIMIZED RF config (Optuna - Sharpe +11.32)")
        print("🚀 Using OPTIMIZED RF config")
    else:
        defaults = MODEL_DEFAULTS.get(model_type, {}).copy()
        defaults["random_state"] = random_state
        if model_type == "xgb":
            logger.info("📊 Using LEGACY XGBoost config")
            print("📊 Using LEGACY XGBoost config")
        elif model_type == "rf":
            logger.info("📊 Using LEGACY RF config")
            print("📊 Using LEGACY RF config")
    
    # Apply any overrides
    defaults.update(kwargs)
    
    if model_type == "rf":
        if task == "clf":
            return RandomForestClassifier(**defaults)
        return RandomForestRegressor(**defaults)
    
    elif model_type == "xgb":
        if not HAS_XGB:
            raise ImportError("XGBoost not installed")
        if task == "clf":
            return XGBClassifier(**defaults, use_label_encoder=False, eval_metric="logloss")
        return XGBRegressor(**defaults)
    
    elif model_type == "gbrt":
        if task == "clf":
            # GradientBoosting doesn't have a separate classifier with same API
            # Use RandomForest as fallback for classification
            return RandomForestClassifier(**MODEL_DEFAULTS.get("rf", {}), random_state=random_state)
        return GradientBoostingRegressor(**defaults)
    
    elif model_type == "linreg":
        return LinearRegression()
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    task: str = "reg",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model with train/test split.
    
    Args:
        X: Feature matrix
        y: Target vector
        model_type: Model type
        task: 'reg' or 'clf'
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        Tuple of (fitted_model, metrics_dict)
    """
    # Chronological split (not random for time series)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = make_model(model_type, task, random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    if task == "reg":
        metrics = {
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "direction_accuracy": np.mean(np.sign(y_pred) == np.sign(y_test)),
        }
    else:
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
        }
    
    return model, metrics


def train_on_full_data(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    task: str = "reg",
    random_state: int = 42,
) -> Any:
    """Train model on full dataset (for final prediction)."""
    model = make_model(model_type, task, random_state)
    model.fit(X, y)
    return model


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str] | None = None,
    dates: Optional[pd.DatetimeIndex] = None,
    horizon: int = 1,
    l1_ratio: float = 0.5,
    n_splits: int = 5,
    cv_folds: int | None = None,  # alias
    min_features: int = 12,
) -> tuple[np.ndarray, List[str], np.ndarray]:
    """
    Select features using Elastic Net with cross-validation.
    
    Args:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        feature_names: List of feature names
        dates: Optional date index (for time-series CV)
        horizon: Prediction horizon (for embargo calculation)
        l1_ratio: L1/L2 ratio (0=Ridge, 1=Lasso)
        n_splits: Number of CV folds
        cv_folds: Alias for n_splits
        min_features: Minimum number of features to keep
    
    Returns:
        (X_selected, selected_names, mask)
    """
    from sklearn.preprocessing import StandardScaler
    
    if cv_folds is not None:
        n_splits = cv_folds
    
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    
    # Handle NaN
    X_clean = np.nan_to_num(X, nan=0.0)
    y_clean = np.nan_to_num(y, nan=0.0)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Fit Elastic Net
    enet = ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=np.logspace(-4, 1, 50),
        cv=n_splits,
        max_iter=10000,
        random_state=42,
    )
    enet.fit(X_scaled, y_clean)
    
    # Select non-zero coefficients
    coef_abs = np.abs(enet.coef_)
    mask = coef_abs > 1e-10
    
    # Ensure minimum features
    if mask.sum() < min_features:
        top_idx = np.argsort(coef_abs)[::-1][:min_features]
        mask = np.zeros_like(mask, dtype=bool)
        mask[top_idx] = True
    
    selected_names = [feature_names[i] for i in range(len(mask)) if mask[i]]
    X_selected = X[:, mask]
    
    return X_selected, selected_names, mask


def select_features_ols_pvalue(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.05,
    top_k: int = 50,
    min_features: int = 8,
) -> List[str]:
    """
    Select features using OLS p-value filtering.
    
    Returns:
        List of selected feature names
    """
    # Handle NaN
    X_clean = X.fillna(0)
    y_clean = y.fillna(0)
    
    # Fit OLS
    X_const = sm.add_constant(X_clean)
    try:
        model = sm.OLS(y_clean, X_const).fit()
        pvalues = model.pvalues.drop("const", errors="ignore")
    except Exception:
        # If OLS fails, return all features
        return list(X.columns)
    
    # Filter by p-value
    significant = pvalues[pvalues < alpha].index.tolist()
    
    # Ensure minimum features
    if len(significant) < min_features:
        significant = pvalues.nsmallest(max(min_features, len(pvalues))).index.tolist()
    
    # Cap at top_k
    if len(significant) > top_k:
        significant = pvalues.nsmallest(top_k).index.tolist()
    
    return significant


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.Series:
    """
    Extract feature importance from a fitted model.
    
    Returns:
        Series with feature names as index and importance as values
    """
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_), index=feature_names).sort_values(ascending=False)
    else:
        return pd.Series(dtype=float)


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

class PurgedKFold:
    """
    Time-series cross-validation with purging to prevent look-ahead bias.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 5,
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups=None):
        """
        Generate train/test indices for each fold.
        """
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end + self.purge_gap
            test_end = test_start + fold_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "rf",
    task: str = "reg",
    n_splits: int = 5,
) -> Dict[str, float]:
    """
    Perform time-series cross-validation.
    
    Returns:
        Dict with mean metrics across folds
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    metrics_list = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = make_model(model_type, task)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task == "reg":
            metrics_list.append({
                "r2": r2_score(y_test, y_pred),
                "direction_accuracy": np.mean(np.sign(y_pred) == np.sign(y_test)),
            })
        else:
            metrics_list.append({
                "accuracy": accuracy_score(y_test, y_pred),
            })
    
    # Average across folds
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    return avg_metrics


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def tune_xgb_hyperparams(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    n_iter: int = 20,
) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters using randomized search.
    
    Returns:
        Dict of best hyperparameters
    """
    if not HAS_XGB:
        raise ImportError("XGBoost not installed")
    
    from sklearn.model_selection import RandomizedSearchCV
    
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    
    model = XGBRegressor(random_state=random_state)
    tscv = TimeSeriesSplit(n_splits=3)
    
    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=n_iter,
        cv=tscv,
        scoring="neg_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)
    
    return search.best_params_
