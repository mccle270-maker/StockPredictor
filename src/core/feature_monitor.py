"""
Feature Importance Tracker
==========================

Tracks feature importances across walk-forward folds to identify
stable, consistently important features vs noisy one-off spikes.

Usage:
    from src.core.feature_monitor import FeatureImportanceTracker
    
    tracker = FeatureImportanceTracker()
    
    # During walk-forward training
    for fold_id, (X_train, y_train, X_test, y_test) in enumerate(folds):
        model.fit(X_train, y_train)
        tracker.update_importance(fold_id, feature_names, model.feature_importances_)
    
    # Get stable features
    stable = tracker.get_stable_features(min_folds=5, min_avg_importance=0.01)
    print(f"Stable features: {stable}")
    
    # Visualize
    df = tracker.get_importance_history_df()
    
    # Save for later
    tracker.save()
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
import pandas as pd

from ..config import CACHE_DIR

logger = logging.getLogger(__name__)

# Default save location
FEATURE_MONITOR_PATH = CACHE_DIR / "feature_importance_history.json"


class FeatureImportanceTracker:
    """
    Tracks feature importances across walk-forward folds.
    
    Helps identify:
    - Stable features: Consistently important across multiple folds
    - Unstable features: Spike in one fold but not others (likely noise)
    - Feature trends: Which features are gaining/losing importance
    
    Attributes:
        history: List of dicts, one per fold, with feature importances
        metadata: Dict with tracker metadata (created, updated, etc.)
    """
    
    def __init__(
        self,
        save_path: Optional[Path] = None,
        auto_save: bool = True,
    ):
        """
        Initialize the tracker.
        
        Args:
            save_path: Path to save/load state (default: CACHE_DIR/feature_importance_history.json)
            auto_save: If True, automatically save after each update
        """
        self.save_path = Path(save_path) if save_path else FEATURE_MONITOR_PATH
        self.auto_save = auto_save
        
        # History: list of fold records
        self.history: List[Dict[str, Any]] = []
        
        # Metadata
        self.metadata: Dict[str, Any] = {
            "created": datetime.utcnow().isoformat(),
            "updated": datetime.utcnow().isoformat(),
            "total_folds": 0,
        }
        
        # Try to load existing state
        if self.save_path.exists():
            try:
                self.load()
                logger.info(f"Loaded {len(self.history)} folds from {self.save_path}")
            except Exception as e:
                logger.warning(f"Could not load existing state: {e}")
    
    def update_importance(
        self,
        fold_id: int,
        feature_names: List[str],
        importances: np.ndarray,
        ticker: Optional[str] = None,
        model_type: Optional[str] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Record feature importances from a training fold.
        
        Args:
            fold_id: Unique identifier for this fold (e.g., 0, 1, 2...)
            feature_names: List of feature names
            importances: Array of importance values (same length as feature_names)
            ticker: Optional ticker symbol for context
            model_type: Optional model type (rf, xgb, etc.)
            extra: Optional dict of additional metadata
        """
        if len(feature_names) != len(importances):
            raise ValueError(f"Mismatch: {len(feature_names)} names vs {len(importances)} importances")
        
        # Normalize importances to sum to 1
        importances = np.array(importances, dtype=float)
        if importances.sum() > 0:
            importances = importances / importances.sum()
        
        # Build record
        record = {
            "fold_id": fold_id,
            "timestamp": datetime.utcnow().isoformat(),
            "ticker": ticker,
            "model_type": model_type,
            "n_features": len(feature_names),
            "importances": dict(zip(feature_names, importances.tolist())),
        }
        
        if extra:
            record["extra"] = extra
        
        # Check if fold_id already exists (update vs append)
        existing_idx = None
        for i, h in enumerate(self.history):
            if h.get("fold_id") == fold_id and h.get("ticker") == ticker:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.history[existing_idx] = record
            logger.debug(f"Updated fold {fold_id} for {ticker}")
        else:
            self.history.append(record)
            logger.debug(f"Added fold {fold_id} for {ticker}")
        
        # Update metadata
        self.metadata["updated"] = datetime.utcnow().isoformat()
        self.metadata["total_folds"] = len(self.history)
        
        # Auto-save
        if self.auto_save:
            self.save()
    
    def get_stable_features(
        self,
        min_folds: int = 5,
        min_avg_importance: float = 0.01,
        ticker: Optional[str] = None,
    ) -> List[str]:
        """
        Get features that are consistently important across folds.
        
        Args:
            min_folds: Minimum number of folds a feature must appear in
            min_avg_importance: Minimum average importance (normalized, so 0.01 = 1%)
            ticker: Filter by ticker (None = all tickers)
        
        Returns:
            List of stable feature names, sorted by average importance (descending)
        """
        # Filter history by ticker if specified
        records = self.history
        if ticker:
            records = [r for r in records if r.get("ticker") == ticker]
        
        if len(records) < min_folds:
            logger.warning(f"Only {len(records)} folds available, need {min_folds}")
            return []
        
        # Aggregate importances across folds
        feature_stats: Dict[str, Dict[str, Any]] = {}
        
        for record in records:
            for feat, imp in record.get("importances", {}).items():
                if feat not in feature_stats:
                    feature_stats[feat] = {"values": [], "count": 0}
                feature_stats[feat]["values"].append(imp)
                feature_stats[feat]["count"] += 1
        
        # Calculate stats and filter
        stable_features = []
        for feat, stats in feature_stats.items():
            count = stats["count"]
            avg_imp = np.mean(stats["values"])
            
            if count >= min_folds and avg_imp >= min_avg_importance:
                stable_features.append((feat, avg_imp, count))
        
        # Sort by average importance (descending)
        stable_features.sort(key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in stable_features]
    
    def get_importance_history_df(
        self,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get feature importance history as a DataFrame for visualization.
        
        Args:
            ticker: Filter by ticker (None = all tickers)
        
        Returns:
            DataFrame with columns: fold_id, feature, importance, ticker, timestamp
        """
        rows = []
        
        for record in self.history:
            if ticker and record.get("ticker") != ticker:
                continue
            
            fold_id = record.get("fold_id")
            rec_ticker = record.get("ticker")
            timestamp = record.get("timestamp")
            
            for feat, imp in record.get("importances", {}).items():
                rows.append({
                    "fold_id": fold_id,
                    "feature": feat,
                    "importance": imp,
                    "ticker": rec_ticker,
                    "timestamp": timestamp,
                })
        
        if not rows:
            return pd.DataFrame(columns=["fold_id", "feature", "importance", "ticker", "timestamp"])
        
        return pd.DataFrame(rows)
    
    def get_summary_df(
        self,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get summary statistics for each feature.
        
        Returns DataFrame with columns:
            feature, count, mean, std, min, max, cv (coefficient of variation)
        """
        df = self.get_importance_history_df(ticker=ticker)
        
        if df.empty:
            return pd.DataFrame(columns=["feature", "count", "mean", "std", "min", "max", "cv"])
        
        summary = df.groupby("feature")["importance"].agg(["count", "mean", "std", "min", "max"])
        summary["cv"] = summary["std"] / summary["mean"].replace(0, np.nan)  # Coefficient of variation
        summary = summary.sort_values("mean", ascending=False).reset_index()
        
        return summary
    
    def get_top_features(
        self,
        n: int = 10,
        metric: str = "mean",
        ticker: Optional[str] = None,
    ) -> List[str]:
        """
        Get top N features by specified metric.
        
        Args:
            n: Number of features to return
            metric: 'mean' (average importance) or 'stability' (low CV)
            ticker: Filter by ticker
        
        Returns:
            List of top feature names
        """
        summary = self.get_summary_df(ticker=ticker)
        
        if summary.empty:
            return []
        
        if metric == "stability":
            # Low CV = more stable
            summary = summary.dropna(subset=["cv"])
            summary = summary.sort_values("cv", ascending=True)
        else:
            # Default: sort by mean importance
            summary = summary.sort_values("mean", ascending=False)
        
        return summary["feature"].head(n).tolist()
    
    def get_unstable_features(
        self,
        cv_threshold: float = 1.0,
        min_folds: int = 3,
        ticker: Optional[str] = None,
    ) -> List[str]:
        """
        Get features with high variance (potentially unstable/noisy).
        
        Args:
            cv_threshold: Coefficient of variation threshold (higher = more unstable)
            min_folds: Minimum folds to consider
            ticker: Filter by ticker
        
        Returns:
            List of unstable feature names
        """
        summary = self.get_summary_df(ticker=ticker)
        
        if summary.empty:
            return []
        
        unstable = summary[
            (summary["count"] >= min_folds) & 
            (summary["cv"] > cv_threshold)
        ]
        
        return unstable["feature"].tolist()
    
    def clear(self) -> None:
        """Clear all history."""
        self.history = []
        self.metadata["updated"] = datetime.utcnow().isoformat()
        self.metadata["total_folds"] = 0
        
        if self.auto_save:
            self.save()
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save state to JSON file.
        
        Args:
            path: Path to save (default: self.save_path)
        
        Returns:
            Path where file was saved
        """
        path = Path(path) if path else self.save_path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "metadata": self.metadata,
            "history": self.history,
        }
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.debug(f"Saved {len(self.history)} folds to {path}")
        return path
    
    def load(self, path: Optional[Path] = None) -> None:
        """
        Load state from JSON file.
        
        Args:
            path: Path to load (default: self.save_path)
        """
        path = Path(path) if path else self.save_path
        
        with open(path, "r") as f:
            state = json.load(f)
        
        self.metadata = state.get("metadata", {})
        self.history = state.get("history", [])
        
        logger.debug(f"Loaded {len(self.history)} folds from {path}")
    
    def __len__(self) -> int:
        """Number of recorded folds."""
        return len(self.history)
    
    def __bool__(self) -> bool:
        """Tracker is always truthy when it exists (not empty-based)."""
        return True
    
    def __repr__(self) -> str:
        return f"FeatureImportanceTracker(folds={len(self.history)}, path={self.save_path})"


# Convenience function for global tracker
_global_tracker: Optional[FeatureImportanceTracker] = None


def get_feature_tracker() -> FeatureImportanceTracker:
    """Get the global feature importance tracker (singleton)."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = FeatureImportanceTracker()
    return _global_tracker


def reset_feature_tracker() -> None:
    """Reset the global tracker."""
    global _global_tracker
    _global_tracker = None
