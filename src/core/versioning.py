"""
Configuration Snapshot & Versioning Utilities
============================================

Captures all key config parameters for reproducibility.
"""
import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..config import (
    TRADE_LIMIT_CONFIG,
    POSITION_SIZING_CONFIG,
    REGIME_FILTER_CONFIG,
    ZSCORE_GATING_CONFIG,
)

# --- Utility: Hash a dict for feature set versioning ---
def dict_hash(d: dict) -> str:
    """Return a short hash of a dictionary (for feature set versioning)."""
    dhash = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
    return dhash[:10]

# --- Main snapshot function ---
def create_config_snapshot(
    model_version: str,
    feature_set: dict,
    run_id: str = None,
    extra: dict = None,
) -> Dict[str, Any]:
    """
    Build a config snapshot dict for reproducibility.
    """
    snapshot = {
        "run_id": run_id or str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "feature_set_hash": dict_hash(feature_set),
        "zscore_threshold": ZSCORE_GATING_CONFIG.get("min_zscore", 1.0),
        "volatility_sizing": {
            "target_daily_vol": POSITION_SIZING_CONFIG.get("target_daily_vol"),
            "vol_lookback_days": POSITION_SIZING_CONFIG.get("vol_lookback_days"),
            "max_leverage": POSITION_SIZING_CONFIG.get("max_leverage"),
            "min_position_pct": POSITION_SIZING_CONFIG.get("min_position_pct"),
        },
        "trade_limits": {
            "max_trades_per_ticker": TRADE_LIMIT_CONFIG.get("max_trades_per_ticker"),
            "period": TRADE_LIMIT_CONFIG.get("period"),
        },
        "regime_filter": {
            "enabled": REGIME_FILTER_CONFIG.get("enabled", True),
            "spy_dma_period": REGIME_FILTER_CONFIG.get("spy_dma_period"),
            "vix_high_threshold": REGIME_FILTER_CONFIG.get("vix_high_threshold"),
            "vix_extreme_threshold": REGIME_FILTER_CONFIG.get("vix_extreme_threshold"),
            "min_conviction_override": REGIME_FILTER_CONFIG.get("min_conviction_override"),
        },
    }
    if extra:
        snapshot.update(extra)
    return snapshot

# --- Save snapshot to file ---
def save_config_snapshot(snapshot: dict, out_dir: Path = None) -> Path:
    """
    Save config snapshot as a JSON file. Returns the file path.
    """
    out_dir = out_dir or Path("snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = snapshot.get("run_id")
    fname = f"config_snapshot_{run_id}.json"
    fpath = out_dir / fname
    with open(fpath, "w") as f:
        json.dump(snapshot, f, indent=2)
    return fpath
