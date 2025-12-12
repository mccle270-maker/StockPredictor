import os
import sys
import numpy as np
import pandas as pd

# Make sure we can import from project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from prediction_model import (
    build_features_and_target,
    FEATURE_COLUMNS,
    MACRO_COLUMNS,
)
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error


def train_and_compare(
    ticker="^GSPC",
    period="5y",
    horizon=1,
    cnn_csv="cnn_meta_features_gspc.csv",
):
    """
    Compare baseline XGBoost vs. stacked XGBoost that uses cnn_prob_up
    as an extra feature. This is an offline experiment only.
    """

    print(f"Loading base features for {ticker}...")
    X, y, _, _, _ = build_features_and_target(
        ticker=ticker,
        period=period,
        horizon=horizon,
        use_vol_scaled_target=False,
    )

    feat_cols = FEATURE_COLUMNS + MACRO_COLUMNS
    print(f"Base feature matrix shape: {X.shape} (features: {len(feat_cols)})")

    print(f"Loading CNN meta-features from {cnn_csv}...")
    cnn_df = pd.read_csv(cnn_csv)

    # Basic cleanup
    if "cnn_prob_up" not in cnn_df.columns:
        raise ValueError(f"{cnn_csv} must contain a 'cnn_prob_up' column")

    cnn_vals = cnn_df["cnn_prob_up"].values

    # Align lengths by taking the last n rows of each (simple but safe start)
    n = min(len(X), len(cnn_vals), len(y))
    X_base = X[-n:]
    y_base = y[-n:]
    cnn_feature = cnn_vals[-n:].reshape(-1, 1)

    print(f"Aligned samples: {n}")

    # Build stacked feature matrix
    X_stacked = np.hstack([X_base, cnn_feature])

    # Simple 80/20 time-based split
    split_idx = int(n * 0.8)
    Xb_train, Xb_test = X_base[:split_idx], X_base[split_idx:]
    Xs_train, Xs_test = X_stacked[:split_idx], X_stacked[split_idx:]
    y_train, y_test = y_base[:split_idx], y_base[split_idx:]

    # Baseline XGBoost
    print("\nTraining BASE XGBoost (no CNN feature)...")
    xgb_base = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        tree_method="hist",
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    xgb_base.fit(Xb_train, y_train)
    y_pred_base = xgb_base.predict(Xb_test)

    r2_base = r2_score(y_test, y_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
    hit_base = (np.sign(y_pred_base) == np.sign(y_test)).mean()

    # Stacked XGBoost (with cnn_prob_up)
    print("\nTraining STACKED XGBoost (with cnn_prob_up)...")
    xgb_stack = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        tree_method="hist",
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    xgb_stack.fit(Xs_train, y_train)
    y_pred_stack = xgb_stack.predict(Xs_test)

    r2_stack = r2_score(y_test, y_pred_stack)
    rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    hit_stack = (np.sign(y_pred_stack) == np.sign(y_test)).mean()

    print("\n===== RESULTS (hold-out 20%) =====")
    print(f"Samples in test set: {len(y_test)}")
    print("\nBASE XGB (no CNN):")
    print(f"  R^2   : {r2_base:.4f}")
    print(f"  RMSE  : {rmse_base:.6f}")
    print(f"  Hit%  : {hit_base*100:.2f}%")

    print("\nSTACKED XGB (with cnn_prob_up):")
    print(f"  R^2   : {r2_stack:.4f}")
    print(f"  RMSE  : {rmse_stack:.6f}")
    print(f"  Hit%  : {hit_stack*100:.2f}%")

    return {
        "r2_base": r2_base,
        "rmse_base": rmse_base,
        "hit_base": hit_base,
        "r2_stack": r2_stack,
        "rmse_stack": rmse_stack,
        "hit_stack": hit_stack,
    }


if __name__ == "__main__":
    metrics = train_and_compare(
        ticker="^GSPC",
        period="5y",
        horizon=1,
        cnn_csv="cnn_meta_features_gspc.csv",
    )
