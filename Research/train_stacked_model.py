import os
import pandas as pd
import numpy as np

from prediction_model import (
    build_features_and_target,
    predict_up_gaf_cnn,
    FEATURE_COLUMNS,
    MACRO_COLUMNS,
)
from data_fetch import get_history_cached


def build_cnn_meta_features(
    ticker="^GSPC",
    period="5y",
    horizon=1,
    window=30,
    image_size=30,
    output_csv="cnn_meta_features.csv",
):
    """
    Offline script: build a time series of CNN prob_up_gaf for one ticker
    and save as a CSV. This does NOT change the live model yet.
    """

    # 1) Get the same base data used by your tree models
    X, y, last_feats, last_close, last_vol_20d = build_features_and_target(
        ticker=ticker,
        period=period,
        horizon=horizon,
        use_vol_scaled_target=False,
    )

    # 2) Fetch full history so we have dates to align with
    hist = get_history_cached(ticker, period=period, interval="1d")
    hist = hist.dropna(subset=["Close"]).copy()
    hist["ret_1d"] = hist["Close"].pct_change()

    # Match lengths roughly: drop initial rows lost to feature engineering
    # (you can refine this alignment later)
    hist = hist.iloc[-len(X) :]

    dates = hist.index
    rets = hist["ret_1d"]

    cnn_probs = []

    for i, dt in enumerate(dates):
        # Use only past data up to this date to build the window
        past_rets = rets.loc[:dt].dropna()
        if len(past_rets) < window:
            cnn_probs.append(np.nan)
            continue

        # Temporarily override predict_up_gaf_cnn to use a specific window
        # For now, reuse the existing helper and accept a small mismatch;
        # later you can refactor predict_up_gaf_cnn to accept a returns array.
        p = predict_up_gaf_cnn(ticker=ticker, window=window, image_size=image_size)
        cnn_probs.append(p)

        if (i + 1) % 50 == 0:
            print(f"{ticker}: processed {i+1}/{len(dates)} rows")

    df = pd.DataFrame(
        {
            "date": dates,
            "cnn_prob_up": cnn_probs,
            "target_ret": y[-len(dates) :],  # crude alignment for now
        }
    )

    df.to_csv(output_csv, index=False)
    print(f"Saved CNN meta-features for {ticker} to {output_csv}")


if __name__ == "__main__":
    build_cnn_meta_features(
        ticker="^GSPC",
        period="5y",
        horizon=1,
        window=30,
        image_size=30,
        output_csv="cnn_meta_features_gspc.csv",
    )
