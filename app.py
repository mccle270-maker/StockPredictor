import streamlit as st
import pandas as pd

from stock_screener import screen_stocks
from prediction_model import predict_next_for_ticker
from data_fetch import get_history, get_option_snapshot_features


def classify_alignment(pred_ret, put_call_oi_ratio):
    """
    Simple logic to see if RF/GBRT prediction and options sentiment 'agree'.
    """
    if pred_ret is None or put_call_oi_ratio is None:
        return "unknown"
    if pred_ret > 0 and put_call_oi_ratio < 1.0:
        return "bullish-aligned"
    if pred_ret < 0 and put_call_oi_ratio > 1.0:
        return "bearish-aligned"
    return "disagree"


def run_app():
    st.title("Stock Predictor Dashboard")

    st.sidebar.header("Settings")
    default_watchlist = "AAPL, TSLA, NVDA, MSFT, SPY, ^GSPC"
    watchlist_text = st.sidebar.text_input(
        "Watchlist (comma-separated tickers)",
        value=default_watchlist,
    )

    # NEW: model choice
    model_label = st.sidebar.selectbox(
        "Model",
        ["Random Forest", "Gradient Boosting"],
    )  # simple dropdown widget [web:196][web:205]
    model_type = "rf" if model_label == "Random Forest" else "gbrt"

    ret_thresh = st.sidebar.slider("Min |recent return| (%)", 0.0, 10.0, 3.0, 0.5)
    vol_spike_thresh = st.sidebar.slider("Min volume spike (× avg)", 0.5, 5.0, 1.5, 0.1)

    st.sidebar.markdown(
        """
        **Filters explanation**
        - Min |recent return|: required % move over the lookback window.
        - Min volume spike: how many × above average today's volume must be.
        """
    )

    tickers = [t.strip() for t in watchlist_text.split(",") if t.strip()]

    if st.sidebar.button("Run Screener + Model"):
        if not tickers:
            st.error("Please enter at least one ticker.")
            return

        # Screener results
        st.subheader("Screener Results")
        screener_df = screen_stocks(
            tickers,
            ret_thresh=ret_thresh / 100.0,
            vol_spike_thresh=vol_spike_thresh,
        )
        if screener_df.empty:
            st.warning("No data returned for these tickers.")
            return

        st.dataframe(screener_df)

        flagged_df = screener_df[screener_df["flag"] == True]
        if not flagged_df.empty:
            st.write("**Flagged by screener:**")
            st.dataframe(flagged_df)

        flagged = flagged_df["ticker"].tolist()
        if not flagged:
            st.info("No tickers flagged by screener; using full watchlist.")
            flagged = tickers

        st.subheader(f"Next-Day Predictions ({model_label}) + Options Snapshot")
        results = []
        for tk in flagged:
            try:
                # Use chosen model type here
                out = predict_next_for_ticker(tk, period="5y", model_type=model_type)

                # Live options features
                opt = get_option_snapshot_features(tk)
                out.update(opt)

                out["signal_alignment"] = classify_alignment(
                    out["pred_next_ret"],
                    out["put_call_oi_ratio"],
                )

                results.append(out)
            except Exception as e:
                st.warning(f"{tk}: ERROR {e}")

        if results:
            pred_df = pd.DataFrame(results)
            pred_df["pred_next_ret_pct"] = pred_df["pred_next_ret"] * 100
            display = pred_df[[
                "ticker",
                "model_type",
                "last_close",
                "vol_20d",
                "pe_ratio",
                "num_features",
                "atm_iv",
                "put_call_oi_ratio",
                "pred_next_ret_pct",
                "pred_next_price",
                "opt_exp",
                "signal_alignment",
            ]].copy()
            
            display.rename(columns={
                "ticker": "Ticker",
                "model_type": "Model",
                "last_close": "Last Close",
                "vol_20d": "Vol 20D",
                "pe_ratio": "P/E",
                "num_features": "# Features",
                "atm_iv": "ATM IV",
                "put_call_oi_ratio": "Put/Call OI Ratio",
                "pred_next_ret_pct": "Predicted Return (%)",
                "pred_next_price": "Predicted Price",
                "opt_exp": "Opt Expiry",
                "signal_alignment": "Signal",
            }, inplace=True)
            
            st.dataframe(display)
            
            # Add feature importance display
            st.subheader("Top Features by Ticker")
            for _, row in pred_df.iterrows():
                with st.expander(f"{row['ticker']} - Top 5 Most Important Features"):
                    st.write(row['top_features'])

            # Bar chart of predicted returns [web:236][web:245]
            bar_data = display.set_index("Ticker")["Predicted Return (%)"]
            st.subheader("Predicted Returns by Ticker")
            st.bar_chart(bar_data)

            # Line chart for one ticker [web:241][web:245]
            chosen = st.selectbox("Show price history for:", display["Ticker"])
            hist = get_history(chosen, period="3mo", interval="1d")
            prices = hist["Close"].copy()
            if not prices.empty:
                last_date = prices.index[-1]
                row = pred_df[pred_df["ticker"] == chosen].iloc[0]
                pred_price = row["pred_next_price"]

                extra_point = pd.Series(
                    [pred_price],
                    index=[last_date + pd.Timedelta(days=1)],
                )
                future = pd.concat([prices, extra_point])

                st.subheader(f"{chosen} recent prices + predicted next price")
                st.line_chart(future)
            else:
                st.warning(f"No recent price data for {chosen}.")
        else:
            st.warning("No predictions generated.")


if __name__ == "__main__":
    run_app()
