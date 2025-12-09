import streamlit as st
import pandas as pd
import time
import numpy as np

from stock_screener import screen_stocks
from prediction_model import predict_next_for_ticker, track_predictions
from data_fetch import get_history, get_option_snapshot_features

def classify_alignment(pred_ret, put_call_oi_ratio):
    """
    Simple logic to see if model prediction and options sentiment 'agree'.
    """
    if pred_ret is None or put_call_oi_ratio is None:
        return "unknown"
    if pred_ret > 0 and put_call_oi_ratio < 1.0:
        return "bullish-aligned"
    if pred_ret < 0 and put_call_oi_ratio > 1.0:
        return "bearish-aligned"
    return "disagree"

def suggest_options_strategy(pred_ret, put_call_ratio, atm_iv):
    """
    Suggest options strategy based on model prediction and market indicators.
    """
    pred_pct = pred_ret * 100
    
    # Strong directional prediction
    if abs(pred_pct) > 2.0:
        if pred_pct > 0:
            if put_call_ratio and put_call_ratio > 1.2:
                return "🚀 BULLISH: Buy Calls (high put OI suggests potential short squeeze)", "bullish"
            else:
                return "📈 BULLISH: Buy Calls or Bull Call Spread", "bullish"
        else:
            if put_call_ratio and put_call_ratio < 0.8:
                return "📉 BEARISH: Buy Puts (low protection in market)", "bearish"
            else:
                return "🔻 BEARISH: Buy Puts or Bear Put Spread", "bearish"
    
    # Moderate prediction with high IV = sell premium
    elif abs(pred_pct) < 1.0 and atm_iv and atm_iv > 0.3:
        return "⚖️ NEUTRAL: Sell Iron Condor or Straddle (high IV)", "neutral"
    
    # Low conviction
    else:
        return "⏸️ NEUTRAL: Wait for clearer signal or diagonal spread", "neutral"

def run_app():
    st.title("Stock Predictor Dashboard")

    # Initialize session state
    if 'pred_df' not in st.session_state:
        st.session_state.pred_df = None
    if 'model_type' not in st.session_state:  # ADD THIS LINE
        st.session_state.model_type = "rf"

    st.sidebar.header("Settings")
    default_watchlist = "AAPL, NVDA"
    watchlist_text = st.sidebar.text_input(
        "Watchlist (comma-separated tickers)",
        value=default_watchlist,
    )

    model_label = st.sidebar.selectbox(
        "Model",
        ["Random Forest", "Gradient Boosting"],
    )
    model_type = "rf" if model_label == "Random Forest" else "gbrt"

    ret_thresh = st.sidebar.slider("Min |recent return| (%)", 0.0, 10.0, 3.0, 0.5)
    vol_spike_thresh = st.sidebar.slider("Min volume spike (× avg)", 0.5, 5.0, 1.5, 0.1)

    st.sidebar.markdown(
        """
        **Filters explanation**
        - Min |recent return|: required % move over the lookback window.
        - Min volume spike: how many × above average today's volume must be.
        
        **Note**: Processing includes delays to avoid rate limits.
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

        st.subheader(
            f"Next-Day Predictions ({model_label}) + Options Snapshot"
        )
        
        # Show progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, tk in enumerate(flagged):
            progress = (i + 1) / len(flagged)
            progress_bar.progress(progress)
            status_text.text(f"Processing {tk}... ({i+1}/{len(flagged)})")
            
            if i > 0:
                time.sleep(3)
            
            try:
                out = predict_next_for_ticker(
                    tk,
                    period="5y",
                    model_type=model_type,
                )
                opt = get_option_snapshot_features(tk)
                out.update(opt)
                out["signal_alignment"] = classify_alignment(
                    out["pred_next_ret"],
                    out["put_call_oi_ratio"],
                )
                results.append(out)
            except Exception as e:
                st.warning(f"{tk}: ERROR {e}")
        
        progress_bar.empty()
        status_text.empty()

        if results:
            # Store in session state
            st.session_state.pred_df = pd.DataFrame(results)
            st.session_state.pred_df["pred_next_ret_pct"] = st.session_state.pred_df["pred_next_ret"] * 100
            st.session_state.model_type = model_type
        else:
            st.warning("No predictions generated.")
            return

    # Display results if they exist in session state
    if st.session_state.pred_df is not None:
        pred_df = st.session_state.pred_df
        model_type = st.session_state.model_type

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

        # Bar chart of predicted returns
        bar_data = display.set_index("Ticker")["Predicted Return (%)"]
        st.subheader("Predicted Returns by Ticker")
        st.bar_chart(bar_data)

        # Feature importance display
        st.subheader("Top Features by Ticker")
        for _, row in pred_df.iterrows():
            with st.expander(f"{row['ticker']} - Top 5 Most Important Features"):
                st.markdown(row['top_features'])

        # Options Strategy Recommendations
        st.subheader("Options Strategy Recommendations")
        
        for _, row in pred_df.iterrows():
            strategy, sentiment = suggest_options_strategy(
                row['pred_next_ret'],
                row.get('put_call_oi_ratio'),
                row.get('atm_iv')
            )
            
            color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}[sentiment]
            
            with st.expander(f"{color} {row['ticker']} - Options Strategy"):
                st.write(f"**Prediction:** {row['pred_next_ret']*100:.2f}%")
                st.write(f"**Put/Call Ratio:** {row.get('put_call_oi_ratio', 'N/A'):.3f}" if row.get('put_call_oi_ratio') else "**Put/Call Ratio:** N/A")
                st.write(f"**IV:** {row.get('atm_iv', 'N/A'):.3f}" if row.get('atm_iv') else "**IV:** N/A")
                st.write(f"**Strategy:** {strategy}")
                
                if row.get('atm_iv'):
                    expected_move = row['last_close'] * row['atm_iv'] * np.sqrt(1/252)
                    st.write(f"**Expected 1-day move:** ±${expected_move:.2f}")
                    st.write(f"**Target strikes:** ${row['last_close'] - expected_move:.2f} to ${row['last_close'] + expected_move:.2f}")

        # Model Accuracy Testing
        st.subheader("Model Accuracy Testing")
        test_ticker = st.selectbox("Test prediction accuracy for:", display["Ticker"])
        
        if st.button("Run Accuracy Test"):
            with st.spinner(f"Testing {test_ticker} predictions..."):
                try:
                    results_test, accuracy = track_predictions(test_ticker, period="1y", model_type=model_type)
                    
                    if not results_test.empty:
                        st.metric("Direction Accuracy (Last 20 Days)", f"{accuracy*100:.1f}%")
                        
                        display_results = results_test[['date', 'predicted_return', 'actual_return', 
                                                      'predicted_price', 'actual_close', 'correct_direction']].copy()
                        display_results['predicted_return'] = display_results['predicted_return'] * 100
                        display_results['actual_return'] = display_results['actual_return'] * 100
                        
                        display_results.rename(columns={
                            'date': 'Date',
                            'predicted_return': 'Pred Return (%)',
                            'actual_return': 'Actual Return (%)',
                            'predicted_price': 'Pred Price',
                            'actual_close': 'Actual Price',
                            'correct_direction': 'Correct?',
                        }, inplace=True)
                        
                        st.dataframe(display_results.tail(10))
                        
                        chart_df = pd.DataFrame({
                            'Predicted': results_test['predicted_return'].values * 100,
                            'Actual': results_test['actual_return'].values * 100,
                        }, index=results_test['date'])
                        
                        st.line_chart(chart_df)
                    else:
                        st.warning("Not enough data to test accuracy.")
                except Exception as e:
                    st.error(f"Error testing accuracy: {e}")

        # Line chart for one ticker
        chosen = st.selectbox("Show price history for:", display["Ticker"], key="price_history_selector")
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

if __name__ == "__main__":
    run_app()
