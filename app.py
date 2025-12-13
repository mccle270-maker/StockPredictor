import os
import streamlit as st
import pandas as pd
import time
import numpy as np


# Make FRED_API_KEY available to prediction_model via environment variable
if "FRED_API_KEY" in st.secrets:
    os.environ["FRED_API_KEY"] = st.secrets["FRED_API_KEY"]
    print(f"[DEBUG] FRED_API_KEY set: {os.environ['FRED_API_KEY'][:8]}...")
else:
    print("[DEBUG] FRED_API_KEY NOT in secrets")


from prediction_model import (
    predict_next_for_ticker,
    track_predictions,
    backtest_one_ticker,
    backtest_compare_one_ticker,
    walk_forward_backtest,
    analyze_feature_significance,
)


from stock_screener import screen_stocks
from prediction_model import (
    predict_next_for_ticker,
    track_predictions,
    analyze_feature_significance,
    make_gaf_image_from_returns,
)
from data_fetch import (
    get_history_cached,
    get_option_snapshot_features,
    get_news_for_ticker,
    get_atm_greeks,
)
from yfinance.exceptions import YFRateLimitError
from monte_carlo_pricer import option_mc_ev
from scipy.stats import norm


from option_pricing import (
    OptionSpec,
    HestonParams,
    PricingModel,
    price_option,
)


try:
    import squarequant as sq
except ImportError:
    sq = None


def get_heston_params_for_ticker(ticker: str) -> HestonParams | None:
    params_by_ticker = {
        "AAPL": HestonParams(v0=0.04, theta=0.04, kappa=1.5, sigma=0.3, rho=-0.6),
        "NVDA": HestonParams(v0=0.06, theta=0.05, kappa=1.2, sigma=0.5, rho=-0.7),
    }
    return params_by_ticker.get(ticker.upper())


def deflated_sharpe_ratio(daily_returns: pd.Series, n_trials: int, risk_free: float = 0.0):
    r = daily_returns.dropna()
    if n_trials is None or n_trials <= 0 or len(r) < 5 or r.std() == 0:
        return None


    excess = r - risk_free
    mu = excess.mean()
    sigma = excess.std()
    T = len(r)


    sharpe_daily = mu / sigma
    z_strat = sharpe_daily * np.sqrt(T)


    if n_trials == 1:
        return float(norm.cdf(z_strat))


    z_alpha = norm.ppf(1.0 - 1.0 / n_trials)
    z_deflated = z_strat - z_alpha
    return float(norm.cdf(z_deflated))


def compute_sharpe(daily_returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252):
    daily_excess = daily_returns - risk_free
    if len(daily_excess) < 2 or daily_excess.std() == 0:
        return None
    mean_ret = daily_excess.mean()
    vol = daily_excess.std()
    return (mean_ret / vol) * np.sqrt(periods_per_year)


def detect_big_news(articles, sent_thresh: float = 0.5) -> bool:
    if not articles:
        return False


    keywords = [
        "earnings", "guidance", "downgrade", "upgrade",
        "lawsuit", "investigation", "merger", "acquisition",
        "bankruptcy", "sec charges", "fraud", "buyback",
    ]


    for art in articles:
        title = (art.get("title") or "").lower()
        sent = art.get("sentiment")
        if any(k in title for k in keywords):
            return True
        if isinstance(sent, (int, float)) and abs(sent) >= sent_thresh:
            return True


    return False


def suggest_model_for_ticker(ticker: str, horizon: int = 1) -> str:
    tk = ticker.upper()
    if horizon == 1:
        if tk in ["AAPL", "GOOGL"]:
            return "xgb"
        if tk in ["NVDA"]:
            return "gbrt"
        if tk in ["MSFT"]:
            return "rf"
        if tk in ["AMZN"]:
            return "rf"
        return "rf"
    else:
        return "rf"


def classify_alignment(pred_ret, put_call_oi_ratio):
    if pred_ret is None or put_call_oi_ratio is None:
        return "unknown"
    if pred_ret > 0 and put_call_oi_ratio < 1.0:
        return "bullish-aligned"
    if pred_ret < 0 and put_call_oi_ratio > 1.0:
        return "bearish-aligned"
    return "disagree"


def suggest_options_strategy(pred_ret, put_call_ratio, atm_iv, horizon=1):
    pred_pct = pred_ret * 100


    threshold_multiplier = {1: 1.0, 2: 1.4, 3: 1.7, 4: 2.0, 5: 2.3}.get(horizon, 1.0)
    adjusted_threshold = 1.0 * threshold_multiplier


    if abs(pred_pct) > adjusted_threshold:
        if pred_pct > 0:
            if put_call_ratio and put_call_ratio > 1.2:
                return (
                    "🚀 BULLISH: Buy Calls (high put OI suggests potential short squeeze)",
                    "bullish",
                )
            else:
                return "📈 BULLISH: Buy Calls or Bull Call Spread", "bullish"
        else:
            if put_call_ratio and put_call_ratio < 0.8:
                return "📉 BEARISH: Buy Puts (low protection in market)", "bearish"
            else:
                return "🔻 BEARISH: Buy Puts or Bear Put Spread", "bearish"


    elif abs(pred_pct) < (0.5 * threshold_multiplier) and atm_iv and atm_iv > 0.35:
        return "⚖️ NEUTRAL: Sell Iron Condor or Straddle (high IV)", "neutral"


    else:
        return "⏸️ NEUTRAL: Wait for clearer signal or diagonal spread", "neutral"


def run_app():
    st.title("Stock Predictor Dashboard")


    if "pred_df" not in st.session_state:
        st.session_state.pred_df = None
    if "model_type" not in st.session_state:
        st.session_state.model_type = "rf"


    # ----- Sidebar controls -----
    st.sidebar.header("Settings")
    default_watchlist = "AAPL, NVDA"
    watchlist_text = st.sidebar.text_input(
        "Watchlist (comma-separated tickers)",
        value=default_watchlist,
    )


    # ⭐ UPDATED: Horizon selector now supports 1-5 days
    st.sidebar.subheader("Prediction Settings")
    prediction_horizon = st.sidebar.selectbox(
        "Prediction Horizon",
        [1, 2, 3, 4, 5],
        index=0,
        help="How many days ahead to predict (1=next day, 5=week out)",
    )
    horizon_label = {1: "1-Day", 2: "2-Day", 3: "3-Day", 4: "4-Day", 5: "5-Day"}[prediction_horizon]

    # ⭐ NEW: Auto-optimization toggle
    auto_optimize = st.sidebar.checkbox(
        "Auto-optimize features per stock",
        value=True,
        help="Automatically prunes weak features for each stock to improve predictions. Recommended: ON",
    )


    # Model selection
    st.sidebar.subheader("Model Selection")
    model_label = st.sidebar.selectbox(
        "Model",
        ["Auto", "Random Forest", "Gradient Boosting", "XGBoost"],
    )


    if model_label == "Auto":
        if prediction_horizon == 1:
            model_type = "xgb"
        else:
            model_type = "rf"
    else:
        model_type = {
            "Random Forest": "rf",
            "Gradient Boosting": "gbrt",
            "XGBoost": "xgb",
        }[model_label]


    # ⭐ UPDATED: Adjusted recommendations for longer horizons
    if prediction_horizon == 1:
        recommended = "XGBoost"
        rec_detail = "Gradient boosting models often work well for very short-term moves."
    elif prediction_horizon <= 3:
        recommended = "Random Forest"
        rec_detail = "Tree ensembles tend to be more stable for 2–3 day horizons with FRED macro features."
    else:
        recommended = "Random Forest"
        rec_detail = "4-5 day predictions benefit from macro features; RF handles multi-day stability well."


    st.sidebar.info(f"💡 Suggested for {horizon_label}: {recommended}\n\n{rec_detail}")


    if model_type == "xgb" and prediction_horizon > 1:
        st.sidebar.warning(
            "⚠️ XGBoost can be unstable on multi-day horizons. Consider Random Forest for 2-5 day predictions."
        )


    st.sidebar.subheader("Option Pricing Model")
    pricing_model_label = st.sidebar.selectbox(
        "Pricing Engine",
        ["Black-Scholes", "Heston (stochastic vol)"],
        index=0,
        help="Black-Scholes is fast and simple; Heston uses stochastic volatility (requires calibrated params).",
    )
    pricing_model = (
        PricingModel.BLACK_SCHOLES
        if pricing_model_label == "Black-Scholes"
        else PricingModel.HESTON
    )


    st.sidebar.subheader("Screener Filters")
    ret_thresh = st.sidebar.slider("Min |recent return| (%)", 0.0, 10.0, 3.0, 0.5)
    vol_spike_thresh = st.sidebar.slider("Min volume spike (× avg)", 0.5, 5.0, 1.5, 0.1)


    max_tickers = st.sidebar.slider(
        "Max tickers per run (to avoid rate limits)",
        1,
        20,
        5,
    )


    st.sidebar.markdown(
        """
        **Filters explanation**
        - Min |recent return|: required % move over the lookback window.
        - Min volume spike: how many × above average today's volume must be.
        
        **Note**: Processing includes delays to avoid rate limits.
        """
    )


    st.sidebar.subheader("Candidate Filters")
    min_move = st.sidebar.slider(
        "Min |predicted return| (%) for candidates",
        0.0,
        5.0,
        1.0,
        0.1,
    )
    min_iv = st.sidebar.slider("Min ATM IV", 0.0, 1.0, 0.2, 0.05)
    max_iv = st.sidebar.slider("Max ATM IV", 0.0, 1.0, 0.8, 0.05)
    exclude_disagree = st.sidebar.checkbox(
        "Exclude 'disagree' signals from candidates",
        value=True,
    )


    st.sidebar.subheader("Overfitting / DSR")
    n_trials = st.sidebar.slider(
        "Approx. # of strategy variants you tried",
        1,
        100,
        20,
        help="Used for Deflated Sharpe (DSR); higher = stricter test against overfitting.",
    )


    tickers = [t.strip() for t in watchlist_text.split(",") if t.strip()]


    # ---------------- Main run button ----------------
    if st.sidebar.button("Run Screener + Model"):
        if not tickers:
            st.error("Please enter at least one ticker.")
            return


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


        if "flag" in screener_df.columns:
            flagged_df = screener_df[screener_df["flag"] == True]
        else:
            flagged_df = pd.DataFrame(columns=screener_df.columns)


        if not flagged_df.empty:
            st.write("**Flagged by screener:**")
            st.dataframe(flagged_df)


        flagged = flagged_df["ticker"].tolist()
        if not flagged:
            st.info("No tickers flagged by screener; using full watchlist.")
            flagged = tickers


        if len(flagged) > max_tickers:
            st.warning(
                f"Limiting to first {max_tickers} tickers this run to avoid Yahoo Finance rate limits."
            )
            flagged = flagged[:max_tickers]


        st.subheader(
            f"{horizon_label} Predictions ({model_label}) + Options Snapshot"
        )


        progress_bar = st.progress(0)
        status_text = st.empty()


        results = []
        for i, tk in enumerate(flagged):
            progress = (i + 1) / len(flagged)
            progress_bar.progress(progress)
            status_text.text(f"Processing {tk}... ({i+1}/{len(flagged)})")


            if i > 0:
                time.sleep(5)


            try:
                # ⭐ KEY CHANGE: Pass auto_optimize parameter
                out = predict_next_for_ticker(
                    tk,
                    period="5y",
                    model_type=model_type,
                    horizon=prediction_horizon,
                    use_vol_scaled_target=False,
                    auto_optimize=auto_optimize,  # ← NEW: Use auto-optimization
                )


                opt = get_option_snapshot_features(tk)
                out.update(opt)


                atm_iv = out.get("atm_iv")
                last_close = out.get("last_close")


                if atm_iv is not None and out.get("vol_20d") is not None:
                    try:
                        out["iv_minus_realized"] = float(atm_iv) - float(out["vol_20d"])
                    except Exception:
                        out["iv_minus_realized"] = None
                else:
                    out["iv_minus_realized"] = None


                out["theo_atm_call_price"] = None
                try:
                    opt_exp = out.get("opt_exp")
                    if last_close is not None and atm_iv is not None and opt_exp:
                        opt_exp_date = pd.to_datetime(opt_exp).date()
                        val_date = pd.Timestamp.today().date()
                        opt_spec = OptionSpec(
                            spot=float(last_close),
                            strike=float(last_close),
                            maturity_date=opt_exp_date,
                            valuation_date=val_date,
                            rate=0.05,
                            div_yield=0.0,
                            vol=float(atm_iv),
                            is_call=True,
                        )
                        heston_params = None
                        if pricing_model == PricingModel.HESTON:
                            heston_params = get_heston_params_for_ticker(tk)
                            if heston_params is None:
                                theo_price = price_option(opt_spec, model=PricingModel.BLACK_SCHOLES)
                            else:
                                theo_price = price_option(opt_spec, model=pricing_model, heston_params=heston_params)
                        else:
                            theo_price = price_option(opt_spec, model=pricing_model)


                        out["theo_atm_call_price"] = float(theo_price)
                except Exception as pe:
                    print(f"Pricing error for {tk}: {pe}")
                    out["theo_atm_call_price"] = None


                if atm_iv is not None and last_close is not None:
                    try:
                        mc_res = option_mc_ev(
                            s0=float(last_close),
                            mu=float(out["pred_next_ret"]),
                            sigma=float(atm_iv),
                            days=int(prediction_horizon),
                            premium=1.0,
                            strike=float(last_close),
                            n_paths=5000,
                            is_call=True,
                        )
                        out.update(mc_res)
                    except Exception as mc_e:
                        print(f"MC error for {tk}: {mc_e}")


                out["signal_alignment"] = classify_alignment(
                    out["pred_next_ret"],
                    out.get("put_call_oi_ratio"),
                )
                results.append(out)


            except YFRateLimitError:
                st.error(
                    "Yahoo Finance is rate limiting this app (Too Many Requests). "
                    "Try again later and/or use fewer tickers per run."
                )
                break


            except Exception as e:
                st.warning(f"{tk}: ERROR {e}")


        progress_bar.empty()
        status_text.empty()


        if results:
            st.session_state.pred_df = pd.DataFrame(results)
            st.session_state.pred_df["pred_next_ret_pct"] = (
                st.session_state.pred_df["pred_next_ret"] * 100
            )
            st.session_state.model_type = model_type
            st.session_state.screener_df = screener_df
            st.session_state.prediction_horizon = prediction_horizon
            st.session_state.auto_optimize = auto_optimize  # ← Store setting
        else:
            st.warning("No predictions generated.")
            return


    # ---------------- Display results ----------------
    if st.session_state.pred_df is not None:
        pred_df = st.session_state.pred_df
        model_type = st.session_state.model_type
        display_horizon = st.session_state.get("prediction_horizon", 1)
        display_horizon_label = {1: "1-Day", 2: "2-Day", 3: "3-Day", 4: "4-Day", 5: "5-Day"}[display_horizon]
        is_auto_optimized = st.session_state.get("auto_optimize", True)

        # ⭐ NEW: Show optimization status
        if is_auto_optimized:
            st.info("🔧 Auto-optimization: ON - Using optimized features per stock")
        else:
            st.info("🔧 Auto-optimization: OFF - Using all 60 features")


        cols_to_show = [
            "ticker",
            "model_type",
            "horizon",
            "last_close",
            "vol_20d",
            "pe_ratio",
            "num_features",
            "atm_iv",
            "iv_minus_realized",
            "put_call_oi_ratio",
            "pred_next_ret_pct",
            "pred_next_price",
            "prob_up",
            "prob_down",
            "prob_up_gaf",
            "opt_exp",
            "theo_atm_call_price",
            "signal_alignment",
        ]


        for mc_col in ["mc_ev", "mc_pop_gt0"]:
            if mc_col in pred_df.columns:
                cols_to_show.append(mc_col)


        display = pred_df[cols_to_show].copy()


        rename_map = {
            "ticker": "Ticker",
            "model_type": "Model",
            "horizon": "Days Ahead",
            "last_close": "Last Close",
            "vol_20d": "Vol 20D",
            "pe_ratio": "P/E",
            "num_features": "# Features Used",  # ← Updated label
            "atm_iv": "ATM IV",
            "iv_minus_realized": "IV - Realized Vol",
            "put_call_oi_ratio": "Put/Call OI Ratio",
            "pred_next_ret_pct": f"Predicted {display_horizon_label} Return (%)",
            "pred_next_price": "Predicted Price",
            "prob_up": "Prob Up",
            "prob_down": "Prob Down",
            "prob_up_gaf": "GAF-CNN Prob Up",
            "opt_exp": "Opt Expiry",
            "theo_atm_call_price": "Theo ATM Call",
            "signal_alignment": "Signal",
            "mc_ev": "MC EV (P/L)",
            "mc_pop_gt0": "MC POP (>0)",
        }
        display.rename(columns=rename_map, inplace=True)


        st.dataframe(display)


        # --- Top candidates to watch today ---
        cand_df = st.session_state.pred_df.copy()
        cand_df["abs_pred_pct"] = cand_df["pred_next_ret_pct"].abs()


        mask = cand_df["abs_pred_pct"] >= min_move
        mask &= cand_df["atm_iv"].between(min_iv, max_iv)
        if exclude_disagree:
            mask &= cand_df["signal_alignment"] != "disagree"


        cand_df = cand_df[mask]


        st.subheader("Top Model Candidates (filtered)")
        if not cand_df.empty:
            cand_df["score"] = cand_df["abs_pred_pct"]
            cand_df = cand_df.sort_values("score", ascending=False)


            st.dataframe(
                cand_df[
                    [
                        "ticker",
                        "pred_next_ret_pct",
                        "pred_next_price",
                        "num_features",  # ← Show feature count in candidates
                        "atm_iv",
                        "put_call_oi_ratio",
                        "signal_alignment",
                        "prob_up",
                        "prob_up_gaf",
                    ]
                ].rename(
                    columns={
                        "ticker": "Ticker",
                        "num_features": "Features",
                        "prob_up": "Prob Up",
                        "prob_up_gaf": "GAF-CNN Prob Up",
                    }
                )
            )


            tickers_list = cand_df["ticker"].tolist()
            selected_ticker = st.selectbox(
                "Recommended tickers to watch (based on your filters)",
                options=tickers_list,
                key="recommended_ticker_select",
            )
            st.write(f"You selected **{selected_ticker}** from today's candidates.")
        else:
            st.write("No strong candidates today based on current filters.")


        bar_data = display.set_index("Ticker")[
            f"Predicted {display_horizon_label} Return (%)"
        ]
        st.subheader(f"Predicted {display_horizon_label} Returns by Ticker")
        st.bar_chart(bar_data)


        st.subheader("Top Features by Ticker")
        for _, row in pred_df.iterrows():
            with st.expander(f"{row['ticker']} - Top 5 Most Important Features"):
                st.markdown(row["top_features"])


        st.subheader("Options Strategy Recommendations")


        for _, row in pred_df.iterrows():
            strategy, sentiment = suggest_options_strategy(
                row["pred_next_ret"],
                row.get("put_call_oi_ratio"),
                row.get("atm_iv"),
                horizon=display_horizon,
            )


            color = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}[sentiment]


            warnings = []
            ticker_screener_data = (
                st.session_state.screener_df[
                    st.session_state.screener_df["ticker"] == row["ticker"]
                ]
                if "screener_df" in st.session_state
                else pd.DataFrame()
            )
            if not ticker_screener_data.empty:
                days_to_earnings = ticker_screener_data.iloc[0].get("days_to_earnings")
                if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
                    warnings.append(f"⚠️ Earnings in {days_to_earnings} days")


            if row.get("atm_iv") and row["atm_iv"] > 0.6:
                warnings.append("⚠️ Very high IV (60%+) - event expected")


            if not ticker_screener_data.empty:
                vol_spike = ticker_screener_data.iloc[0].get("volume_spike")
                if vol_spike and vol_spike > 3.0:
                    warnings.append(f"⚠️ Volume spike {vol_spike:.1f}x - unusual activity")


            if row.get("signal_alignment") == "disagree":
                warnings.append("⚠️ Model and options market disagree")


            title = f"{color} {row['ticker']} - Options Strategy ({display_horizon_label})"
            if warnings:
                title += " ⚠️"


            with st.expander(title):
                if warnings:
                    for warning in warnings:
                        st.warning(warning)


                st.write(f"**{display_horizon_label} Prediction:** {row['pred_next_ret']*100:.2f}%")
                
                # ⭐ NEW: Show feature optimization info
                st.write(f"**Features Used:** {row['num_features']}/60")


                prob_up = row.get("prob_up")
                if prob_up is not None:
                    st.write(f"**Prob Up Move (RF/XGB):** {prob_up*100:.1f}%")
                else:
                    st.write("**Prob Up Move (RF/XGB):** N/A")


                prob_up_gaf = row.get("prob_up_gaf")
                if prob_up_gaf is not None:
                    st.write(f"**Prob Up Move (GAF-CNN):** {prob_up_gaf*100:.1f}%")
                else:
                    st.write("**Prob Up Move (GAF-CNN):** N/A")


                st.write(
                    f"**Put/Call Ratio:** {row.get('put_call_oi_ratio', 'N/A'):.3f}"
                    if row.get("put_call_oi_ratio")
                    else "**Put/Call Ratio:** N/A"
                )
                st.write(
                    f"**IV:** {row.get('atm_iv', 'N/A'):.3f}"
                    if row.get("atm_iv")
                    else "**IV:** N/A"
                )
                st.write(f"**Strategy:** {strategy}")


                if row.get("atm_iv"):
                    expected_move = row["last_close"] * row["atm_iv"] * np.sqrt(
                        display_horizon / 252
                    )
                    st.write(
                        f"**Expected {display_horizon_label} move:** ±${expected_move:.2f}"
                    )
                    st.write(
                        f"**Target strikes:** ${row['last_close'] - expected_move:.2f} "
                        f"to ${row['last_close'] + expected_move:.2f}"
                    )


                theo_price = row.get("theo_atm_call_price")
                if theo_price is not None:
                    st.write(
                        f"**Theoretical ATM call price ({pricing_model_label}):** "
                        f"${theo_price:.2f}"
                    )
                else:
                    st.write(
                        f"**Theoretical ATM call price ({pricing_model_label}):** N/A "
                        f"(no IV/expiry/params)"
                    )


                iv_gap = row.get("iv_minus_realized")
                if iv_gap is not None:
                    st.write(f"**IV - 20D realized vol:** {iv_gap:.3f}")


                try:
                    greeks_info = get_atm_greeks(row["ticker"])
                except YFRateLimitError:
                    greeks_info = None


                if greeks_info:
                    cg = greeks_info["call_greeks"]
                    pg = greeks_info["put_greeks"]
                    st.markdown("**ATM Greeks (nearest expiry):**")
                    st.write(
                        f"Call Δ: {cg['delta']:.2f}, Γ: {cg['gamma']:.4f}, "
                        f"Vega: {cg['vega']:.2f}, Θ: {cg['theta']:.2f}"
                    )
                    st.write(
                        f"Put  Δ: {pg['delta']:.2f}, Γ: {pg['gamma']:.4f}, "
                        f"Vega: {pg['vega']:.2f}, Θ: {pg['theta']:.2f}"
                    )


                    cm = greeks_info.get("call_mispricing")
                    pm = greeks_info.get("put_mispricing")


                    if cm is not None:
                        if cm > 0:
                            st.write(f"Call mispricing: +${cm:.2f} vs BS (rich/overvalued).")
                        elif cm < 0:
                            st.write(f"Call mispricing: -${abs(cm):.2f} vs BS (cheap/undervalued).")
                        else:
                            st.write("Call mispricing: ~$0 vs BS (fair).")


                    if pm is not None:
                        if pm > 0:
                            st.write(f"Put mispricing: +${pm:.2f} vs BS (rich/overvalued).")
                        elif pm < 0:
                            st.write(f"Put mispricing: -${abs(pm):.2f} vs BS (cheap/undervalued).")
                        else:
                            st.write("Put mispricing: ~$0 vs BS (fair).")


                else:
                    st.write("ATM Greeks: N/A (no option data or rate-limited).")


                news = get_news_for_ticker(row["ticker"], limit=3)
                has_big_news = detect_big_news(news)
                if has_big_news:
                    st.warning("⚠️ Recent BIG news/headlines detected for this ticker.")


                if news:
                    st.markdown("**Key recent headlines:**")
                    for art in news:
                        title_h = art.get("title", "No title")
                        src = art.get("source", "Unknown")
                        url = art.get("url")
                        sent = art.get("sentiment")
                        sent_label = (
                            f" (sentiment: {sent:.2f})"
                            if isinstance(sent, (int, float))
                            else ""
                        )
                        if url:
                            st.markdown(f"- [{title_h}]({url}) — {src}{sent_label}")
                        else:
                            st.markdown(f"- {title_h} — {src}{sent_label}")
                else:
                    st.markdown(
                        "**Key recent headlines:** none available or API not configured."
                    )


        # Model Accuracy Testing
        st.subheader("Model Accuracy Testing")
        test_ticker = st.selectbox("Test prediction accuracy for:", display["Ticker"])


        if st.button("Run Accuracy Test"):
            with st.spinner(f"Testing {test_ticker} {display_horizon_label} predictions..."):
                try:
                    results_test, accuracy = track_predictions(
                        test_ticker,
                        period="5y",
                        model_type=model_type,
                        horizon=display_horizon,
                    )


                    if not results_test.empty:
                        num_test_days = len(results_test)
                        st.metric(
                            f"Direction Accuracy (Last {num_test_days} Days, {display_horizon_label} Horizon)",
                            f"{accuracy*100:.1f}%",
                        )


                        baseline_returns = results_test["actual_return"].dropna()


                        conf_thresh = 0.01
                        strat = results_test.copy()
                        strat["position"] = np.where(
                            strat["predicted_return"] > conf_thresh,
                            1.0,
                            0.0,
                        )
                        strat["strategy_ret_no_cost"] = strat["actual_return"] * strat["position"]


                        cost_per_trade = 0.001
                        strat["position_change"] = strat["position"].diff().abs().fillna(0.0)
                        strat["strategy_ret_with_cost"] = (
                            strat["actual_return"] * strat["position"]
                            - cost_per_trade * strat["position_change"]
                        )


                        sharpe_baseline = compute_sharpe(baseline_returns)
                        sharpe_signal_no_cost = compute_sharpe(
                            strat["strategy_ret_no_cost"].dropna()
                        )
                        sharpe_signal_with_cost = compute_sharpe(
                            strat["strategy_ret_with_cost"].dropna()
                        )


                        dsr_baseline = deflated_sharpe_ratio(baseline_returns, n_trials)
                        dsr_signal_with_cost = deflated_sharpe_ratio(
                            strat["strategy_ret_with_cost"], n_trials
                        )


                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            f"Sharpe (Always Long, {display_horizon_label})",
                            "N/A" if sharpe_baseline is None else f"{sharpe_baseline:.2f}",
                        )
                        col2.metric(
                            f"Sharpe (Signal, no cost, {display_horizon_label})",
                            "N/A" if sharpe_signal_no_cost is None else f"{sharpe_signal_no_cost:.2f}",
                        )
                        col3.metric(
                            f"Sharpe (Signal, with cost, {display_horizon_label})",
                            "N/A" if sharpe_signal_with_cost is None else f"{sharpe_signal_with_cost:.2f}",
                        )


                        st.write(
                            f"**DSR (Always Long, {display_horizon_label}):** "
                            f"{'N/A' if dsr_baseline is None else f'{dsr_baseline:.2f}'} "
                            f"(using ~{n_trials} trials)"
                        )
                        st.write(
                            f"**DSR (Signal, with cost, {display_horizon_label}):** "
                            f"{'N/A' if dsr_signal_with_cost is None else f'{dsr_signal_with_cost:.2f}'} "
                            f"(using ~{n_trials} trials)"
                        )


                        if sq is not None:
                            try:
                                sq_report = sq.performance_summary(
                                    strat["strategy_ret_with_cost"].dropna(),
                                    benchmark=baseline_returns.loc[
                                        strat["strategy_ret_with_cost"].dropna().index
                                    ],
                                )
                                st.subheader("SquareQuant Performance Summary")
                                st.dataframe(sq_report)
                            except Exception as e:
                                st.write(f"SquareQuant analysis error: {e}")


                        display_results = results_test[
                            [
                                "date",
                                "predicted_return",
                                "actual_return",
                                "predicted_price",
                                "actual_close",
                                "correct_direction",
                            ]
                        ].copy()
                        display_results["predicted_return"] *= 100
                        display_results["actual_return"] *= 100


                        display_results.rename(
                            columns={
                                "date": "Date",
                                "predicted_return": f"Pred {display_horizon_label} Return (%)",
                                "actual_return": f"Actual {display_horizon_label} Return (%)",
                                "predicted_price": "Pred Price",
                                "actual_close": "Actual Price",
                                "correct_direction": "Correct?",
                            },
                            inplace=True,
                        )


                        st.dataframe(display_results)


                        chart_df = pd.DataFrame(
                            {
                                "Predicted": results_test["predicted_return"].values * 100,
                                "Actual": results_test["actual_return"].values * 100,
                            },
                            index=results_test["date"],
                        )
                        st.line_chart(chart_df)
                    else:
                        st.warning("Not enough data to test accuracy.")
                except Exception as e:
                    st.error(f"Error testing accuracy: {e}")


        st.subheader("Feature Significance (OLS, p-values)")
        fs_ticker = st.selectbox(
            "Run feature significance for:",
            display["Ticker"],
            key="fs_ticker_select",
        )
        if st.button("Analyze Feature Significance"):
            with st.spinner(f"Running OLS feature significance for {fs_ticker} ({display_horizon_label})..."):
                try:
                    ols_model, sig_df = analyze_feature_significance(
                        ticker=fs_ticker,
                        period="5y",
                        horizon=display_horizon,
                        use_vol_scaled_target=False,
                    )
                    st.write("Top features by lowest p-value (most significant first):")
                    st.dataframe(sig_df.head(25))
                except Exception as e:
                    st.error(f"Error computing feature significance: {e}")


        chosen = st.selectbox(
            "Show price history for:", display["Ticker"], key="price_history_selector"
        )
        hist = get_history_cached(chosen, period="3mo", interval="1d")
        prices = hist["Close"].copy()
        if not prices.empty:
            last_date = prices.index[-1]
            row = pred_df[pred_df["ticker"] == chosen].iloc[0]
            pred_price = row["pred_next_price"]


            extra_point = pd.Series(
                [pred_price],
                index=[last_date + pd.Timedelta(days=display_horizon)],
            )
            future = pd.concat([prices, extra_point])


            st.subheader(f"{chosen} recent prices + predicted {display_horizon_label} price")
            st.line_chart(future)
        else:
            st.warning(f"No recent price data for {chosen}.")


        if not hist.empty:
            rets = hist["Close"].pct_change()
            fig_gaf, ax_gaf = make_gaf_image_from_returns(rets, window=60, image_size=30)


            with st.expander(f"{chosen} Gramian Angular Field (GAF) Heatmap", expanded=False):
                if fig_gaf is not None:
                    st.pyplot(fig_gaf)
                else:
                    st.write("Not enough data to build GAF image.")


        st.subheader(f"{chosen} multi-horizon predictions (1–5 days)")


        multi_rows = []
        for h in [1, 2, 3, 4, 5]:
            try:
                out_h = predict_next_for_ticker(
                    chosen,
                    period="5y",
                    model_type=model_type,
                    horizon=h,
                    use_vol_scaled_target=False,
                    auto_optimize=is_auto_optimized,  # ← Use same setting
                )


                mc_res = {}
                atm_iv_h = out_h.get("atm_iv")
                last_close_h = out_h.get("last_close")
                if atm_iv_h is not None and last_close_h is not None:
                    try:
                        mc_res = option_mc_ev(
                            s0=float(last_close_h),
                            mu=float(out_h["pred_next_ret"]),
                            sigma=float(atm_iv_h),
                            days=int(h),
                            premium=1.0,
                            strike=float(last_close_h),
                            n_paths=5000,
                            is_call=True,
                        )
                    except Exception as mc_e:
                        print(f"MC error (multi) for {chosen}, h={h}: {mc_e}")
                        mc_res = {}


                multi_rows.append(
                    {
                        "Horizon (days)": h,
                        "Predicted Return (%)": out_h["pred_next_ret"] * 100,
                        "Predicted Price": out_h["pred_next_price"],
                        "Features Used": out_h["num_features"],  # ← Show features
                        "MC EV (P/L)": mc_res.get("mc_ev"),
                        "MC POP (>0)": mc_res.get("mc_pop_gt0"),
                    }
                )
            except YFRateLimitError:
                st.warning(
                    "Yahoo Finance rate limited multi-horizon predictions. "
                    "Try again later or use fewer tickers."
                )
                break
            except Exception as e:
                multi_rows.append(
                    {
                        "Horizon (days)": h,
                        "Predicted Return (%)": None,
                        "Predicted Price": None,
                        "Features Used": None,
                        "MC EV (P/L)": None,
                        "MC POP (>0)": None,
                    }
                )


        if multi_rows:
            mh_df = pd.DataFrame(multi_rows)
            st.dataframe(mh_df)



if __name__ == "__main__":
    run_app()
