import os, sys, time, json, tempfile, subprocess
from pathlib import Path
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import numpy as np

try:
    from sklearn.linear_model import ElasticNetCV
except Exception:
    ElasticNetCV = None

from prediction_model import (
    predict_next_for_ticker,
    track_predictions,
    backtest_one_ticker,
    backtest_compare_one_ticker,
    walk_forward_backtest,
    analyze_feature_significance,
    make_gaf_image_from_returns,
    walkforward_cross_sectional,
)

from stock_screener import screen_stocks
from data_fetch import get_history_cached, get_history_intraday_cached, get_option_snapshot_features, get_news_for_ticker, get_atm_greeks
from yfinance.exceptions import YFRateLimitError
from monte_carlo_pricer import option_mc_ev
from scipy.stats import norm
from option_pricing import OptionSpec, HestonParams, PricingModel, price_option

try:
    import squarequant as sq
except ImportError:
    sq = None


BASE_DIR = Path(__file__).resolve().parent
SIGNALS_OUT_PATH = BASE_DIR / "signals.json"
TRADER_PATH = BASE_DIR / "auto_paper_trade.py"

def write_signals_json_atomic(signals: dict, path: str = "signals.json"):
    fd, tmp_path = tempfile.mkstemp(prefix="signals_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(signals, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

# Make FRED_API_KEY available to prediction_model via environment variable
if "FRED_API_KEY" in st.secrets:
    os.environ["FRED_API_KEY"] = st.secrets["FRED_API_KEY"]
    print(f"[DEBUG] FRED_API_KEY set: {os.environ['FRED_API_KEY'][:8]}...")
else:
    print("[DEBUG] FRED_API_KEY NOT in secrets")

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
    mu, sigma, T = excess.mean(), excess.std(), len(r)
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
    return (daily_excess.mean() / daily_excess.std()) * np.sqrt(periods_per_year)

@dataclass
class ExecutionModel:
    delay_days: int = 1
    half_spread_bps: float = 2.0
    slippage_bps: float = 3.0
    fee_bps: float = 0.0

def apply_latency_delay(df: pd.DataFrame, delay_days: int, pred_col: str = "predicted_return") -> pd.DataFrame:
    out = df.copy()
    if delay_days and delay_days > 0:
        out[pred_col] = out[pred_col].shift(delay_days)
    return out

def apply_costs_on_trades(
    strat_df: pd.DataFrame,
    exec_model: ExecutionModel,
    actual_ret_col: str = "actual_return",
    pos_col: str = "position",
) -> pd.Series:
    pos = strat_df[pos_col].fillna(0.0)
    pos_change = pos.diff().abs().fillna(0.0)
    base = strat_df[actual_ret_col].fillna(0.0) * pos
    per_trade_cost = (exec_model.half_spread_bps + exec_model.slippage_bps + exec_model.fee_bps) / 10000.0
    costs = per_trade_cost * pos_change
    return base - costs

def detect_big_news(articles, sent_thresh: float = 0.5) -> bool:
    if not articles:
        return False
    keywords = [
        "earnings","guidance","downgrade","upgrade","lawsuit","investigation",
        "merger","acquisition","bankruptcy","sec charges","fraud","buyback",
    ]
    for art in articles:
        title = (art.get("title") or "").lower()
        sent = art.get("sentiment")
        if any(k in title for k in keywords):
            return True
        if isinstance(sent, (int, float)) and abs(sent) >= sent_thresh:
            return True
    return False

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
    threshold_multiplier = {1:1.0, 2:1.4, 3:1.7, 4:2.0, 5:2.3}.get(horizon, 1.0)
    adjusted_threshold = 1.0 * threshold_multiplier
    if abs(pred_pct) > adjusted_threshold:
        if pred_pct > 0:
            if put_call_ratio and put_call_ratio > 1.2:
                return "🚀 BULLISH: Buy Calls (high put OI suggests potential short squeeze)", "bullish"
            return "📈 BULLISH: Buy Calls or Bull Call Spread", "bullish"
        if put_call_ratio and put_call_ratio < 0.8:
            return "📉 BEARISH: Buy Puts (low protection in market)", "bearish"
        return "🔻 BEARISH: Buy Puts or Bear Put Spread", "bearish"
    if abs(pred_pct) < (0.5 * threshold_multiplier) and atm_iv and atm_iv > 0.35:
        return "⚖️ NEUTRAL: Sell Iron Condor or Straddle (high IV)", "neutral"
    return "⏸️ NEUTRAL: Wait for clearer signal or diagonal spread", "neutral"

def normalize_model_option_strategy(text: str, prefer_spreads: bool) -> str | None:
    s = (text or "").lower()
    if "bullish" in s and "call" in s and "spread" in s:
        return "BULL_CALL_SPREAD" if prefer_spreads else "BUY_CALL"
    if "bullish" in s and "buy call" in s:
        return "BUY_CALL"
    if "bullish" in s and "buy calls" in s:
        return "BUY_CALL"
    if "bearish" in s and "put" in s and "spread" in s:
        return "BEAR_PUT_SPREAD" if prefer_spreads else "BUY_PUT"
    if "bearish" in s and "buy put" in s:
        return "BUY_PUT"
    if "bearish" in s and "buy puts" in s:
        return "BUY_PUT"
    if "iron condor" in s:
        return "IRON_CONDOR"
    return None

def _build_display_df(pred_df: pd.DataFrame, display_horizon: int):
    display_horizon_label = {1:"1-Day",2:"2-Day",3:"3-Day",4:"4-Day",5:"5-Day"}[display_horizon]
    cols_to_show = [
        "ticker","model_type","horizon","last_close","vol_20d","pe_ratio","num_features",
        "atm_iv","iv_minus_realized","put_call_oi_ratio","pred_next_ret_pct","pred_next_price",
        "prob_up","prob_down","prob_up_gaf","opt_exp","theo_atm_call_price","signal_alignment",
    ]
    for mc_col in ["mc_ev","mc_pop_gt0"]:
        if mc_col in pred_df.columns:
            cols_to_show.append(mc_col)
    display = pred_df[cols_to_show].copy()
    rename_map = {
        "ticker":"Ticker","model_type":"Model","horizon":"Days Ahead","last_close":"Last Close","liveprice":"Live Price","livets":"Live As Of","vol_20d":"Vol 20D",
        "pe_ratio":"P/E","num_features":"# Features Used","atm_iv":"ATM IV","iv_minus_realized":"IV - Realized Vol",
        "put_call_oi_ratio":"Put/Call OI Ratio","pred_next_ret_pct":f"Predicted {display_horizon_label} Return (%)",
        "pred_next_price":"Predicted Price","prob_up":"Prob Up","prob_down":"Prob Down","prob_up_gaf":"GAF-CNN Prob Up",
        "opt_exp":"Opt Expiry","theo_atm_call_price":"Theo ATM Call","signal_alignment":"Signal",
        "mc_ev":"MC EV (P/L)","mc_pop_gt0":"MC POP (>0)",
    }
    display.rename(columns=rename_map, inplace=True)
    return display

def run_app():
    st.set_page_config(page_title="Stock Predictor", layout="wide")
    st.title("Stock Predictor Dashboard")

    # session state
    st.session_state.setdefault("pred_df", None)
    st.session_state.setdefault("model_type", "rf")
    st.session_state.setdefault("screener_df", None)
    st.session_state.setdefault("prediction_horizon", 5)
    st.session_state.setdefault("auto_optimize", True)
    st.session_state.setdefault("last_signals", None)
    st.session_state.setdefault("last_trader_stdout", "")
    st.session_state.setdefault("last_trader_stderr", "")
    st.session_state.setdefault("last_trader_rc", None)

    tab_pred, tab_acc, tab_backtest, tab_comp, tab_wf, tab_wfx = st.tabs([
        "📈 Predictions & Options", "✅ Accuracy", "📊 Backtest", "🔬 Comprehensive Test", "🚀 Walk-Forward", "Portfolio WF",
    ])

    # ===================== SIDEBAR (clean) =====================
    st.sidebar.header("Controls")

    with st.sidebar.expander("Core", expanded=True):
        watchlist_text = st.text_input("Tickers (comma-separated)", value="AAPL, NVDA")
        tickers = [t.strip().upper() for t in watchlist_text.split(",") if t.strip()]

        prediction_horizon = st.selectbox("Horizon (days)", [1,2,3,4,5], index=4)
        horizon_label = {1:"1-Day",2:"2-Day",3:"3-Day",4:"4-Day",5:"5-Day"}[prediction_horizon]

        model_label = st.selectbox("Model", ["Auto","Random Forest","Gradient Boosting","XGBoost"])
        if model_label == "Auto":
            model_type = "xgb" if prediction_horizon == 1 else "rf"
        else:
            model_type = {"Random Forest":"rf","Gradient Boosting":"gbrt","XGBoost":"xgb"}[model_label]

        auto_optimize = st.checkbox("Auto-optimize features", value=True)
        pricing_model_label = st.selectbox("Pricing engine", ["Black-Scholes","Heston (stochastic vol)"], index=0)
        pricing_model = PricingModel.BLACK_SCHOLES if pricing_model_label == "Black-Scholes" else PricingModel.HESTON
        

    with st.sidebar.expander("Filters", expanded=False):
        max_tickers = st.slider("Max tickers per run", 1, 20, 5)
        ret_thresh = st.slider("Min |recent return| (%)", 0.0, 10.0, 3.0, 0.5)
        vol_spike_thresh = st.slider("Min volume spike (× avg)", 0.5, 5.0, 1.5, 0.1)

        st.markdown("Candidate filters")
        min_move = st.slider("Min |predicted return| (%)", 0.0, 5.0, 1.0, 0.1)
        min_iv = st.slider("Min ATM IV", 0.0, 1.0, 0.2, 0.05)
        max_iv = st.slider("Max ATM IV", 0.0, 1.0, 0.8, 0.05)
        exclude_disagree = st.checkbox("Exclude 'disagree' signals", value=True)

    with st.sidebar.expander("Advanced", expanded=False):
        st.markdown("Elastic Net feature selection")
        use_elasticnet_select = st.checkbox("Enable Elastic Net selection", value=False)
        en_l1_ratio = st.slider("l1_ratio", 0.0, 1.0, 0.5, 0.05)
        en_cv_folds = st.slider("CV folds", 3, 8, 5, 1)
        if use_elasticnet_select and ElasticNetCV is None:
            st.error("Elastic Net requires scikit-learn (ElasticNetCV not available).")
        os.environ["USE_ELASTICNET_SELECT"] = "1" if use_elasticnet_select else "0"
        os.environ["ELASTICNET_L1_RATIO"] = str(en_l1_ratio)
        os.environ["ELASTICNET_CV_FOLDS"] = str(en_cv_folds)

        st.markdown("DSR / overfitting")
        n_trials = st.slider("Approx. # strategy variants tried", 1, 100, 20)

        st.markdown("Backtest execution (frictions)")
        bt_delay_days = st.selectbox("Execution delay (days)", [0,1,2], index=1)
        bt_half_spread_bps = st.slider("Half-spread (bps)", 0.0, 20.0, 2.0, 0.5)
        bt_slippage_bps = st.slider("Slippage (bps)", 0.0, 30.0, 3.0, 0.5)
        bt_fee_bps = st.slider("Extra fees (bps)", 0.0, 10.0, 0.0, 0.5)
        exec_model = ExecutionModel(
            delay_days=int(bt_delay_days),
            half_spread_bps=float(bt_half_spread_bps),
            slippage_bps=float(bt_slippage_bps),
            fee_bps=float(bt_fee_bps),
        )

        st.markdown("Auto-trader (options)")
        trade_mode = st.selectbox("Trade mode", ["Stocks only","Options if suggested","Options only"], index=1)
        dte_min = st.slider("Min DTE (days)", 0, 30, 0, 1)
        dte_max = st.slider("Max DTE (days)", 1, 180, 45, 1)
        max_strike = st.slider("Max strike", 50, 1000, 500, 10)
        max_premium = st.slider("Max premium ($/contract)", 50, 2000, 500, 50)
        width_pct = st.slider("Spread width (%)", 1, 20, 5, 1) / 100.0
        prefer_spreads = st.checkbox("Prefer spreads", value=True)
        auto_run_trader = st.checkbox("Auto-run trader after signals.json", value=False)
        if dte_max < dte_min:
            st.error("Max DTE must be >= Min DTE")
        run_gaf = st.sidebar.checkbox("Run GAF-CNN (slow)", value=False)


    # ===================== TAB 1: Predictions & Options =====================
    with tab_pred:
        st.caption("Run the screener + model, then use expanders for deeper diagnostics.")
        run_clicked = st.button("Run Screener + Model", type="primary")

        if run_clicked:
            if not tickers:
                st.error("Please enter at least one ticker.")
                st.stop()

            with st.spinner("Running screener..."):
                screener_df = screen_stocks(
                    tickers,
                    ret_thresh=ret_thresh / 100.0,
                    vol_spike_thresh=vol_spike_thresh,
                )
            with st.expander("Screener results (raw)", expanded=False):
                st.dataframe(screener_df, use_container_width=True)

            if screener_df.empty:
                st.warning("No data returned for these tickers.")
                st.stop()

            if "flag" in screener_df.columns:
                flagged = screener_df[screener_df["flag"] == True]["ticker"].tolist()
            else:
                flagged = []
            if not flagged:
                flagged = tickers
            if len(flagged) > max_tickers:
                st.warning(f"Limiting to first {max_tickers} tickers to avoid rate limits.")
                flagged = flagged[:max_tickers]

            st.info(f"Running {horizon_label} predictions on: {', '.join(flagged)}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for i, tk in enumerate(flagged):
                progress_bar.progress((i + 1) / len(flagged))
                status_text.text(f"Processing {tk}... ({i+1}/{len(flagged)})")
                if i > 0:
                    time.sleep(1)

                try:
                    out = predict_next_for_ticker(
                        tk,
                        period="5y",
                        model_type=model_type,
                        horizon=prediction_horizon,
                        use_vol_scaled_target=False,
                        auto_optimize=auto_optimize,
                        run_gaf=run_gaf
                    )
                    opt = get_option_snapshot_features(tk)
                    out.update(opt)

                    atm_iv = out.get("atm_iv")
                    last_close = out.get("last_close")
                    live_price = None
                    live_ts = None
                    try:
                        intraday = get_history_intraday_cached(tk, period="1d", interval="1m")
                        if intraday is not None and (not intraday.empty) and ("Close" in intraday.columns):
                            live_price = float(intraday["Close"].iloc[-1])
                            live_ts = intraday.index[-1]
                    except Exception:
                        live_price = None
                        live_ts = None
                    out["liveprice"] = live_price
                    out["livets"] = str(live_ts) if live_ts is not None else None
                        

                    out["iv_minus_realized"] = None
                    if atm_iv is not None and out.get("vol_20d") is not None:
                        try:
                            out["iv_minus_realized"] = float(atm_iv) - float(out["vol_20d"])
                        except Exception:
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

                    out["signal_alignment"] = classify_alignment(out["pred_next_ret"], out.get("put_call_oi_ratio"))
                    results.append(out)

                except YFRateLimitError:
                    st.error("Yahoo Finance rate limiting. Try later and/or use fewer tickers.")
                    break
                except Exception as e:
                    st.warning(f"{tk}: ERROR {e}")

            progress_bar.empty()
            status_text.empty()

            if not results:
                st.warning("No predictions generated.")
                st.session_state.pred_df = None
                st.stop()

            st.session_state.pred_df = pd.DataFrame(results)
            st.session_state.pred_df["pred_next_ret_pct"] = st.session_state.pred_df["pred_next_ret"] * 100
            st.session_state.model_type = model_type
            st.session_state.screener_df = screener_df
            st.session_state.prediction_horizon = prediction_horizon
            st.session_state.auto_optimize = auto_optimize

            signals = {}
            for _, row in st.session_state.pred_df.iterrows():
                tk = str(row["ticker"]).upper()
                pred = float(row["pred_next_ret"])
                stock_action = "BUY" if pred >= 0.005 else ("SELL" if pred <= -0.005 else "HOLD")

                strat_text, _bias = suggest_options_strategy(
                    pred_ret=pred,
                    put_call_ratio=row.get("put_call_oi_ratio"),
                    atm_iv=row.get("atm_iv"),
                    horizon=prediction_horizon,
                )
                strategy = normalize_model_option_strategy(strat_text, prefer_spreads=prefer_spreads)
                use_options = (trade_mode == "Options only") or (trade_mode == "Options if suggested" and strategy is not None)

                if use_options and strategy is not None:
                    signals[tk] = {
                        "asset": "option",
                        "strategy": strategy,
                        "dte_min": int(dte_min),
                        "dte_max": int(dte_max),
                        "max_strike": float(max_strike),
                        "max_premium": float(max_premium),
                        "width_pct": float(width_pct),
                        "qty": 1,
                        "raw_strategy_text": str(strat_text),
                        "pred_next_ret": float(pred),
                        "last_close": float(row.get("last_close")) if row.get("last_close") is not None else None,
                        "execution": {
                            "delay_days": int(exec_model.delay_days),
                            "half_spread_bps": float(exec_model.half_spread_bps),
                            "slippage_bps": float(exec_model.slippage_bps),
                            "fee_bps": float(exec_model.fee_bps),
                        },
                    }
                else:
                    signals[tk] = {
                        "asset": "stock",
                        "action": stock_action,
                        "qty": 1,
                        "pred_next_ret": float(pred),
                        "execution": {
                            "delay_days": int(exec_model.delay_days),
                            "half_spread_bps": float(exec_model.half_spread_bps),
                            "slippage_bps": float(exec_model.slippage_bps),
                            "fee_bps": float(exec_model.fee_bps),
                        },
                    }

            write_signals_json_atomic(signals, str(SIGNALS_OUT_PATH))
            st.session_state.last_signals = signals
            st.success(f"Wrote signals.json to: {SIGNALS_OUT_PATH}")

            st.session_state.last_trader_stdout = ""
            st.session_state.last_trader_stderr = ""
            st.session_state.last_trader_rc = None
            if auto_run_trader:
                if not TRADER_PATH.exists():
                    st.error(f"Trader script not found: {TRADER_PATH}")
                else:
                    res = subprocess.run(
                        [sys.executable, str(TRADER_PATH)],
                        cwd=str(BASE_DIR),
                        capture_output=True,
                        text=True,
                    )
                    st.session_state.last_trader_stdout = res.stdout or ""
                    st.session_state.last_trader_stderr = res.stderr or ""
                    st.session_state.last_trader_rc = res.returncode

        # IMPORTANT: everything below is inside tab_pred so it never leaks to other tabs
        if st.session_state.pred_df is not None:
            pred_df = st.session_state.pred_df
            model_type = st.session_state.model_type
            display_horizon = st.session_state.get("prediction_horizon", 1)
            display_horizon_label = {1:"1-Day",2:"2-Day",3:"3-Day",4:"4-Day",5:"5-Day"}[display_horizon]
            is_auto_optimized = st.session_state.get("auto_optimize", True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tickers processed", len(pred_df))
            c2.metric("Horizon", display_horizon_label)
            c3.metric("Model", model_type.upper())
            c4.metric("Auto-optimize", "ON" if is_auto_optimized else "OFF")
            st.caption("Auto-optimization ON: using optimized features per stock." if is_auto_optimized else "Auto-optimization OFF: using all features.")

            st.subheader("Auto-trade preview")
            if st.session_state.last_signals:
                sig_rows = []
                for tk, s in st.session_state.last_signals.items():
                    if s.get("asset") == "option":
                        sig_rows.append({
                            "Ticker": tk, "Asset": "option", "Strategy": s.get("strategy"),
                            "DTE max": s.get("dte_max"), "Max premium ($)": s.get("max_premium"),
                            "Pred next ret (%)": round(float(s.get("pred_next_ret", 0.0)) * 100, 2),
                            "Last close": s.get("last_close"),
                        })
                    else:
                        sig_rows.append({
                            "Ticker": tk, "Asset": "stock", "Action": s.get("action"),
                            "Pred next ret (%)": round(float(s.get("pred_next_ret", 0.0)) * 100, 2),
                        })
                st.dataframe(pd.DataFrame(sig_rows), use_container_width=True)
            else:
                st.info("No signals written yet in this session.")

            cand_df = pred_df.copy()
            cand_df["abs_pred_pct"] = cand_df["pred_next_ret_pct"].abs()
            mask = (cand_df["abs_pred_pct"] >= min_move) & cand_df["atm_iv"].between(min_iv, max_iv)
            if exclude_disagree:
                mask &= cand_df["signal_alignment"] != "disagree"
            cand_df = cand_df[mask]

            st.subheader("Top model candidates (filtered)")
            if not cand_df.empty:
                cand_df["score"] = cand_df["abs_pred_pct"]
                cand_df = cand_df.sort_values("score", ascending=False)
                st.dataframe(
                    cand_df[[
                        "ticker","pred_next_ret_pct","pred_next_price","num_features","atm_iv",
                        "put_call_oi_ratio","signal_alignment","prob_up","prob_up_gaf",
                    ]].rename(columns={
                        "ticker":"Ticker","num_features":"Features","prob_up":"Prob Up","prob_up_gaf":"GAF-CNN Prob Up",
                    }),
                    use_container_width=True,
                )
            else:
                st.write("No strong candidates today based on current filters.")

            with st.expander("Full predictions table (all columns)", expanded=False):
                display = _build_display_df(pred_df, display_horizon)
                st.dataframe(display, use_container_width=True)
                bar_data = display.set_index("Ticker")[f"Predicted {display_horizon_label} Return (%)"]
                st.subheader(f"Predicted {display_horizon_label} returns by ticker")
                st.bar_chart(bar_data)

            with st.expander("Options strategy details (per ticker)", expanded=False):
                for _, row in pred_df.iterrows():
                    strategy, sentiment = suggest_options_strategy(
                        row["pred_next_ret"],
                        row.get("put_call_oi_ratio"),
                        row.get("atm_iv"),
                        horizon=display_horizon,
                    )
                    color = {"bullish":"🟢","bearish":"🔴","neutral":"🟡"}[sentiment]
                    warnings = []

                    ticker_screener_data = (
                        st.session_state.screener_df[st.session_state.screener_df["ticker"] == row["ticker"]]
                        if "screener_df" in st.session_state and st.session_state.screener_df is not None
                        else pd.DataFrame()
                    )
                    if not ticker_screener_data.empty:
                        days_to_earnings = ticker_screener_data.iloc[0].get("days_to_earnings")
                        if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
                            warnings.append(f"⚠️ Earnings in {days_to_earnings} days")
                        vol_spike = ticker_screener_data.iloc[0].get("volume_spike")
                        if vol_spike and vol_spike > 3.0:
                            warnings.append(f"⚠️ Volume spike {vol_spike:.1f}x")
                    if row.get("atm_iv") and row["atm_iv"] > 0.6:
                        warnings.append("⚠️ Very high IV (60%+)")
                    if row.get("signal_alignment") == "disagree":
                        warnings.append("⚠️ Model and options market disagree")

                    title = f"{color} {row['ticker']} — {strategy}" + (" ⚠️" if warnings else "")
                    with st.expander(title):
                        for w in warnings:
                            st.warning(w)
                        st.write(f"{display_horizon_label} prediction: {row['pred_next_ret']*100:.2f}%")
                        st.write(f"Features used: {row['num_features']}/60")

                        prob_up = row.get("prob_up")
                        st.write(f"Prob Up Move (RF/XGB): {prob_up*100:.1f}%" if prob_up is not None else "Prob Up Move (RF/XGB): N/A")
                        prob_up_gaf = row.get("prob_up_gaf")
                        st.write(f"Prob Up Move (GAF-CNN): {prob_up_gaf*100:.1f}%" if prob_up_gaf is not None else "Prob Up Move (GAF-CNN): N/A")

                        st.write(f"Put/Call ratio: {row.get('put_call_oi_ratio'):.3f}" if row.get("put_call_oi_ratio") else "Put/Call ratio: N/A")
                        st.write(f"IV: {row.get('atm_iv'):.3f}" if row.get("atm_iv") else "IV: N/A")

                        if row.get("atm_iv"):
                            expected_move = row["last_close"] * row["atm_iv"] * np.sqrt(display_horizon / 252)
                            st.write(f"Expected {display_horizon_label} move: ±${expected_move:.2f}")
                            st.write(f"Target strikes: ${row['last_close'] - expected_move:.2f} to ${row['last_close'] + expected_move:.2f}")

                        theo_price = row.get("theo_atm_call_price")
                        st.write(f"Theoretical ATM call price ({pricing_model_label}): ${theo_price:.2f}" if theo_price is not None else f"Theoretical ATM call price ({pricing_model_label}): N/A")

                        iv_gap = row.get("iv_minus_realized")
                        if iv_gap is not None:
                            st.write(f"IV - 20D realized vol: {iv_gap:.3f}")

                        try:
                            greeks_info = get_atm_greeks(row["ticker"])
                        except YFRateLimitError:
                            greeks_info = None

                        if greeks_info:
                            cg, pg = greeks_info["call_greeks"], greeks_info["put_greeks"]
                            st.markdown("ATM Greeks (nearest expiry):")
                            st.write(f"Call Δ: {cg['delta']:.2f}, Γ: {cg['gamma']:.4f}, Vega: {cg['vega']:.2f}, Θ: {cg['theta']:.2f}")
                            st.write(f"Put  Δ: {pg['delta']:.2f}, Γ: {pg['gamma']:.4f}, Vega: {pg['vega']:.2f}, Θ: {pg['theta']:.2f}")
                        else:
                            st.write("ATM Greeks: N/A")

                        news = get_news_for_ticker(row["ticker"], limit=3)
                        if detect_big_news(news):
                            st.warning("⚠️ Recent BIG news/headlines detected.")
                        if news:
                            st.markdown("Key recent headlines:")
                            for art in news:
                                title_h = art.get("title", "No title")
                                src = art.get("source", "Unknown")
                                url = art.get("url")
                                sent = art.get("sentiment")
                                sent_label = f" (sentiment: {sent:.2f})" if isinstance(sent, (int, float)) else ""
                                st.markdown(f"- [{title_h}]({url}) — {src}{sent_label}" if url else f"- {title_h} — {src}{sent_label}")
                        else:
                            st.markdown("Key recent headlines: none available or API not configured.")

            with st.expander("Auto-trader output (stdout/stderr)", expanded=False):
                st.write(f"signals.json path: {SIGNALS_OUT_PATH}")
                if st.session_state.last_trader_rc is None:
                    st.info("Trader has not been run yet in this session (or auto-run is off).")
                else:
                    st.write(f"Return code: {st.session_state.last_trader_rc}")
                    st.code(st.session_state.last_trader_stdout or "(no stdout)", language="text")
                    if st.session_state.last_trader_rc != 0:
                        st.code(st.session_state.last_trader_stderr or "(no stderr)", language="text")

    # ===================== TAB 2: Accuracy =====================
    with tab_acc:
        st.header("✅ Model Accuracy Testing")
        st.caption("This tab stays clean; it reuses the last Predictions run.")
        if st.session_state.pred_df is None:
            st.warning("Run Predictions first so this tab can reuse tickers/model/horizon.")
            st.stop()

        pred_df = st.session_state.pred_df
        model_type = st.session_state.model_type
        display_horizon = st.session_state.get("prediction_horizon", 1)
        display_horizon_label = {1:"1-Day",2:"2-Day",3:"3-Day",4:"4-Day",5:"5-Day"}[display_horizon]
        display = _build_display_df(pred_df, display_horizon)

        test_ticker = st.selectbox("Ticker", display["Ticker"])
        if st.button("Run Accuracy Test"):
            with st.spinner(f"Testing {test_ticker} {display_horizon_label} predictions..."):
                try:
                    results_test, accuracy = track_predictions(
                        test_ticker,
                        period="5y",
                        model_type=model_type,
                        horizon=display_horizon,
                    )
                    st.session_state["results_test"] = results_test
                    st.session_state["accuracy"] = accuracy
                    if results_test.empty:
                        st.warning("Not enough data to test accuracy.")
                        st.stop()

                    num_test_days = len(results_test)
                    st.metric(f"Direction Accuracy (Last {num_test_days} Days, {display_horizon_label})", f"{accuracy*100:.1f}%")

                    baseline_returns = results_test["actual_return"].dropna()
                    results_exec = apply_latency_delay(results_test, delay_days=exec_model.delay_days, pred_col="predicted_return")

                    conf_thresh = 0.01
                    strat = results_exec.copy()
                    strat["position"] = np.where(strat["predicted_return"] > conf_thresh, 1.0, 0.0)
                    strat["strategy_ret_no_cost"] = strat["actual_return"] * strat["position"]
                    strat["strategy_ret_with_cost"] = apply_costs_on_trades(strat, exec_model)

                    sharpe_baseline = compute_sharpe(baseline_returns)
                    sharpe_signal_no_cost = compute_sharpe(strat["strategy_ret_no_cost"].dropna())
                    sharpe_signal_with_cost = compute_sharpe(strat["strategy_ret_with_cost"].dropna())

                    dsr_baseline = deflated_sharpe_ratio(baseline_returns, n_trials)
                    dsr_signal_with_cost = deflated_sharpe_ratio(strat["strategy_ret_with_cost"], n_trials)

                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"Sharpe (Always Long, {display_horizon_label})", "N/A" if sharpe_baseline is None else f"{sharpe_baseline:.2f}")
                    c2.metric(f"Sharpe (Signal, no cost)", "N/A" if sharpe_signal_no_cost is None else f"{sharpe_signal_no_cost:.2f}")
                    c3.metric(f"Sharpe (Signal, with cost)", "N/A" if sharpe_signal_with_cost is None else f"{sharpe_signal_with_cost:.2f}")

                    st.write(f"DSR (Always Long): {'N/A' if dsr_baseline is None else f'{dsr_baseline:.2f}'} (using ~{n_trials} trials)")
                    st.write(f"DSR (Signal, with cost): {'N/A' if dsr_signal_with_cost is None else f'{dsr_signal_with_cost:.2f}'} (using ~{n_trials} trials)")

                    if sq is not None:
                        try:
                            sq_report = sq.performance_summary(
                                strat["strategy_ret_with_cost"].dropna(),
                                benchmark=baseline_returns.loc[strat["strategy_ret_with_cost"].dropna().index],
                            )
                            st.subheader("SquareQuant Performance Summary")
                            st.dataframe(sq_report, use_container_width=True)
                        except Exception as e:
                            st.write(f"SquareQuant analysis error: {e}")

                    display_results = results_test[["date","predicted_return","actual_return","predicted_price","actual_close","correct_direction"]].copy()
                    display_results["predicted_return"] *= 100
                    display_results["actual_return"] *= 100
                    display_results.rename(columns={
                        "date":"Date",
                        "predicted_return":f"Pred {display_horizon_label} Return (%)",
                        "actual_return":f"Actual {display_horizon_label} Return (%)",
                        "predicted_price":"Pred Price",
                        "actual_close":"Actual Price",
                        "correct_direction":"Correct?",
                    }, inplace=True)
                    st.dataframe(display_results, use_container_width=True)

                    chart_df = pd.DataFrame(
                        {"Predicted": results_test["predicted_return"].values * 100, "Actual": results_test["actual_return"].values * 100},
                        index=results_test["date"],
                    )
                    st.line_chart(chart_df)

                except Exception as e:
                    st.error(f"Error testing accuracy: {e}")

        with st.expander("Feature Significance + Diagnostics", expanded=False):
            fs_ticker = st.selectbox("Feature significance ticker", display["Ticker"], key="fs_ticker_select")
            if st.button("Analyze Feature Significance"):
                with st.spinner(f"Running OLS feature significance for {fs_ticker} ({display_horizon_label})..."):
                    try:
                        _ols_model, sig_df = analyze_feature_significance(
                            ticker=fs_ticker, period="5y", horizon=display_horizon, use_vol_scaled_target=False
                        )
                        st.dataframe(sig_df.head(25), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error computing feature significance: {e}")

            chosen = st.selectbox("Diagnostics ticker", display["Ticker"], key="price_history_selector")
            hist = get_history_cached(chosen, period="3mo", interval="1d")
            prices = hist["Close"].copy() if not hist.empty else pd.Series(dtype=float)
            if not prices.empty:
                last_date = prices.index[-1]
                row = pred_df[pred_df["ticker"] == chosen].iloc[0]
                pred_price = row["pred_next_price"]
                extra_point = pd.Series([pred_price], index=[last_date + pd.Timedelta(days=display_horizon)])
                st.line_chart(pd.concat([prices, extra_point]))
            else:
                st.warning(f"No recent price data for {chosen}.")

            if not hist.empty:
                rets = hist["Close"].pct_change()
                fig_gaf, _ax = make_gaf_image_from_returns(rets, window=60, image_size=30)
                if fig_gaf is not None:
                    st.pyplot(fig_gaf)

            st.subheader(f"{chosen} multi-horizon predictions (1–5 days)")
            multi_rows = []
            for h in [1,2,3,4,5]:
                try:
                    out_h = predict_next_for_ticker(
                        chosen, period="5y", model_type=model_type, horizon=h,
                        use_vol_scaled_target=False, auto_optimize=st.session_state.get("auto_optimize", True), run_gaf=run_gaf,
                    )
                    mc_res = {}
                    atm_iv_h = out_h.get("atm_iv")
                    last_close_h = out_h.get("last_close")
                    if atm_iv_h is not None and last_close_h is not None:
                        try:
                            mc_res = option_mc_ev(
                                s0=float(last_close_h), mu=float(out_h["pred_next_ret"]), sigma=float(atm_iv_h),
                                days=int(h), premium=1.0, strike=float(last_close_h), n_paths=5000, is_call=True,
                            )
                        except Exception as mc_e:
                            print(f"MC error (multi) for {chosen}, h={h}: {mc_e}")
                    multi_rows.append({
                        "Horizon (days)": h,
                        "Predicted Return (%)": out_h["pred_next_ret"] * 100,
                        "Predicted Price": out_h["pred_next_price"],
                        "Features Used": out_h["num_features"],
                        "MC EV (P/L)": mc_res.get("mc_ev"),
                        "MC POP (>0)": mc_res.get("mc_pop_gt0"),
                    })
                except YFRateLimitError:
                    st.warning("Rate limited multi-horizon predictions. Try later or use fewer tickers.")
                    break
                except Exception:
                    multi_rows.append({
                        "Horizon (days)": h, "Predicted Return (%)": None, "Predicted Price": None,
                        "Features Used": None, "MC EV (P/L)": None, "MC POP (>0)": None,
                    })
            if multi_rows:
                st.dataframe(pd.DataFrame(multi_rows), use_container_width=True)

                rt = st.session_state.get("results_test")
            if rt is not None and not rt.empty:
                price_df = pd.DataFrame(
                    {
                        "Actual": rt["actual_close"].values,
                        "ML Pred": rt["predicted_price"].values,
                        "GBM Median": rt["gbm_med_price"].values,
                        "GBM P05": rt["gbm_p05_price"].values,
                        "GBM P95": rt["gbm_p95_price"].values,
                    },
                    index=rt["date"],
                )
                st.line_chart(price_df)
            else:
                st.info("Run Accuracy Test first.")

    # ===================== TAB 3: Backtest =====================
    with tab_backtest:
        st.header("📊 Single-Stock Backtest")
        st.caption("Out-of-sample test using track_predictions (plus latency/cost simulation).")

        bt_ticker = st.text_input("Ticker", "NVDA", key="backtest_ticker")
        bt_horizon = st.selectbox("Horizon (days)", [1,2,3,4,5], index=4, key="bt_horizon")
        bt_model = st.selectbox("Model", ["rf","xgb","gbrt"], index=0, key="bt_model")

        if st.button("Run Backtest", key="run_backtest"):
            with st.spinner(f"Running backtest for {bt_ticker}..."):
                try:
                    results_test, accuracy = track_predictions(bt_ticker, period="5y", model_type=bt_model, horizon=bt_horizon)
                    if results_test.empty:
                        st.warning("Not enough data to backtest.")
                        st.stop()

                    baseline_returns = results_test["actual_return"].dropna()
                    results_exec = apply_latency_delay(results_test, delay_days=exec_model.delay_days, pred_col="predicted_return")

                    conf_thresh = 0.002
                    strat = results_exec.copy()
                    strat["position"] = np.where(strat["predicted_return"] > conf_thresh, 1.0, 0.0)
                    strat["strategy_ret_no_cost"] = strat["actual_return"] * strat["position"]
                    strat["strategy_ret_with_cost"] = apply_costs_on_trades(strat, exec_model)

                    sharpe_baseline = compute_sharpe(baseline_returns)
                    sharpe_strategy_no_cost = compute_sharpe(strat["strategy_ret_no_cost"].dropna())
                    sharpe_strategy_with_cost = compute_sharpe(strat["strategy_ret_with_cost"].dropna())

                    total_return_baseline = (1 + baseline_returns).prod() - 1
                    total_return_strategy_no_cost = (1 + strat["strategy_ret_no_cost"].dropna()).prod() - 1
                    total_return_strategy_with_cost = (1 + strat["strategy_ret_with_cost"].dropna()).prod() - 1

                    num_trades = strat["position"].diff().abs().sum() / 2

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Sharpe (with cost)", "N/A" if sharpe_strategy_with_cost is None else f"{sharpe_strategy_with_cost:.2f}")
                    c2.metric("Hit Rate", f"{accuracy*100:.1f}%")
                    c3.metric("Total Return (with cost)", f"{total_return_strategy_with_cost*100:.1f}%")
                    c4.metric("Test Days", len(results_test))

                    with st.expander("Backtest details", expanded=True):
                        st.write(f"Model: {bt_model.upper()} | Horizon: {bt_horizon} days | Threshold: {conf_thresh*100:.2f}%")
                        st.write(f"Execution sim: delay_days={exec_model.delay_days}, half_spread_bps={exec_model.half_spread_bps}, slippage_bps={exec_model.slippage_bps}, fee_bps={exec_model.fee_bps}")
                        st.write(f"Trades: {int(num_trades)}")
                        st.write(f"Sharpe (no cost): {'N/A' if sharpe_strategy_no_cost is None else f'{sharpe_strategy_no_cost:.2f}'}")
                        st.write(f"Sharpe (with cost): {'N/A' if sharpe_strategy_with_cost is None else f'{sharpe_strategy_with_cost:.2f}'}")                   
                        st.write(f"Return (no cost): {total_return_strategy_no_cost*100:.1f}% | Return (with cost): {total_return_strategy_with_cost*100:.1f}%")
                        st.write(f"Buy & Hold: Sharpe={'N/A' if sharpe_baseline is None else f'{sharpe_baseline:.2f}'}, Return={total_return_baseline*100:.1f}%")
                        st.write(f"Signal threshold: {conf_thresh*100:.2f}% | Trades: {int(num_trades)}")

                        cum_baseline = (1 + baseline_returns).cumprod()
                        cum_strategy = (1 + strat["strategy_ret_with_cost"].dropna()).cumprod()
                        chart_df = pd.DataFrame(
                            {"Buy & Hold": cum_baseline.values, "Strategy (with cost)": cum_strategy.values},
                            index=results_test["date"],
                        )
                        st.subheader("Cumulative returns")
                        st.line_chart(chart_df)

                        recent = results_test.tail(25)[["date","predicted_return","actual_return","correct_direction"]].copy()
                        recent["predicted_return"] *= 100
                        recent["actual_return"] *= 100
                        recent.columns = ["Date","Predicted %","Actual %","Correct?"]
                        st.subheader("Recent predictions")
                        st.dataframe(recent, use_container_width=True)

                except Exception as e:
                    st.error(f"Error running backtest: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # ===================== TAB 4: Comprehensive Test =====================
    with tab_comp:
        st.header("🔬 Comprehensive Test")
        st.caption("Loads a precomputed CSV (or you can upload one) and shows the key winners cleanly.")

        csv_path = "backtest_results_comprehensive.csv"
        comp_results = None

        if os.path.exists(csv_path):
            try:
                comp_results = pd.read_csv(csv_path)
                st.success("Loaded backtest_results_comprehensive.csv")
            except Exception as e:
                st.error(f"Failed to read {csv_path}: {e}")

        if comp_results is None:
            up = st.file_uploader("Upload a comprehensive results CSV", type="csv")
            if up is not None:
                try:
                    comp_results = pd.read_csv(up)
                except Exception as e:
                    st.error(f"Could not read uploaded CSV: {e}")

        if comp_results is None:
            st.info("No comprehensive results loaded yet.")
            st.stop()

        # Keep this resilient to slightly different column names
        # (your original expects RF_Sharpe / RF_HitRate / RF_Return) [file:114]
        sharpe_col = "RF_Sharpe" if "RF_Sharpe" in comp_results.columns else None
        hit_col = "RF_HitRate" if "RF_HitRate" in comp_results.columns else None
        ret_col = "RF_Return" if "RF_Return" in comp_results.columns else None

        c1, c2, c3 = st.columns(3)
        c1.metric("Total tested", len(comp_results))
        if sharpe_col:
            tradeable = comp_results[comp_results[sharpe_col] > 1.0]
            elite = comp_results[comp_results[sharpe_col] > 2.0]
            c2.metric("Tradeable (Sharpe>1)", len(tradeable))
            c3.metric("Elite (Sharpe>2)", len(elite))
        else:
            c2.metric("Tradeable", "N/A")
            c3.metric("Elite", "N/A")

        with st.expander("All results", expanded=False):
            st.dataframe(comp_results, use_container_width=True)

        if sharpe_col:
            st.subheader("Top 10 (by RF Sharpe)")
            cols = ["Ticker","Category",sharpe_col]
            if hit_col: cols.append(hit_col)
            if ret_col: cols.append(ret_col)
            cols = [c for c in cols if c in comp_results.columns]
            st.dataframe(comp_results.nlargest(10, sharpe_col)[cols], use_container_width=True)

    # ===================== TAB 5: Walk-Forward =====================
    with tab_wf:
        st.header("🚀 Walk-Forward Validation")
        st.caption("Multiple train/test folds to check stability across time (no single split).")

        wf_ticker = st.text_input("Ticker", "NVDA", key="wf_ticker")
        wf_horizon = st.selectbox("Horizon (days)", [1,2,3,4,5], index=4, key="wf_horizon")
        wf_model = st.selectbox("Model", ["rf","xgb"], index=0, key="wf_model")

        # keep the same threshold you used originally
        wf_threshold = st.slider("Signal threshold", 0.0, 1.0, 0.2, 0.05) / 100.0
        wf_step_days = st.selectbox(
        "Fold stride (trading days)",
        [5, 10, 21, 63, 126],
        index=2,                 # default = 21
        key="wf_step_days",
    )

        if st.button("Run Walk-Forward", key="run_wf"):
            with st.spinner("Running walk-forward..."):
                try:
                    fold_results = walk_forward_backtest(
                        ticker=wf_ticker,
                        period="5y",
                        horizon=wf_horizon,
                        model_type=wf_model,
                        train_years=3,
                        test_years=1,
                        step_days=int(wf_step_days),
                        threshold=float(wf_threshold),
                    )
                    if not fold_results:
                        st.warning("No folds produced (not enough data or settings too strict).")
                        st.stop()

                    sharpes = [f.get("sharpe", 0.0) for f in fold_results if f.get("sharpe") is not None]
                    avg_sh = (sum(sharpes)/len(sharpes)) if sharpes else None
                    st.metric("Avg Sharpe", "N/A" if avg_sh is None else f"{avg_sh:.3f}")

                    for i, fold in enumerate(fold_results, 1):
                        sh = fold.get("sharpe")
                        hr = fold.get("hit_rate")
                        with st.expander(f"Fold {i} — Sharpe: {'N/A' if sh is None else f'{sh:.3f}'}"):
                            st.write(f"Train: {fold.get('train_start')} → {fold.get('train_end')}")
                            st.write(f"Test: {fold.get('test_start')} → {fold.get('test_end')}")
                            st.write(f"Sharpe: {'N/A' if sh is None else f'{sh:.3f}'}")
                            st.write(f"Hit rate: {'N/A' if hr is None else f'{hr*100:.1f}%'}")
                            st.write(f"Trades: {fold.get('num_trades')}")
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with tab_wfx:
        st.header("Portfolio Walk-Forward (Cross-Sectional)")
        st.caption("Build a daily portfolio across multiple tickers and test stability across folds.")

        universe_text = st.text_input(
            "Universe tickers (comma separated)",
            value="AAPL, NVDA, MSFT, AMZN, META",
            key="wfx_universe",
        )
        universe = [t.strip().upper() for t in universe_text.split(",") if t.strip()]

        wfx_horizon = st.selectbox(
            "Horizon (days ahead)",
            [1, 2, 3, 4, 5],
            index=2,
            key="wfx_horizon",
        )

        wfx_model = st.selectbox(
            "Model",
            ["rf", "xgb"],
            index=0,
            key="wfx_model",
        )

        wfx_train_years = st.selectbox(
            "Train window (years)",
            [2, 3, 4, 5],
            index=1,
            key="wfx_train_years",
        )
        wfx_test_years = st.selectbox(
            "Test window (years)",
            [1, 2],
            index=0,
            key="wfx_test_years",
        )

        col1, col2 = st.columns(2)
        with col1:
            wfx_top_pct = st.slider(
                "Top percentile long",
                min_value=0.05,
                max_value=0.3,
                value=0.1,
                step=0.05,
            )
        with col2:
            use_longshort = st.checkbox("Use long-short (also short bottom percentile)", value=False)
            wfx_bottom_pct = (
                st.slider(
                    "Bottom percentile short",
                    min_value=0.05,
                    max_value=0.3,
                    value=0.1,
                    step=0.05,
                )
                if use_longshort
                else None
            )

        if st.button("Run Portfolio Walk-Forward", key="run_wfx"):
            if not universe:
                st.error("Please enter at least one ticker in the universe.")
                st.stop()
            with st.spinner("Running portfolio walk-forward..."):
                try:
                    folds = walkforward_cross_sectional(
                        tickers=universe,
                        period="5y",
                        horizon=int(wfx_horizon),
                        modeltype=wfx_model,
                        trainyears=int(wfx_train_years),
                        testyears=int(wfx_test_years),
                        top_pct=float(wfx_top_pct),
                        bottom_pct=float(wfx_bottom_pct) if use_longshort else None,
                    )
                    if not folds:
                        st.warning("No folds produced; not enough data or settings too strict.")
                        st.stop()

                    sharpes = [f.get("sharpe") for f in folds if f.get("sharpe") is not None]
                    avg_sh = np.mean(sharpes) if sharpes else None
                    med_sh = np.median(sharpes) if sharpes else None
                    worst_sh = np.min(sharpes) if sharpes else None
                    pos_pct = np.mean([s > 1.0 for s in sharpes]) if sharpes else None

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Avg Sharpe", "NA" if avg_sh is None else f"{avg_sh:.2f}")
                    c2.metric("Median Sharpe", "NA" if med_sh is None else f"{med_sh:.2f}")
                    c3.metric("Worst Sharpe", "NA" if worst_sh is None else f"{worst_sh:.2f}")
                    c4.metric(
                        "% folds Sharpe > 1",
                        "NA" if pos_pct is None else f"{100.0 * pos_pct:.1f}",
                    )

                    df_folds = pd.DataFrame(folds)
                    st.subheader("Fold metrics")
                    st.dataframe(df_folds, use_container_width=True)

                except Exception as e:
                    st.error(f"Error running portfolio walk-forward: {e}")


if __name__ == "__main__":
    run_app()
