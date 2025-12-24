import os
import sys
import time
import json
import tempfile
import subprocess
import importlib.metadata as m
import plotly.graph_objects as go
from pathlib import Path
from dataclasses import dataclass

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import importlib.metadata as m
st.write("Plotly version:", m.version("plotly"))

try:
    from sklearn.linear_model import ElasticNetCV
except Exception:
    ElasticNetCV = None

from yfinance.exceptions import YFRateLimitError
from scipy.stats import norm

# ======= Core model entrypoints (DO NOT RENAME) =======
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

# ======= Data / options helpers (existing) =======
from stock_screener import screen_stocks
from data_fetch import (
    get_history_cached,
    get_history_intraday_cached,
    get_option_snapshot_features,
    get_news_for_ticker,
    get_atm_greeks,
)
from monte_carlo_pricer import option_mc_ev
from option_pricing import OptionSpec, HestonParams, PricingModel, price_option

try:
    import squarequant as sq
except ImportError:
    sq = None

BASE_DIR = Path(__file__).resolve().parent
SIGNALS_OUT_PATH = BASE_DIR / "signals.json"
TRADER_PATH = BASE_DIR / "auto_paper_trade.py"


# Make FRED_API_KEY available to prediction_model via environment variable
if "FRED_API_KEY" in st.secrets:
    os.environ["FRED_API_KEY"] = st.secrets["FRED_API_KEY"]


# ---------------------------
# Small utilities / contracts
# ---------------------------
def write_signals_json_atomic(signals: dict, path: str):
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

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _cached_accuracy_row(
    tk: str,
    period: str,
    model_type: str,
    horizon: int,
    delay_days: int,
    half_spread_bps: float,
    slippage_bps: float,
    fee_bps: float,
    signal_threshold_pct: float,
    n_trials: int,
):
    results_test, accuracy = _cached_track_predictions(tk, period=period, model_type=model_type, horizon=int(horizon))
    if results_test is None or results_test.empty:
        return {"ticker": tk, "direction_accuracy": None, "sharpe_signal_w_cost": None, "dsr_signal_w_cost": None, "acc_n": 0}

    exec_model = ExecutionModel(
        delay_days=int(delay_days),
        half_spread_bps=float(half_spread_bps),
        slippage_bps=float(slippage_bps),
        fee_bps=float(fee_bps),
    )

    baseline_returns = results_test["actual_return"].dropna()
    results_exec = apply_latency_delay(results_test, delay_days=exec_model.delay_days, pred_col="predicted_return")

    strat = results_exec.copy()
    strat["position"] = np.where(strat["predicted_return"] > float(signal_threshold_pct), 1.0, 0.0)
    strat["strategy_ret_with_cost"] = apply_costs_on_trades(strat, exec_model)

    sharpe_signal_w_cost = compute_sharpe(strat["strategy_ret_with_cost"].dropna())
    dsr_signal_w_cost = deflated_sharpe_ratio(strat["strategy_ret_with_cost"], n_trials=n_trials)

    return {
        "ticker": tk,
        "direction_accuracy": float(accuracy),
        "sharpe_signal_w_cost": None if sharpe_signal_w_cost is None else float(sharpe_signal_w_cost),
        "dsr_signal_w_cost": None if dsr_signal_w_cost is None else float(dsr_signal_w_cost),
        "acc_n": int(len(results_test)),
    }

def _fmt_num(x, decimals=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"

def _fmt_pct(x, decimals=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x) * 100:.{decimals}f}%"
    except Exception:
        return "—"


def enrich_signals_in_place(signals: dict, pred_df: pd.DataFrame, *, context: dict):
    if not signals or pred_df is None or pred_df.empty:
        return signals
    by_ticker = {str(r["ticker"]).upper(): r for _, r in pred_df.iterrows() if "ticker" in pred_df.columns}
    for tk, sig in signals.items():
        r = by_ticker.get(str(tk).upper(), {})
        sig["context"] = dict(context)

        # attach accuracy if available
        sig["accuracy"] = {
            "direction_accuracy": r.get("direction_accuracy"),
            "sharpe_signal_w_cost": r.get("sharpe_signal_w_cost"),
            "dsr_signal_w_cost": r.get("dsr_signal_w_cost"),
            "n": r.get("acc_n"),
        }
    return signals


def get_heston_params_for_ticker(ticker: str) -> HestonParams | None:
    params_by_ticker = {
        "AAPL": HestonParams(v0=0.04, theta=0.04, kappa=1.5, sigma=0.3, rho=-0.6),
        "NVDA": HestonParams(v0=0.06, theta=0.05, kappa=1.2, sigma=0.5, rho=-0.7),
    }
    return params_by_ticker.get(ticker.upper())


def compute_sharpe(daily_returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252):
    daily_excess = daily_returns.dropna() - risk_free
    if len(daily_excess) < 2 or daily_excess.std() == 0:
        return None
    return (daily_excess.mean() / daily_excess.std()) * np.sqrt(periods_per_year)


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

FRICTION_PRESETS = {
    "Default": dict(delay_days=1, half_spread_bps=2.0, slippage_bps=3.0, fee_bps=0.0),
    "Loose (optimistic)": dict(delay_days=0, half_spread_bps=1.0, slippage_bps=1.0, fee_bps=0.0),
    "Strict (pessimistic)": dict(delay_days=1, half_spread_bps=5.0, slippage_bps=8.0, fee_bps=1.0),
}

OPTIONS_PRESETS = {
    "Default": dict(dte_min=0, dte_max=45, width_pct=0.05, prefer_spreads=True),
    "Loose": dict(dte_min=0, dte_max=90, width_pct=0.10, prefer_spreads=True),
    "Strict": dict(dte_min=0, dte_max=21, width_pct=0.03, prefer_spreads=True),
}

def _parse_int(s: str, default: int | None = None) -> int | None:
    try:
        return int(str(s).strip())
    except Exception:
        return default

def _parse_float(s: str, default: float | None = None) -> float | None:
    try:
        return float(str(s).strip())
    except Exception:
        return default


@dataclass
class ExecutionModel:
    delay_days: int = 1
    half_spread_bps: float = 2.0
    slippage_bps: float = 3.0
    fee_bps: float = 0.0


def apply_latency_delay(df: pd.DataFrame, delay_days: int, pred_col: str = "predicted_return") -> pd.DataFrame:
    out = df.copy()
    if delay_days and delay_days > 0 and pred_col in out.columns:
        out[pred_col] = out[pred_col].shift(delay_days)
    return out

@st.cache_data(ttl=30, show_spinner=False)
def _cached_intraday_series(tk: str):
    df = get_history_intraday_cached(tk, period="1d", interval="1m")
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return df[["Close"]].copy()

@st.cache_data(ttl=10 * 60, show_spinner=False)
def _cached_daily_series(tk: str):
    df = get_history_cached(tk, period="6mo", interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return df[["Close"]].copy()

def render_price_with_prediction(*, tk: str, horizon_days: int, pred_next_price, prefer_intraday: bool = True):
    px = _cached_intraday_series(tk) if prefer_intraday else None
    if px is None:
        px = _cached_daily_series(tk)

    if px is None or px.empty:
        st.warning("No price data available for chart.")
        return

    x = px.index
    y = px["Close"].astype(float)

    last_x = x[-1]
    last_y = float(y.iloc[-1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Price"))

    if pred_next_price is not None:
        pred_x = last_x + pd.Timedelta(days=int(horizon_days))
        fig.add_trace(
            go.Scatter(
                x=[last_x, pred_x],
                y=[last_y, float(pred_next_price)],
                mode="lines",
                name="Predicted price",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def apply_costs_on_trades(
    strat_df: pd.DataFrame,
    exec_model: ExecutionModel,
    actual_ret_col: str = "actual_return",
    pos_col: str = "position",
) -> pd.Series:
    pos = strat_df.get(pos_col, pd.Series(index=strat_df.index, data=0.0)).fillna(0.0)
    pos_change = pos.diff().abs().fillna(0.0)
    base = strat_df.get(actual_ret_col, pd.Series(index=strat_df.index, data=0.0)).fillna(0.0) * pos
    per_trade_cost = (exec_model.half_spread_bps + exec_model.slippage_bps + exec_model.fee_bps) / 10000.0
    costs = per_trade_cost * pos_change
    return base - costs


def detect_big_news(articles, sent_thresh: float = 0.5) -> bool:
    if not articles:
        return False
    keywords = [
        "earnings",
        "guidance",
        "downgrade",
        "upgrade",
        "lawsuit",
        "investigation",
        "merger",
        "acquisition",
        "bankruptcy",
        "sec charges",
        "fraud",
        "buyback",
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
    pred_pct = float(pred_ret or 0.0) * 100.0
    threshold_multiplier = {1: 1.0, 2: 1.4, 3: 1.7, 4: 2.0, 5: 2.3}.get(int(horizon), 1.0)
    adjusted_threshold = 1.0 * threshold_multiplier

    if abs(pred_pct) > adjusted_threshold:
        if pred_pct > 0:
            if put_call_ratio and put_call_ratio > 1.2:
                return "BULLISH: Buy Calls (high put OI suggests potential squeeze)", "bullish"
            return "BULLISH: Buy Calls or Bull Call Spread", "bullish"
        else:
            if put_call_ratio and put_call_ratio < 0.8:
                return "BEARISH: Buy Puts (low protection)", "bearish"
            return "BEARISH: Buy Puts or Bear Put Spread", "bearish"

    if abs(pred_pct) < (0.5 * threshold_multiplier) and atm_iv and float(atm_iv) > 0.35:
        return "NEUTRAL: Sell Iron Condor or Straddle (high IV)", "neutral"

    return "NEUTRAL: Wait / no-trade", "neutral"


def normalize_model_option_strategy(text: str, prefer_spreads: bool) -> str | None:
    s = (text or "").lower()
    if "bullish" in s and "call" in s and "spread" in s:
        return "BULL_CALL_SPREAD" if prefer_spreads else "BUY_CALL"
    if "bullish" in s and "buy call" in s:
        return "BUY_CALL"
    if "bearish" in s and "put" in s and "spread" in s:
        return "BEAR_PUT_SPREAD" if prefer_spreads else "BUY_PUT"
    if "bearish" in s and "buy put" in s:
        return "BUY_PUT"
    if "iron condor" in s:
        return "IRON_CONDOR"
    return None


def _build_display_df(pred_df: pd.DataFrame, display_horizon: int):
    horizon_label = {1: "1-Day", 2: "2-Day", 3: "3-Day", 4: "4-Day", 5: "5-Day"}[int(display_horizon)]
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

    cols_to_show = [c for c in cols_to_show if c in pred_df.columns]
    display = pred_df[cols_to_show].copy()

    rename_map = {
        "ticker": "Ticker",
        "model_type": "Model",
        "horizon": "Days Ahead",
        "last_close": "Last Close",
        "liveprice": "Live Price",
        "livets": "Live As Of",
        "vol_20d": "Vol 20D",
        "pe_ratio": "P/E",
        "num_features": "# Features",
        "atm_iv": "ATM IV",
        "iv_minus_realized": "IV - Realized",
        "put_call_oi_ratio": "Put/Call OI",
        "pred_next_ret_pct": f"Pred {horizon_label} Return (%)",
        "pred_next_price": "Pred Price",
        "prob_up": "Prob Up",
        "prob_down": "Prob Down",
        "prob_up_gaf": "GAF Prob Up",
        "opt_exp": "Opt Expiry",
        "theo_atm_call_price": "Theo ATM Call",
        "signal_alignment": "Alignment",
        "mc_ev": "MC EV",
        "mc_pop_gt0": "MC POP>0",
    }
    display.rename(columns=rename_map, inplace=True)
    return display


def build_signals_from_pred_df(
    pred_df: pd.DataFrame,
    *,
    prediction_horizon: int,
    trade_mode: str,
    prefer_spreads: bool,
    dte_min: int,
    dte_max: int,
    max_strike: float,
    max_premium: float,
    width_pct: float,
    exec_model: ExecutionModel,
) -> dict:
    signals = {}
    if pred_df is None or pred_df.empty:
        return signals

    for _, row in pred_df.iterrows():
        tk = str(row.get("ticker", "")).upper().strip()
        if not tk:
            continue

        pred = float(row.get("pred_next_ret") or 0.0)
        stock_action = "BUY" if pred >= 0.005 else ("SELL" if pred <= -0.005 else "HOLD")

        strat_text, _bias = suggest_options_strategy(
            pred_ret=pred,
            put_call_ratio=row.get("put_call_oi_ratio"),
            atm_iv=row.get("atm_iv"),
            horizon=int(prediction_horizon),
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

    return signals


def build_signals_from_results(results_df: pd.DataFrame, universe_text: str) -> dict:
    """Convert portfolio backtest → simple BUY/SELL/HOLD signals for auto trader."""
    if results_df is None or results_df.empty:
        return {}

    recent_sharpe = results_df["sharpe"].tail(3).mean()
    if recent_sharpe > 1.0:
        action = "BUY"
    elif recent_sharpe < -0.5:
        action = "SELL"
    else:
        action = "HOLD"

    tickers = [t.strip().upper() for t in universe_text.split(",") if t.strip()]
    return {t: {"asset": "stock", "action": action, "qty": 1} for t in tickers}


# ---------------------------
# Cached wrappers (UI-only)
# ---------------------------
@st.cache_data(show_spinner=False, ttl=15 * 60)
def _cached_screen_stocks(tickers_tuple, ret_thresh, vol_spike_thresh):
    tickers = list(tickers_tuple)
    return screen_stocks(tickers, ret_thresh=ret_thresh, vol_spike_thresh=vol_spike_thresh)


@st.cache_data(show_spinner=False, ttl=15 * 60)
def _cached_predict_bundle(
    tk: str,
    period: str,
    model_type: str,
    prediction_horizon: int,
    auto_optimize: bool,
    run_gaf: bool,
    pricing_model_name: str,
    fetch_live_price: bool,
    run_mc: bool,
):
    pricing_model = getattr(PricingModel, pricing_model_name)

    out = predict_next_for_ticker(
        tk,
        period=period,
        model_type=model_type,
        horizon=int(prediction_horizon),
        use_vol_scaled_target=False,
        auto_optimize=auto_optimize,
        run_gaf=run_gaf,
    )

    opt = get_option_snapshot_features(tk) or {}
    if isinstance(opt, dict):
        out.update(opt)

    atm_iv = out.get("atm_iv")
    last_close = out.get("last_close")

    # Optional intraday (slow)
    live_price, live_ts = None, None
    if fetch_live_price:
        try:
            intraday = get_history_intraday_cached(tk, period="1d", interval="1m")
            if intraday is not None and (not intraday.empty) and ("Close" in intraday.columns):
                live_price = float(intraday["Close"].iloc[-1])
                live_ts = intraday.index[-1]
        except Exception:
            live_price, live_ts = None, None
    out["liveprice"] = live_price
    out["livets"] = str(live_ts) if live_ts is not None else None

    # IV minus realized
    out["iv_minus_realized"] = None
    if atm_iv is not None and out.get("vol_20d") is not None:
        try:
            out["iv_minus_realized"] = float(atm_iv) - float(out["vol_20d"])
        except Exception:
            out["iv_minus_realized"] = None

    # Theo ATM call
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

            if pricing_model == PricingModel.HESTON:
                hp = get_heston_params_for_ticker(tk)
                if hp is None:
                    theo_price = price_option(opt_spec, model=PricingModel.BLACK_SCHOLES)
                else:
                    theo_price = price_option(opt_spec, model=pricing_model, heston_params=hp)
            else:
                theo_price = price_option(opt_spec, model=pricing_model)

            out["theo_atm_call_price"] = float(theo_price)
    except Exception:
        out["theo_atm_call_price"] = None

    # Optional MC (slow)
    if run_mc and (atm_iv is not None) and (last_close is not None):
        try:
            mc_res = option_mc_ev(
                s0=float(last_close),
                mu=float(out.get("pred_next_ret") or 0.0),
                sigma=float(atm_iv),
                days=int(prediction_horizon),
                premium=1.0,
                strike=float(last_close),
                n_paths=5000,
                is_call=True,
            )
            if isinstance(mc_res, dict):
                out.update(mc_res)
        except Exception:
            pass

    out["signal_alignment"] = classify_alignment(out.get("pred_next_ret"), out.get("put_call_oi_ratio"))
    return out


@st.cache_data(show_spinner=False, ttl=30 * 60)
def _cached_track_predictions(ticker: str, period: str, model_type: str, horizon: int):
    return track_predictions(ticker, period=period, model_type=model_type, horizon=int(horizon))

@st.cache_data(ttl=30, show_spinner=False)
def _cached_intraday_close(tk: str):
    df = get_history_intraday_cached(tk, period="1d", interval="1m")
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return df[["Close"]].copy()

@st.cache_data(ttl=10 * 60, show_spinner=False)
def _cached_daily_close(tk: str):
    df = get_history_cached(tk, period="6mo", interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return df[["Close"]].copy()

def render_price_with_prediction(tk: str, horizon_days: int, pred_next_price, prefer_intraday: bool = True):
    px = _cached_intraday_close(tk) if prefer_intraday else None
    if px is None:
        px = _cached_daily_close(tk)

    if px is None or px.empty:
        st.warning("No price data available for chart.")
        return

    x = px.index
    y = px["Close"].astype(float)

    last_x = x[-1]
    last_y = float(y.iloc[-1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Price"))

    if pred_next_price is not None:
        pred_x = last_x + pd.Timedelta(days=int(horizon_days))
        fig.add_trace(
            go.Scatter(
                x=[last_x, pred_x],
                y=[last_y, float(pred_next_price)],
                mode="lines",
                name="Predicted price",
                line=dict(dash="dash"),
            )
        )

    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False, ttl=30 * 60)
def _cached_walk_forward_backtest(
    ticker: str,
    period: str,
    horizon: int,
    model_type: str,
    train_years: int,
    test_years: int,
    step_days: int,
    threshold: float,
):
    return walk_forward_backtest(
        ticker=ticker,
        period=period,
        horizon=int(horizon),
        model_type=model_type,
        train_years=int(train_years),
        test_years=int(test_years),
        step_days=int(step_days),
        threshold=float(threshold),
    )


@st.cache_data(show_spinner=False, ttl=30 * 60)
def _cached_walkforward_cross_sectional(
    tickers_tuple,
    period: str,
    horizon: int,
    model_type: str,
    train_years: int,
    test_years: int,
    top_pct_long: float,
    top_pct_short: float,
    vix_filter,
):
    return walkforward_cross_sectional(
        tickers=list(tickers_tuple),
        period=period,
        horizon=int(horizon),
        model_type=model_type,
        train_years=int(train_years),
        test_years=int(test_years),
        top_pct_long=float(top_pct_long),
        top_pct_short=float(top_pct_short),
        vix_filter=vix_filter,
    )


# ---------------------------
# UI helpers
# ---------------------------
def _signal_label(pred_ret: float | None, prob_up: float | None) -> str:
    pr = float(pred_ret or 0.0)
    pu = None if prob_up is None else float(prob_up)
    if pu is not None:
        if pr > 0.015 and pu > 0.60:
            return "STRONG BUY"
        if pr > 0.005 and pu > 0.55:
            return "BUY"
        if pr < -0.015 and (1 - pu) > 0.60:
            return "STRONG SELL"
        if pr < -0.005 and (1 - pu) > 0.55:
            return "SELL"
    if pr > 0.005:
        return "BUY"
    if pr < -0.005:
        return "SELL"
    return "NEUTRAL"


def run_app():
    st.set_page_config(page_title="Stock Predictor", layout="wide")
    st.title("Stock Predictor")

    # Session state
    st.session_state.setdefault("pred_df", None)
    st.session_state.setdefault("model_type", "rf")
    st.session_state.setdefault("screener_df", None)
    st.session_state.setdefault("prediction_horizon", 5)
    st.session_state.setdefault("period", "5y")
    st.session_state.setdefault("auto_optimize", True)
    st.session_state.setdefault("last_signals", None)
    st.session_state.setdefault("last_trader_stdout", "")
    st.session_state.setdefault("last_trader_stderr", "")
    st.session_state.setdefault("last_trader_rc", None)

    # Tabs
    tab_dash, tab_research, tab_backtests, tab_port = st.tabs(
        ["Dashboard", "Research", "Backtests", "Portfolio WF"]
    )

    # ===================== SIDEBAR =====================
    st.sidebar.header("Run")

    watchlist_text = st.sidebar.text_input("Tickers (comma-separated)", value="AAPL, NVDA")
    tickers = [t.strip().upper() for t in watchlist_text.split(",") if t.strip()]

    period = st.sidebar.selectbox("Training period", ["2y", "5y", "10y"], index=1)

    prediction_horizon = st.sidebar.selectbox("Horizon (days)", [1, 2, 3, 4, 5], index=4)
    horizon_label = {1: "1-Day", 2: "2-Day", 3: "3-Day", 4: "4-Day", 5: "5-Day"}[prediction_horizon]

    model_label = st.sidebar.selectbox("Model", ["rf", "gbrt", "xgb"], index=0)
    model_type = model_label

    auto_optimize = st.sidebar.checkbox("Auto-optimize features", value=True)

    st.sidebar.subheader("Candidate filters")
    max_tickers = st.sidebar.slider("Max tickers per run", 1, 50, 20)
    ret_thresh = st.sidebar.slider("Min |recent return| (%)", 0.0, 10.0, 3.0, 0.5)
    vol_spike_thresh = st.sidebar.slider("Min volume spike (× avg)", 0.5, 5.0, 1.5, 0.1)
    min_move = st.sidebar.slider("Min |predicted return| (%)", 0.0, 5.0, 1.0, 0.1)
    min_iv, max_iv = st.sidebar.slider("ATM IV range", 0.0, 1.0, (0.2, 0.8), 0.05)
    exclude_disagree = st.sidebar.checkbox("Hide 'disagree'", value=True)

    st.sidebar.subheader("Validation")
    signal_threshold_pct = st.sidebar.slider("Signal threshold (%)", 0.0, 2.0, 0.5, 0.05) / 100.0

    with st.sidebar.expander("Advanced", expanded=False):
        pricing_model_label = st.selectbox("Pricing engine", ["Black-Scholes", "Heston"], index=0)
        pricing_model = PricingModel.BLACK_SCHOLES if pricing_model_label == "Black-Scholes" else PricingModel.HESTON

        st.markdown("Elastic Net feature selection")
        use_elasticnet_select = st.checkbox("Enable Elastic Net selection", value=False)
        en_l1_ratio = st.slider("l1_ratio", 0.0, 1.0, 0.5, 0.05)
        en_cv_folds = st.slider("CV folds", 3, 8, 5, 1)

        if use_elasticnet_select and ElasticNetCV is None:
            st.error("Elastic Net requires scikit-learn (ElasticNetCV not available).")

        os.environ["USE_ELASTICNET_SELECT"] = "1" if use_elasticnet_select else "0"
        os.environ["ELASTICNET_L1_RATIO"] = str(en_l1_ratio)
        os.environ["ELASTICNET_CV_FOLDS"] = str(en_cv_folds)

        run_gaf = st.checkbox("Run GAF-CNN (slow)", value=False)
        fetch_live_price = st.checkbox("Fetch intraday live price (slower)", value=False)
        run_mc = st.checkbox("Compute Monte Carlo metrics (slower)", value=False)

                # --- Friction presets ---
        st.markdown("Execution (frictions)")
        friction_preset = st.selectbox("Friction preset", list(FRICTION_PRESETS.keys()), index=0)
        fp = FRICTION_PRESETS[friction_preset]
        exec_model = ExecutionModel(
            delay_days=int(fp["delay_days"]),
            half_spread_bps=float(fp["half_spread_bps"]),
            slippage_bps=float(fp["slippage_bps"]),
            fee_bps=float(fp["fee_bps"]),
)

        with st.expander("Override frictions (optional)", expanded=False):
            exec_model.delay_days = _parse_int(st.text_input("Delay days", value=str(exec_model.delay_days)), exec_model.delay_days)
            exec_model.half_spread_bps = _parse_float(st.text_input("Half-spread (bps)", value=str(exec_model.half_spread_bps)), exec_model.half_spread_bps)
            exec_model.slippage_bps = _parse_float(st.text_input("Slippage (bps)", value=str(exec_model.slippage_bps)), exec_model.slippage_bps)
            exec_model.fee_bps = _parse_float(st.text_input("Fees (bps)", value=str(exec_model.fee_bps)), exec_model.fee_bps)

        # --- Options budget (typed) ---
        st.markdown("Auto-trader (options)")
        trade_mode = st.selectbox("Trade mode", ["Stocks only", "Options if suggested", "Options only"], index=1)

        options_preset = st.selectbox("Options preset", list(OPTIONS_PRESETS.keys()), index=0)
        op = OPTIONS_PRESETS[options_preset]

        budget_per_contract = _parse_float(st.text_input("Max premium ($/contract)", value="500"), 500.0)
        max_premium = float(budget_per_contract)

        max_strike = _parse_float(st.text_input("Max strike", value="500"), 500.0)

        dte_min = _parse_int(st.text_input("Min DTE (days)", value=str(op["dte_min"])), int(op["dte_min"]))
        dte_max = _parse_int(st.text_input("Max DTE (days)", value=str(op["dte_max"])), int(op["dte_max"]))

        width_pct_in = _parse_float(st.text_input("Spread width (%)", value=str(op["width_pct"] * 100.0)), op["width_pct"] * 100.0)
        width_pct = float(width_pct_in) / 100.0

        prefer_spreads = st.checkbox("Prefer spreads", value=bool(op["prefer_spreads"]))
        auto_run_trader = st.checkbox("Auto-run trader after signals.json", value=False)


    # Defaults if Advanced not opened (needed because variables are defined inside the expander block above)
    if "run_gaf" not in locals():
        run_gaf = False
    if "fetch_live_price" not in locals():
        fetch_live_price = False
    if "run_mc" not in locals():
        run_mc = False
    if "pricing_model" not in locals():
        pricing_model = PricingModel.BLACK_SCHOLES
    if "exec_model" not in locals():
        exec_model = ExecutionModel()
    if "n_trials" not in locals():
        n_trials = 20
    if "trade_mode" not in locals():
        trade_mode = "Options if suggested"
    if "dte_min" not in locals():
        dte_min = 0
    if "dte_max" not in locals():
        dte_max = 45
    if "max_strike" not in locals():
        max_strike = 500
    if "max_premium" not in locals():
        max_premium = 500
    if "width_pct" not in locals():
        width_pct = 0.05
    if "prefer_spreads" not in locals():
        prefer_spreads = True
    if "auto_run_trader" not in locals():
        auto_run_trader = False
    if "friction_preset" not in locals():
        friction_preset = "Default"
    if "options_preset" not in locals():
        options_preset = "Default"
    if "budget_per_contract" not in locals():
        budget_per_contract = 500.0
    if "use_ols_sig_select" not in locals():
        os.environ["USE_OLSSIGSELECT"] = "0"
    if "ols_alpha" not in locals():
        os.environ["OLSSIG_ALPHA"] = "0.05"
    if "ols_topk" not in locals():
        os.environ["OLSSIG_TOPK"] = "50"

    # ===================== TAB: Dashboard =====================
    with tab_dash:
        st.caption("Run the screener + model, then drill into one ticker at a time.")

        run_clicked = st.button("Run Screener + Model", type="primary")
        if run_clicked:
            if not tickers:
                st.error("Please enter at least one ticker.")
                st.stop()

            try:
                with st.spinner("Running screener..."):
                    screener_df = _cached_screen_stocks(tuple(tickers), ret_thresh / 100.0, vol_spike_thresh)

                with st.expander("Screener results (raw)", expanded=False):
                    st.dataframe(screener_df, use_container_width=True)

                if screener_df is None or screener_df.empty:
                    st.warning("No screener data returned.")
                    st.stop()

                if "flag" in screener_df.columns:
                    flagged = screener_df.loc[screener_df["flag"] == True, "ticker"].tolist()
                else:
                    flagged = []

                if not flagged:
                    flagged = tickers

                if len(flagged) > max_tickers:
                    st.warning(f"Limiting to first {max_tickers} tickers to avoid rate limits.")
                    flagged = flagged[:max_tickers]

                st.info(f"Running {horizon_label} predictions on: {', '.join(flagged)}")

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                results = []
                for i, tk in enumerate(flagged):
                    progress_bar.progress((i + 1) / len(flagged))
                    status_text.text(f"Processing {tk}... ({i+1}/{len(flagged)})")
                    try:
                        out = _cached_predict_bundle(
                            tk=tk,
                            period=period,
                            model_type=model_type,
                            prediction_horizon=prediction_horizon,
                            auto_optimize=auto_optimize,
                            run_gaf=run_gaf,
                            pricing_model_name=pricing_model.name,
                            fetch_live_price=fetch_live_price,
                            run_mc=run_mc,
                        )
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

                pred_df = pd.DataFrame(results)
                if "pred_next_ret" in pred_df.columns:
                    pred_df["pred_next_ret_pct"] = pred_df["pred_next_ret"] * 100.0

                # Add convenience columns for ranking
                if "vol_20d" in pred_df.columns and "pred_next_ret" in pred_df.columns:
                    pred_df["vol_adjusted_edge"] = pred_df["pred_next_ret"] / pred_df["vol_20d"].replace(0, np.nan)

                if "pred_next_ret" in pred_df.columns:
                    pred_df["signal_label"] = [
                        _signal_label(r.get("pred_next_ret"), r.get("prob_up"))
                        for _, r in pred_df.iterrows()
                    ]

                st.session_state.pred_df = pred_df
                st.session_state.model_type = model_type
                st.session_state.screener_df = screener_df
                st.session_state.prediction_horizon = prediction_horizon
                st.session_state.period = period
                st.session_state.auto_optimize = auto_optimize

                signals = build_signals_from_pred_df(
                    pred_df,
                    prediction_horizon=prediction_horizon,
                    trade_mode=trade_mode,
                    prefer_spreads=prefer_spreads,
                    dte_min=dte_min,
                    dte_max=dte_max,
                    max_strike=max_strike,
                    max_premium=max_premium,
                    width_pct=width_pct,
                    exec_model=exec_model,
                )

                signals = enrich_signals_in_place(
                    signals,
                    pred_df=pred_df,
                    context={
                        "model_type": model_type,
                        "period": period,
                        "horizon": prediction_horizon,
                        "friction_preset": friction_preset,
                        "options_preset": options_preset,
                        "budget_per_contract": max_premium,
                        "signal_threshold_pct": signal_threshold_pct,
                    },
                )
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

            except Exception as e:
                st.error(f"Run failed: {e}")
                st.stop()

        if st.session_state.pred_df is None:
            st.info("Run the model to populate the dashboard.")
            st.stop()

        pred_df = st.session_state.pred_df
        display_horizon = st.session_state.get("prediction_horizon", 1)
        display_horizon_label = {1: "1-Day", 2: "2-Day", 3: "3-Day", 4: "4-Day", 5: "5-Day"}[display_horizon]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickers", len(pred_df))
        c2.metric("Horizon", display_horizon_label)
        c3.metric("Model", st.session_state.model_type.upper())
        c4.metric("Period", st.session_state.get("period", "5y"))

        # Default view: candidates + drilldown
        st.subheader("Top candidates (filtered)")
        cand_df = pred_df.copy()

        if "pred_next_ret_pct" in cand_df.columns:
            cand_df["abs_pred_pct"] = cand_df["pred_next_ret_pct"].abs()
        else:
            cand_df["abs_pred_pct"] = np.nan

        if "atm_iv" in cand_df.columns:
            mask = cand_df["atm_iv"].between(min_iv, max_iv)
        else:
            mask = pd.Series(True, index=cand_df.index)

        if "abs_pred_pct" in cand_df.columns:
            mask &= (cand_df["abs_pred_pct"] >= min_move)

        if exclude_disagree and "signal_alignment" in cand_df.columns:
            mask &= (cand_df["signal_alignment"] != "disagree")

        cand_df = cand_df[mask].copy()

        sort_col = "vol_adjusted_edge" if "vol_adjusted_edge" in cand_df.columns else "abs_pred_pct"
        if sort_col in cand_df.columns:
            cand_df = cand_df.sort_values(sort_col, ascending=False)

        if cand_df.empty:
            st.write("No candidates matched your filters.")
            detail_universe = pred_df["ticker"].tolist()
        else:
            cols = [c for c in [
                "ticker",
                "signal_label",
                "pred_next_ret_pct",
                "prob_up",
                "vol_20d",
                "vol_adjusted_edge",
                "atm_iv",
                "put_call_oi_ratio",
                "signal_alignment",
                "pred_next_price",
                "num_features",
                "prob_up_gaf",
            ] if c in cand_df.columns]
            st.dataframe(cand_df[cols], use_container_width=True, height=280)
            detail_universe = cand_df["ticker"].tolist()
        #========== Accuracy=========
        st.subheader("Accuracy (optional)")
        run_acc_all = st.checkbox("Compute accuracy for ALL tickers (slow)", value=False)
        if run_acc_all and st.button("Run accuracy now", type="primary"):
            rows = []
            pbar = st.progress(0.0)
            for i, tk in enumerate(pred_df["ticker"].tolist()):
                pbar.progress((i + 1) / len(pred_df))
                rows.append(
                    _cached_accuracy_row(
                        tk=tk,
                        period=st.session_state.get("period", "5y"),
                        model_type=st.session_state.get("modeltype", "rf"),
                        horizon=display_horizon,
                        delay_days=exec_model.delay_days,
                        half_spread_bps=exec_model.half_spread_bps,
                        slippage_bps=exec_model.slippage_bps,
                        fee_bps=exec_model.fee_bps,
                        signal_threshold_pct=signal_threshold_pct,
                        n_trials=n_trials,
                    )
                )
                pbar.empty()

            acc_df = pd.DataFrame(rows)
            pred_df = pred_df.merge(acc_df, on="ticker", how="left")
            st.session_state.pred_df = pred_df  # persist merged metrics
            st.success("Accuracy merged into Dashboard table.")

        st.subheader("Ticker details")
        selected = st.selectbox("Select ticker", detail_universe, key="dash_selected_ticker")
        row = pred_df[pred_df["ticker"] == selected].iloc[0]
        st.subheader("Live price + forecast")
        render_price_with_prediction(
            tk=selected,
            horizon_days=display_horizon,
            pred_next_price=row.get("pred_next_price"),
            prefer_intraday=True,
        )
        left, right = st.columns([1.25, 1])
        with left:
            with st.expander("Model", expanded=True):
                c1, c2, c3 = st.columns(3)

                # Pred return is stored as decimal in row.get("prednextret")
                pred_ret = row.get("pred_next_ret", None)
                prob_up = row.get("prob_up", None)

                c1.metric(f"{display_horizon_label} prediction", _fmt_pct(pred_ret, 2))
                c2.metric("Prob up", "—" if prob_up is None else f"{float(prob_up):.3f}")
                c3.metric("Features used", "—" if row.get("num_features") is None else str(int(row.get("num_features"))))

                st.markdown(
                    f"""
        - Last close: {_fmt_num(row.get("last_close"), 2)}
        - Predicted price: {_fmt_num(row.get("pred_next_price"), 2)}
        - Alignment: {row.get("signal_alignment", "—")}
        - Vol-adjusted edge: {_fmt_num(row.get("vol_adjusted_edge"), 3)}
                    """.strip()
                )

        import plotly.graph_objects as go
        import pandas as pd

        with st.expander("Price chart", expanded=True):
            hist = get_history_cached(selected, period="3mo", interval="1d")
            if hist is None or hist.empty or "Close" not in hist.columns:
                st.warning("No price history available for chart.")
            else:
                close = hist["Close"].dropna()
                last_dt = close.index[-1]
                pred_price = row.get("prednextprice")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Close", mode="lines"))

                if pred_price is not None:
                    future_dt = last_dt + pd.Timedelta(days=int(display_horizon))
                    fig.add_trace(go.Scatter(
                        x=[future_dt], y=[pred_price],
                        name="Predicted", mode="markers",
                        marker=dict(size=10)
                    ))

                st.plotly_chart(fig, use_container_width=True)

        with right:
            with st.expander("Options", expanded=True):
                strat, _bias = suggest_options_strategy(
                    float(row.get("pred_next_ret") or 0.0),
                    row.get("put_call_oi_ratio"),
                    row.get("atm_iv"),
                    horizon=int(display_horizon),
                )

                st.write(strat)

                o1, o2, o3 = st.columns(3)
                o1.metric("ATM IV", _fmt_num(row.get("atm_iv"), 3))
                o2.metric("Put/Call OI", _fmt_num(row.get("put_call_oi_ratio"), 3))
                o3.metric("Theo ATM call", _fmt_num(row.get("theo_atm_call_price"), 3))

                st.write(f"IV - realized: {_fmt_num(row.get('iv_minus_realized'), 3)}")

            with st.expander("Greeks + News (slow)", expanded=False):
                loadslow = st.checkbox("Load Greeks + News", value=False, key="dash_load_slow_expander")
                if loadslow:
                    greeksinfo = None
                    try:
                        greeksinfo = get_atm_greeks(selected)
                    except Exception:
                        greeksinfo = None

                    if greeksinfo:
                        cg, pg = greeksinfo.get("call_greeks"), greeksinfo.get("put_greeks")
                        if cg:
                            st.write(f"Call Δ {cg.get('delta', 0):.2f} Γ {cg.get('gamma', 0):.4f} Vega {cg.get('vega', 0):.2f} Θ {cg.get('theta', 0):.2f}")
                        if pg:
                            st.write(f"Put  Δ {pg.get('delta', 0):.2f} Γ {pg.get('gamma', 0):.4f} Vega {pg.get('vega', 0):.2f} Θ {pg.get('theta', 0):.2f}")
                    else:
                        st.write("Greeks: NA")

                    news = get_news_for_ticker(selected, limit=5)
                    if detect_big_news(news):
                        st.warning("Recent BIG news headlines detected.")
                    if news:
                        for art in news:
                            title = art.get("title") or "No title"
                            url = art.get("url")
                            st.markdown(f"- [{title}]({url})" if url else f"- {title}")
                    else:
                        st.write("News: none / not configured")


        with st.expander("Full predictions (all tickers)", expanded=False):
            display = _build_display_df(pred_df, display_horizon)
            st.dataframe(display, use_container_width=True)
            pred_col = f"Pred {display_horizon_label} Return (%)"
            if pred_col in display.columns:
                st.subheader(f"{pred_col} by ticker")
                st.bar_chart(display.set_index("Ticker")[pred_col])

        with st.expander("Signals + trader output", expanded=False):
            st.write(f"signals.json path: {SIGNALS_OUT_PATH}")
            if st.session_state.last_signals:
                sig_rows = []
                for tk, s in st.session_state.last_signals.items():
                    if s.get("asset") == "option":
                        sig_rows.append(
                            {
                                "Ticker": tk,
                                "Asset": "option",
                                "Strategy": s.get("strategy"),
                                "DTE max": s.get("dte_max"),
                                "Max premium ($)": s.get("max_premium"),
                                "Pred next ret (%)": round(float(s.get("pred_next_ret", 0.0)) * 100, 2),
                            }
                        )
                    else:
                        sig_rows.append(
                            {
                                "Ticker": tk,
                                "Asset": "stock",
                                "Action": s.get("action"),
                                "Pred next ret (%)": round(float(s.get("pred_next_ret", 0.0)) * 100, 2),
                            }
                        )
                st.dataframe(pd.DataFrame(sig_rows), use_container_width=True)
            else:
                st.info("No signals written yet in this session.")

            if st.session_state.last_trader_rc is None:
                st.info("Trader has not been run yet (or auto-run is off).")
            else:
                st.write(f"Return code: {st.session_state.last_trader_rc}")
                st.code(st.session_state.last_trader_stdout or "(no stdout)", language="text")
                if st.session_state.last_trader_rc != 0:
                    st.code(st.session_state.last_trader_stderr or "(no stderr)", language="text")

    # ===================== TAB: Research =====================
    with tab_research:
        st.header("Research")
        if st.session_state.pred_df is None:
            st.warning("Run Dashboard first so this tab can reuse tickers/model/horizon.")
            st.stop()

        pred_df = st.session_state.pred_df
        display_horizon = st.session_state.get("prediction_horizon", 1)
        period = st.session_state.get("period", "5y")
        model_type = st.session_state.get("model_type", model_type)

        tickers_for_select = pred_df["ticker"].tolist()
        test_ticker = st.selectbox("Ticker", tickers_for_select, key="research_test_ticker")

        if st.button("Run Accuracy Test", key="research_run_acc"):
            try:
                with st.spinner(f"Testing {test_ticker}..."):
                    results_test, accuracy = _cached_track_predictions(
                        test_ticker, period=period, model_type=model_type, horizon=display_horizon
                    )
                if results_test is None or results_test.empty:
                    st.warning("Not enough data to test accuracy.")
                    st.stop()

                st.metric("Directional accuracy", f"{accuracy * 100:.1f}%")

                baseline_returns = results_test["actual_return"].dropna()
                results_exec = apply_latency_delay(results_test, delay_days=exec_model.delay_days, pred_col="predicted_return")

                strat = results_exec.copy()
                strat["position"] = np.where(strat["predicted_return"] > signal_threshold_pct, 1.0, 0.0)
                strat["strategy_ret_no_cost"] = strat["actual_return"] * strat["position"]
                strat["strategy_ret_with_cost"] = apply_costs_on_trades(strat, exec_model)

                sharpe_baseline = compute_sharpe(baseline_returns)
                sharpe_signal_no_cost = compute_sharpe(strat["strategy_ret_no_cost"])
                sharpe_signal_with_cost = compute_sharpe(strat["strategy_ret_with_cost"])
                dsr_baseline = deflated_sharpe_ratio(baseline_returns, n_trials)
                dsr_signal_with_cost = deflated_sharpe_ratio(strat["strategy_ret_with_cost"], n_trials)

                c1, c2, c3 = st.columns(3)
                c1.metric("Sharpe (B&H)", "N/A" if sharpe_baseline is None else f"{sharpe_baseline:.2f}")
                c2.metric("Sharpe (Signal, no cost)", "N/A" if sharpe_signal_no_cost is None else f"{sharpe_signal_no_cost:.2f}")
                c3.metric("Sharpe (Signal, w/ cost)", "N/A" if sharpe_signal_with_cost is None else f"{sharpe_signal_with_cost:.2f}")
                st.write(f"DSR (B&H): {'N/A' if dsr_baseline is None else f'{dsr_baseline:.2f}'} (trials≈{n_trials})")
                st.write(f"DSR (Signal): {'N/A' if dsr_signal_with_cost is None else f'{dsr_signal_with_cost:.2f}'} (trials≈{n_trials})")

                with st.expander("Raw results table", expanded=False):
                    recent_df = results_test[
                        ["date", "predicted_return", "actual_return", "predicted_price", "actual_close", "correct_direction"]
                    ].copy()
                    recent_df["predicted_return"] *= 100
                    recent_df["actual_return"] *= 100
                    st.dataframe(recent_df, use_container_width=True)

                chart_df = pd.DataFrame(
                    {
                        "Predicted (%)": results_test["predicted_return"].values * 100,
                        "Actual (%)": results_test["actual_return"].values * 100,
                    },
                    index=results_test["date"],
                )
                st.line_chart(chart_df)

                if sq is not None:
                    try:
                        sq_report = sq.performance_summary(
                            strat["strategy_ret_with_cost"].dropna(),
                            benchmark=baseline_returns.loc[strat["strategy_ret_with_cost"].dropna().index],
                        )
                        with st.expander("SquareQuant summary", expanded=False):
                            st.dataframe(sq_report, use_container_width=True)
                    except Exception as e:
                        st.info(f"SquareQuant error: {e}")

            except Exception as e:
                st.error(f"Error testing accuracy: {e}")

        with st.expander("Feature significance + diagnostics", expanded=False):
            fs_ticker = st.selectbox("Feature significance ticker", tickers_for_select, key="fs_ticker_select")
            if st.button("Analyze Feature Significance", key="research_sig_btn"):
                try:
                    with st.spinner("Running OLS feature significance..."):
                        _ols_model, sig_df = analyze_feature_significance(
                            ticker=fs_ticker,
                            period=period,
                            horizon=display_horizon,
                            use_vol_scaled_target=False,
                        )
                    st.dataframe(sig_df.head(25), use_container_width=True)
                except Exception as e:
                    st.error(f"Error computing feature significance: {e}")

            chosen = st.selectbox("Diagnostics ticker", tickers_for_select, key="price_history_selector")
            hist = get_history_cached(chosen, period="3mo", interval="1d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                prices = hist["Close"].copy()
                st.line_chart(prices)

                rets = hist["Close"].pct_change()
                fig_gaf, _ax = make_gaf_image_from_returns(rets, window=60, image_size=30)
                if fig_gaf is not None:
                    st.pyplot(fig_gaf)

    # ===================== TAB: Backtests =====================
    with tab_backtests:
        st.header("Backtests")

        with st.expander("Single-stock backtest", expanded=True):
            bt_ticker = st.text_input("Ticker", "NVDA", key="backtest_ticker")
            bt_horizon = st.selectbox("Horizon (days)", [1, 2, 3, 4, 5], index=4, key="bt_horizon")
            bt_model = st.selectbox("Model", ["rf", "xgb", "gbrt"], index=0, key="bt_model")
            bt_period = st.selectbox("Period", ["2y", "5y", "10y"], index=1, key="bt_period")

            if st.button("Run Backtest", key="run_backtest"):
                try:
                    with st.spinner("Running backtest..."):
                        results_test, accuracy = _cached_track_predictions(
                            bt_ticker, period=bt_period, model_type=bt_model, horizon=bt_horizon
                        )
                    if results_test is None or results_test.empty:
                        st.warning("Not enough data to backtest.")
                        st.stop()

                    baseline_returns = results_test["actual_return"].dropna()
                    results_exec = apply_latency_delay(results_test, delay_days=exec_model.delay_days, pred_col="predicted_return")

                    strat = results_exec.copy()
                    strat["position"] = np.where(strat["predicted_return"] > signal_threshold_pct, 1.0, 0.0)
                    strat["strategy_ret_no_cost"] = strat["actual_return"] * strat["position"]
                    strat["strategy_ret_with_cost"] = apply_costs_on_trades(strat, exec_model)

                    sharpe_baseline = compute_sharpe(baseline_returns)
                    sharpe_strategy_with_cost = compute_sharpe(strat["strategy_ret_with_cost"])

                    total_return_baseline = (1 + baseline_returns).prod() - 1
                    total_return_strategy_with_cost = (1 + strat["strategy_ret_with_cost"].dropna()).prod() - 1
                    num_trades = strat["position"].diff().abs().sum() / 2

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Sharpe (with cost)", "N/A" if sharpe_strategy_with_cost is None else f"{sharpe_strategy_with_cost:.2f}")
                    c2.metric("Hit rate", f"{accuracy * 100:.1f}%")
                    c3.metric("Total return (with cost)", f"{total_return_strategy_with_cost * 100:.1f}%")
                    c4.metric("Trades", int(num_trades))

                    st.write(
                        f"Buy & Hold: Sharpe={'N/A' if sharpe_baseline is None else f'{sharpe_baseline:.2f}'}, "
                        f"Return={total_return_baseline * 100:.1f}%"
                    )

                    cum_baseline = (1 + baseline_returns).cumprod()
                    cum_strategy = (1 + strat["strategy_ret_with_cost"].dropna()).cumprod()
                    chart_df = pd.DataFrame(
                        {"Buy & Hold": cum_baseline.values, "Strategy (with cost)": cum_strategy.values},
                        index=results_test["date"],
                    )
                    st.subheader("Cumulative returns")
                    st.line_chart(chart_df)

                except Exception as e:
                    st.error(f"Error running backtest: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        with st.expander("Walk-forward (single stock)", expanded=False):
            wf_ticker = st.text_input("Ticker", "NVDA", key="wf_ticker")
            wf_horizon = st.selectbox("Horizon (days)", [1, 2, 3, 4, 5], index=4, key="wf_horizon")
            wf_model = st.selectbox("Model", ["rf", "xgb", "gbrt"], index=0, key="wf_model")
            wf_period = st.selectbox("Period", ["2y", "5y", "10y"], index=1, key="wf_period")
            wf_threshold_pct = st.slider("Fold threshold (%)", 0.0, 1.0, 0.2, 0.05, key="wf_threshold") / 100.0
            wf_step_days = st.selectbox("Fold stride (trading days)", [5, 10, 21, 63, 126], index=2, key="wf_step_days")

            if st.button("Run Walk-Forward", key="run_wf"):
                try:
                    with st.spinner("Running walk-forward..."):
                        fold_results = _cached_walk_forward_backtest(
                            ticker=wf_ticker,
                            period=wf_period,
                            horizon=wf_horizon,
                            model_type=wf_model,
                            train_years=3,
                            test_years=1,
                            step_days=int(wf_step_days),
                            threshold=float(wf_threshold_pct),
                        )

                    if not fold_results:
                        st.warning("No folds produced (not enough data or settings too strict).")
                        st.stop()

                    sharpes = [f.get("sharpe", 0.0) for f in fold_results if f.get("sharpe") is not None]
                    avg_sh = (sum(sharpes) / len(sharpes)) if sharpes else None
                    st.metric("Avg Sharpe", "N/A" if avg_sh is None else f"{avg_sh:.3f}")

                    for i, fold in enumerate(fold_results, 1):
                        sh = fold.get("sharpe")
                        hr = fold.get("hit_rate")
                        with st.expander(f"Fold {i} — Sharpe: {'N/A' if sh is None else f'{sh:.3f}'}"):
                            st.write(f"Train: {fold.get('train_start')} → {fold.get('train_end')}")
                            st.write(f"Test: {fold.get('test_start')} → {fold.get('test_end')}")
                            st.write(f"Hit rate: {'N/A' if hr is None else f'{hr * 100:.1f}%'}")
                            st.write(f"Trades: {fold.get('num_trades')}")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        with st.expander("Comprehensive CSV", expanded=False):
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
            else:
                st.dataframe(comp_results, use_container_width=True)

    # ===================== TAB: Portfolio WF =====================
    with tab_port:
        st.header("Portfolio Walk-Forward")
        st.markdown("Production ML Portfolio Engine")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Settings")
            universe_text = st.text_input("Universe", value="AAPL,NVDA,MSFT")
            horizon = st.selectbox("Horizon", [1, 3, 5], format_func=lambda x: f"{x}D")
            c1, c2 = st.columns(2)
            with c1:
                train_years = st.slider("Train (years)", 1, 4, 2)
            with c2:
                test_years = st.slider("Test (years)", 0, 2, 1)

        with col2:
            st.subheader("Portfolio")
            top_long = st.slider("Long %", 0.01, 0.20, 0.10, 0.01)
            top_short = st.slider("Short %", 0.20, 0.50, 0.30, 0.01)
            model_type2 = st.selectbox("Model", ["rf", "xgb", "gbrt"])
            use_vix_filter = st.checkbox("VIX Filter", value=True)
            vix_threshold = st.slider("VIX Max", 15, 35, 25) if use_vix_filter else None

            q1, q2, _q3 = st.columns(3)
            with q1:
                if st.button("SP500 Top 10"):
                    st.session_state.quick_universe = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AVGO,JPM,WMT"
            with q2:
                if st.button("Mag 7"):
                    st.session_state.quick_universe = "AAPL,NVDA,MSFT,GOOGL,AMZN,META,TSLA"

        if hasattr(st.session_state, "quick_universe"):
            universe_text = st.session_state.quick_universe
            st.info(f"Quick load: {universe_text}")

        run_col, est_col = st.columns([3, 1])
        with run_col:
            if st.button("Run Backtest", type="primary", use_container_width=True):
                tickers2 = [t.strip().upper() for t in universe_text.split(",") if t.strip()]
                with st.spinner(f"Running {len(tickers2)} tickers..."):
                    results_df = _cached_walkforward_cross_sectional(
                        tuple(tickers2),
                        period="5y",
                        horizon=horizon,
                        model_type=model_type2,
                        train_years=train_years,
                        test_years=test_years,
                        top_pct_long=top_long,
                        top_pct_short=top_short,
                        vix_filter=vix_threshold if use_vix_filter else None,
                    )
                if results_df is not None and not results_df.empty:
                    st.session_state.results = results_df
                    st.session_state.portfolio_tickers = tickers2
                    st.rerun()

        with est_col:
            n_tickers = len([t for t in universe_text.split(",") if t.strip()])
            st.info(f"Est: ~{n_tickers * train_years * 0.4:.0f}s")

        if "results" in st.session_state and st.session_state.results is not None and not st.session_state.results.empty:
            results_df = st.session_state.results

            st.success(f"{len(results_df)} folds complete!")
            median_sharpe = results_df["sharpe"].median()
            avg_return = results_df["ann_return"].mean() * 100
            worst_dd = results_df["max_dd"].min()
            avg_hit = results_df["hit_rate"].mean() * 100
            recent_sharpe = results_df["sharpe"].tail(3).mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sharpe (Median)", f"{median_sharpe:.2f}")
            m2.metric("Hit rate (Avg)", f"{avg_hit:.0f}%")
            m3.metric("Ann return (Avg)", f"{avg_return:.1f}%")
            m4.metric("Max drawdown (Worst)", f"{worst_dd:.2f}")

            left, right = st.columns([2, 1])
            with left:
                st.subheader("Fold results")
                st.dataframe(results_df.round(3), use_container_width=True, height=320)

                st.subheader("Sharpe distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                results_df["sharpe"].hist(bins=12, ax=ax, alpha=0.7, edgecolor="black")
                ax.axvline(median_sharpe, color="green", lw=2, ls="--", label="Median")
                ax.axvline(0, color="red", lw=1, ls="-", label="0")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            with right:
                st.subheader("Live signal")
                if recent_sharpe > 1.0:
                    st.success(f"DEPLOY (recent Sharpe {recent_sharpe:.2f})")
                elif recent_sharpe > 0.3:
                    st.info(f"EDGE (recent Sharpe {recent_sharpe:.2f})")
                else:
                    st.warning(f"STANDBY (recent Sharpe {recent_sharpe:.2f})")

                with st.expander("Write signals.json", expanded=False):
                    if st.button("Write signals.json now", use_container_width=True):
                        signals = build_signals_from_results(results_df, universe_text)
                        (BASE_DIR / "signals.json").write_text(json.dumps(signals, indent=2))
                        st.success(f"signals.json → {len(signals)} signals")
                        st.json(signals)

                    st.metric("Latest Sharpe", f"{results_df['sharpe'].iloc[-1]:.2f}")


if __name__ == "__main__":
    run_app()
