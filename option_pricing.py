# option_pricing.py
from dataclasses import dataclass
from enum import Enum

import math
import datetime as dt

# If you install QuantLib-Python: pip install QuantLib-Python
try:
    import QuantLib as ql
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False


class PricingModel(str, Enum):
    BLACK_SCHOLES = "BLACK_SCHOLES"
    HESTON = "HESTON"


@dataclass
class OptionSpec:
    spot: float
    strike: float
    maturity_date: dt.date  # option expiry
    valuation_date: dt.date
    rate: float             # risk-free (ccy consistent)
    div_yield: float        # continuous dividend yield
    vol: float              # BS implied vol (for BS engine)
    is_call: bool = True


# ---------- Black–Scholes (closed form) ----------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def price_black_scholes(opt: OptionSpec) -> float:
    T = (opt.maturity_date - opt.valuation_date).days / 365.0
    if T <= 0:
        return max(0.0, (opt.spot - opt.strike) if opt.is_call else (opt.strike - opt.spot))

    S, K, r, q, sigma = opt.spot, opt.strike, opt.rate, opt.div_yield, opt.vol
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if opt.is_call:
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


# ---------- Heston via QuantLib (skeleton) ----------

@dataclass
class HestonParams:
    v0: float
    theta: float
    kappa: float
    sigma: float
    rho: float


def price_heston(opt: OptionSpec, params: HestonParams) -> float:
    if not HAS_QUANTLIB:
        raise RuntimeError("QuantLib not available; cannot use Heston model")

    # Dates
    cal = ql.NullCalendar()
    val_date = ql.Date(opt.valuation_date.day, opt.valuation_date.month, opt.valuation_date.year)
    mat_date = ql.Date(opt.maturity_date.day, opt.maturity_date.month, opt.maturity_date.year)
    ql.Settings.instance().evaluationDate = val_date

    # Term structures
    day_count = ql.Actual365Fixed()
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(val_date, opt.rate, day_count)
    )
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(val_date, opt.div_yield, day_count)
    )

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(opt.spot))

    # Heston process and model
    process = ql.HestonProcess(
        risk_free_ts,
        div_ts,
        spot_handle,
        params.v0,
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)  # closed-form pricing for European options [web:586][web:598]

    # Option
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if opt.is_call else ql.Option.Put,
        opt.strike,
    )
    exercise = ql.EuropeanExercise(mat_date)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)

    return option.NPV()


# ---------- Unified entry point ----------

def price_option(opt: OptionSpec,
                 model: PricingModel = PricingModel.BLACK_SCHOLES,
                 heston_params: HestonParams | None = None) -> float:
    if model == PricingModel.BLACK_SCHOLES:
        return price_black_scholes(opt)
    elif model == PricingModel.HESTON:
        if heston_params is None:
            raise ValueError("heston_params must be provided for HESTON model")
        return price_heston(opt, heston_params)
    else:
        raise ValueError(f"Unsupported model: {model}")
