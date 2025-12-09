# options_curve.py
import pandas as pd
import matplotlib.pyplot as plt

from data_fetch import get_option_chain


def build_option_curve(
    ticker="^SPX",
    expiration=None,
    moneyness_window=0.1,
    plot=True,
):
    """
    Build a simple option 'smile' for calls:
    - pulls an option chain
    - keeps strikes near the current underlying price (via moneyness_window)
    - sorts by strike and returns a DataFrame ready to plot
    """
    chain = get_option_chain(ticker, expiration=expiration, calls_only=True)

    # Use mid price and implied volatility from yfinance [web:2][web:5]
    # We need an approximate underlying; use lastPrice of the closest-to-the-money call
    atm_row = chain.iloc[(chain["strike"] - chain["strike"].median()).abs().argmin()]
    underlying_approx = atm_row["strike"]

    # Filter by moneyness (e.g., +/- 10% around underlying price)
    lower = underlying_approx * (1 - moneyness_window)
    upper = underlying_approx * (1 + moneyness_window)
    curve = chain[(chain["strike"] >= lower) & (chain["strike"] <= upper)].copy()

    curve = curve.sort_values("strike")
    curve = curve[["strike", "mid", "impliedVolatility"]]

    if plot:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Strike")
        ax1.set_ylabel("Mid price", color="tab:blue")
        ax1.plot(curve["strike"], curve["mid"], "o-", color="tab:blue", label="Mid price")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Implied Vol", color="tab:red")
        ax2.plot(
            curve["strike"],
            curve["impliedVolatility"],
            "s--",
            color="tab:red",
            label="Implied Vol",
        )
        ax2.tick_params(axis="y", labelcolor="tab:red")

        plt.title(f"{ticker} option curve ({len(curve)} points)")
        fig.tight_layout()
        plt.show()

    return curve
