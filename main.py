# main.py
import pandas as pd
from datetime import datetime

from stock_screener import screen_stocks
from prediction_model import predict_next_for_ticker


def main():
    # 1) Start from a watchlist
    watchlist = ["AAPL", "TSLA", "NVDA", "MSFT", "SPY", "^GSPC"]

    # 2) Run the screener to find “interesting” names
    screener_df = screen_stocks(watchlist)
    print("=== Screener results ===")
    print(screener_df.to_string(index=False))

    # Focus on flagged tickers
    flagged = screener_df[screener_df["flag"] == True]["ticker"].tolist()
    if not flagged:
        print("\nNo tickers passed the screener thresholds. Using full watchlist instead.")
        flagged = watchlist

        print("\n=== Model predictions (next-day) ===")
    results = []
    for tk in flagged:
        try:
            out = predict_next_for_ticker(tk, period="5y")
            results.append(out)
            print(
                f"{tk:6s} | last={out['last_close']:.2f} "
                f"| pred_ret={out['pred_next_ret']:.4%} "
                f"| pred_price={out['pred_next_price']:.2f}"
            )
        except Exception as e:
            print(f"{tk:6s} | ERROR: {e}")

    if not results:
        print("No predictions could be generated.")
        return  # early exit so we don't try to save an empty list

    # Save predictions to CSV log
    df = pd.DataFrame(results)
    df["run_time"] = datetime.utcnow()
    df.to_csv("predictions_log.csv", mode="a", index=False, header=False)
