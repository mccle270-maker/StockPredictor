import pandas as pd
import numpy as np

# Example data: 30 days of returns for one ticker
np.random.seed(42)
df = pd.DataFrame({
    "ticker": ["AAPL"] * 30,
    "pred_next_ret": np.random.normal(0, 0.02, 30)
})

window = 10
min_periods = 5

# Calculate rolling z-score
df["zscore"] = df.groupby("ticker")["pred_next_ret"].transform(
    lambda x: (x - x.rolling(window, min_periods=min_periods).mean()) / x.rolling(window, min_periods=min_periods).std()
)

print(df[["pred_next_ret", "zscore"]].head(15))
