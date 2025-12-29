# Feature Engineering Implementation Guide

## Quick Start: Adding Features to Your Model

### Current Structure
```
prediction_model.py
├── add_price_features()        ← Add new features here
├── add_advanced_features()     ← Already exists
├── get_macro_df()              ← Macro data fetching
├── FEATURE_COLUMNS             ← List of all features (update this!)
├── ElasticNet selection        ← Auto-prunes weak features
└── train_model()               ← Uses selected features
```

---

## Example 1: Adding Support/Resistance Features

### Code to Add (in `add_price_features()`)

```python
def add_price_features(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()
    close = hist["Close"]
    
    # ... existing code ...
    
    # === NEW: Support/Resistance Features ===
    
    # Distance to 50-day high (mean reversion signal)
    high_50d = close.rolling(50).max()
    hist["dist_from_50d_high"] = (close - high_50d) / high_50d  # Negative = below high
    hist["dist_from_50d_high"] = hist["dist_from_50d_high"].shift(1)  # LAG FOR LOOK-AHEAD!
    
    # Distance to 50-day low (support level)
    low_50d = close.rolling(50).min()
    hist["dist_from_50d_low"] = (close - low_50d) / low_50d   # Positive = above low
    hist["dist_from_50d_low"] = hist["dist_from_50d_low"].shift(1)
    
    # Distance to 52-week high (longer-term trend)
    high_252d = close.rolling(252).max()
    hist["dist_from_52w_high"] = (close - high_252d) / high_252d
    hist["dist_from_52w_high"] = hist["dist_from_52w_high"].shift(1)
    
    # Support level strength: how many times did price bounce from support?
    support_level = low_50d
    bounces = 0
    support_strength = []
    for i in range(50, len(close)):
        if close.iloc[i] == support_level.iloc[i]:  # Hit support
            bounces += 1
        if i % 20 == 0:  # Reset every 20 days
            support_strength.append(bounces)
            bounces = 0
    # Pad with most recent value
    support_strength.extend([support_strength[-1]] * (len(close) - len(support_strength)))
    hist["support_bounce_count_20d"] = support_strength
    hist["support_bounce_count_20d"] = hist["support_bounce_count_20d"].shift(1)
    
    # Price position: 0 = at low, 1 = at high, 0.5 = middle (already have this as close_position)
    # Just keep it: hist["close_position"] already exists
    
    return hist
```

### Update FEATURE_COLUMNS

```python
FEATURE_COLUMNS = [
    # ... existing features ...
    
    # NEW Support/Resistance features
    "dist_from_50d_high",
    "dist_from_50d_low", 
    "dist_from_52w_high",
    "support_bounce_count_20d",
    
    # ... rest of features ...
]
```

### Testing

```bash
# Run with ElasticNet to see which survive
USE_ELASTICNET_SELECT=1 ELASTICNET_L1_RATIO=0.5 python -c "
from prediction_model import predict_next_for_ticker
import json
result = predict_next_for_ticker('AAPL', model_type='xgb')
print('Prediction:', result['pred_next_ret'])
"
```

---

## Example 2: Adding Divergence Detection

### Code

```python
def add_price_features(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()
    close = hist["Close"]
    
    # ... existing code ...
    
    # === NEW: Divergence Features ===
    
    # RSI divergence: RSI going up but price going down (or vice versa)
    # This is a reversal signal
    
    rsi_5d_change = hist["rsi14"].diff(5)  # How much RSI changed in 5 days
    price_5d_change = close.pct_change(5)   # How much price changed in 5 days
    
    # Divergence = opposite directions
    # Positive = bullish divergence (RSI up, price down = reversal)
    hist["rsi_price_divergence_5d"] = (rsi_5d_change * -price_5d_change)  # Positive = divergence
    hist["rsi_price_divergence_5d"] = hist["rsi_price_divergence_5d"].shift(1)
    
    # MACD divergence (MACD strength vs price momentum)
    macd_5d_change = hist["macdhist"].diff(5) if "macdhist" in hist.columns else 0
    macd_price_div = (macd_5d_change * -price_5d_change)
    hist["macd_price_divergence_5d"] = macd_price_div
    hist["macd_price_divergence_5d"] = hist["macd_price_divergence_5d"].shift(1)
    
    # Volume divergence: volume not confirming price move
    # High price move but low volume = weak, likely reversal
    volume_5d_avg = hist["Volume"].rolling(5).mean()
    volume_5d_current = hist["Volume"].iloc[-1]
    volume_ratio = volume_5d_current / (volume_5d_avg + 1e-9)
    
    hist["volume_price_divergence"] = (volume_ratio < 1.0).astype(int)  # 1 = weak volume
    hist["volume_price_divergence"] = hist["volume_price_divergence"].shift(1)
    
    return hist
```

### Add to FEATURE_COLUMNS

```python
FEATURE_COLUMNS = [
    # ... existing ...
    
    # NEW Divergence features
    "rsi_price_divergence_5d",
    "macd_price_divergence_5d",
    "volume_price_divergence",
    
    # ... rest ...
]
```

---

## Example 3: Adding News Sentiment

### Step 1: Setup in data_fetch.py

```python
import requests
from datetime import datetime, timedelta

MARKETAUX_API_KEY = os.environ.get("MARKETAUX_API_KEY", None)

def get_news_sentiment(ticker: str, days: int = 7) -> dict:
    """
    Fetch news sentiment for a ticker from Marketaux API (free).
    Returns: {
        'sentiment_score': float (-1 to 1),
        'positive_count': int,
        'negative_count': int,
        'neutral_count': int,
        'article_count': int,
    }
    """
    if MARKETAUX_API_KEY is None:
        print(f"[get_news_sentiment] MARKETAUX_API_KEY not set; returning neutral sentiment")
        return {
            'sentiment_score': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'article_count': 0,
        }
    
    try:
        # Fetch recent news
        url = "https://api.marketaux.com/v1/news/all"
        params = {
            'symbols': ticker,
            'filter_entities': 'true',
            'limit': 100,
            'api_token': MARKETAUX_API_KEY,
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract sentiment
        articles = data.get('data', [])
        
        # Marketaux provides 'sentiment' field: 'Positive', 'Negative', 'Neutral'
        positive_count = sum(1 for a in articles if a.get('sentiment') == 'Positive')
        negative_count = sum(1 for a in articles if a.get('sentiment') == 'Negative')
        neutral_count = sum(1 for a in articles if a.get('sentiment') == 'Neutral')
        total = len(articles)
        
        # Calculate aggregate sentiment (-1 to 1)
        if total == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / total
        
        return {
            'sentiment_score': sentiment_score,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'article_count': total,
        }
    
    except Exception as e:
        print(f"[get_news_sentiment] Failed to fetch news for {ticker}: {e}")
        return {
            'sentiment_score': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'article_count': 0,
        }
```

### Step 2: Add to build_features_and_target()

```python
def build_features_and_target(
    ticker: str,
    period: str = "5y",
    horizon: int = 1,
    ...
):
    # ... existing code to get hist ...
    
    # === NEW: Add news sentiment ===
    sentiment_data = get_news_sentiment(ticker, days=7)
    hist["news_sentiment_7d"] = sentiment_data['sentiment_score']
    hist["news_positive_count_7d"] = sentiment_data['positive_count']
    hist["news_negative_count_7d"] = sentiment_data['negative_count']
    hist["news_article_count_7d"] = sentiment_data['article_count']
    
    # Forward-fill to match historical data
    # (sentiment is static, so fill all rows with same value)
    hist["news_sentiment_7d"] = hist["news_sentiment_7d"].fillna(method='ffill').fillna(0)
    hist["news_positive_count_7d"] = hist["news_positive_count_7d"].fillna(method='ffill').fillna(0)
    hist["news_negative_count_7d"] = hist["news_negative_count_7d"].fillna(method='ffill').fillna(0)
    hist["news_article_count_7d"] = hist["news_article_count_7d"].fillna(method='ffill').fillna(0)
    
    return hist
```

### Step 3: Add to FEATURE_COLUMNS

```python
FEATURE_COLUMNS = [
    # ... existing ...
    
    # NEW Sentiment features
    "news_sentiment_7d",
    "news_positive_count_7d",
    "news_negative_count_7d",
    "news_article_count_7d",
    
    # ... rest ...
]
```

### Step 4: Setup environment

```bash
export MARKETAUX_API_KEY="your_api_key_here"  # Get free from https://www.marketaux.com

# Test
python -c "from data_fetch import get_news_sentiment; print(get_news_sentiment('AAPL'))"
```

---

## Example 4: Adding More FRED Macro Data

### Step 1: Expand MACRO_COLUMNS

```python
MACRO_COLUMNS = [
    # Original
    "mkt_ret_1d",
    "term_spread",
    "t10y",
    "vix",
    
    # NEW
    "unrate",           # Unemployment rate
    "cpi_yoy",          # CPI year-over-year
    "dxy",              # Dollar index
    "hy_oas",           # High-yield OAS (credit spreads)
    "fed_funds_rate",   # Fed Funds effective rate
]
```

### Step 2: Expand get_macro_df()

```python
def get_macro_df(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Fetch macro data from FRED (requires FRED_API_KEY)"""
    
    if FRED_API_KEY is None:
        print("[get_macro_df] FRED_API_KEY not set; using only mkt_ret_1d")
        spx = _get_spx(start_date, end_date)
        if spx is not None:
            macro_df = pd.DataFrame(index=spx.index)
            macro_df["mkt_ret_1d"] = spx.pct_change().shift(1)
        else:
            macro_df = pd.DataFrame()
        return macro_df
    
    try:
        spx = _get_spx(start_date, end_date)
        
        # Fetch FRED series
        # FRED Series IDs:
        # UNRATE = Unemployment rate
        # CPIAUCSL = Consumer Price Index All Urban Consumers
        # DXYNONUS = US Dollar Index
        # BAMLH0A0HYM2 = ICE BofA US High Yield OAS
        # FEDFUNDS = Effective Federal Funds Rate
        
        fred_series = {
            'unrate': 'UNRATE',
            'cpi_yoy': 'CPIAUCSL',
            'dxy': 'DXYNONUS',
            'hy_oas': 'BAMLH0A0HYM2',
            'fed_funds_rate': 'FEDFUNDS',
        }
        
        macro_data = {}
        
        # Fetch each series
        for key, series_id in fred_series.items():
            try:
                s = get_fred_series(series_id, start_date, end_date)
                macro_data[key] = s
                print(f"[get_macro_df] Fetched {key} ({series_id})")
            except Exception as e:
                print(f"[get_macro_df] Failed to fetch {key}: {e}")
                macro_data[key] = None
        
        # Build dataframe
        macro_df = pd.DataFrame(index=spx.index) if spx is not None else pd.DataFrame()
        
        # Market return
        if spx is not None:
            macro_df["mkt_ret_1d"] = spx.pct_change().shift(1)
        
        # Term spread (10Y - 3M)
        if macro_data['cpi_yoy'] is not None and macro_data['dxy'] is not None:
            # This is simplified; normally you'd fetch 10Y and 3M rates separately
            s10 = get_fred_series("DGS10", start_date, end_date)
            s3m = get_fred_series("DGS3MO", start_date, end_date)
            macro_df["term_spread"] = (s10 - s3m) / 100  # Normalize
        
        # Add other FRED series
        for key in macro_series.keys():
            if macro_data[key] is not None:
                # Align to trading dates and forward-fill
                macro_df[key] = macro_data[key].reindex(macro_df.index, method='ffill')
        
        return macro_df.fillna(method='ffill').fillna(0)
    
    except Exception as e:
        print(f"[get_macro_df] Error: {e}")
        return pd.DataFrame()
```

---

## Testing Your New Features

### Step 1: Verify features are being created

```python
from prediction_model import build_features_and_target

# Build features for a test ticker
hist = build_features_and_target('AAPL', period='1y')

# Check which NEW features were created
new_features = [
    'dist_from_50d_high',
    'rsi_price_divergence_5d',
    'news_sentiment_7d',
    'unrate',
]

for feat in new_features:
    if feat in hist.columns:
        print(f"✓ {feat}: mean={hist[feat].mean():.3f}, std={hist[feat].std():.3f}")
    else:
        print(f"✗ {feat}: NOT FOUND")
```

### Step 2: Run prediction with ElasticNet feature selection

```bash
# Enable ElasticNet and run a prediction
USE_ELASTICNET_SELECT=1 \
ELASTICNET_L1_RATIO=0.5 \
ELASTICNET_CV_FOLDS=5 \
python -c "
from prediction_model import predict_next_for_ticker
import json

result = predict_next_for_ticker('AAPL', model_type='xgb')

# Print which features were selected
print('Selected Features:')
for feat in result.get('selected_features', []):
    print(f'  - {feat}')

print(f'\nPrediction: {result[\"pred_next_ret\"]:.4f}')
"
```

### Step 3: Compare model performance

```bash
# Old model (without new features)
python -c "from prediction_model import backtest_one_ticker; print(backtest_one_ticker('AAPL', period='2y'))" > old_model.txt

# New model (with new features, ElasticNet enabled)
USE_ELASTICNET_SELECT=1 python -c "from prediction_model import backtest_one_ticker; print(backtest_one_ticker('AAPL', period='2y'))" > new_model.txt

# Compare Sharpe ratios, returns, etc.
```

---

## Summary Checklist

- [ ] Added support/resistance features
- [ ] Added divergence detection features
- [ ] Setup Marketaux API for news sentiment
- [ ] Expanded FRED macro data (5+ series)
- [ ] Updated FEATURE_COLUMNS list
- [ ] Tested feature creation with `build_features_and_target()`
- [ ] Verified ElasticNet feature selection works
- [ ] Compared backtests (old vs new)
- [ ] Committed changes to git

---

## Expected Improvements

| Feature Addition | Expected Improvement |
|------------------|---------------------|
| Support/Resistance | +2-3% Sharpe |
| Divergence Detection | +1-2% Sharpe |
| News Sentiment | +3-5% Sharpe |
| Macro Expansion | +2-3% Sharpe |
| **Total** | **+8-13% Sharpe** |

Target: From Sharpe 0.8 → 0.9+
