# Sector-Specific Model Implementation Guide

Quick reference for implementing sector-specific configurations in production.

## 1. Sector Detection Helper

```python
# Add to prediction_model.py

def get_sector(ticker):
    """Map ticker to sector"""
    sectors = {
        # Energy
        'CVX': 'energy', 'XOM': 'energy',
        # Industrials
        'CAT': 'industrials',
        # Consumer
        'PG': 'consumer', 'WMT': 'consumer', 'KO': 'consumer',
        # Tech
        'MSFT': 'tech', 'META': 'tech', 'NVDA': 'tech', 'AAPL': 'tech', 'AMD': 'tech',
        # Finance
        'JPM': 'finance', 'GS': 'finance', 'BAC': 'finance', 'C': 'finance', 'WFC': 'finance',
        # Healthcare
        'JNJ': 'healthcare', 'PFE': 'healthcare', 'UNH': 'healthcare', 'ABBV': 'healthcare', 'MRK': 'healthcare'
    }
    return sectors.get(ticker.upper(), 'unknown')


def get_sector_config(sector):
    """Return model configuration for sector"""
    configs = {
        'energy': {
            'use_ensemble': True,
            'use_classification': False,
            'enable_threshold_optimization': True,
            'enable_volatility_weighting': True,
            'enable_position_holding': True,
            'position_holding_days': 3,
        },
        'industrials': {
            'use_ensemble': True,
            'use_classification': False,
            'enable_threshold_optimization': True,
            'enable_volatility_weighting': True,
            'enable_position_holding': True,
            'position_holding_days': 3,
        },
        'consumer': {
            'use_ensemble': True,
            'use_classification': False,
            'enable_threshold_optimization': True,
            'enable_volatility_weighting': True,
            'enable_position_holding': True,
            'position_holding_days': 3,
        },
        'tech': {
            'use_ensemble': False,  # CRITICAL: No ensemble
            'use_classification': False,
            'enable_threshold_optimization': False,  # CRITICAL: No threshold
            'enable_volatility_weighting': False,    # CRITICAL: No vol weight
            'enable_position_holding': False,        # CRITICAL: No holding
            'position_holding_days': 1,
        },
        'finance': {
            'use_ensemble': False,  # CRITICAL: No ensemble
            'use_classification': False,
            'enable_threshold_optimization': False,  # CRITICAL: No threshold
            'enable_volatility_weighting': False,    # CRITICAL: No vol weight
            'enable_position_holding': False,        # CRITICAL: No holding
            'position_holding_days': 1,
        },
        'healthcare': {
            'use_ensemble': True,        # YES
            'use_classification': True,  # CRITICAL: Use classification
            'enable_threshold_optimization': False,  # CRITICAL: NO threshold
            'enable_volatility_weighting': True,     # YES
            'enable_position_holding': True,
            'position_holding_days': 7,  # CRITICAL: 7 days
        },
    }
    return configs.get(sector, configs['energy'])  # Default to energy config
```

## 2. Modified predict_next_for_ticker()

```python
def predict_next_for_ticker(
    tk, 
    period='5y', 
    model_type='rf', 
    horizon=1,
    auto_select_model=False,
    **kwargs
):
    """
    Predict next-period return for ticker with sector-specific config.
    """
    
    # Get sector config
    sector = get_sector(tk)
    sector_config = get_sector_config(sector)
    
    # Merge sector config with kwargs (kwargs override sector config)
    config = {**sector_config, **kwargs}
    
    # ... rest of existing code ...
    # Use 'config' dict for all boolean flags
    
    return {
        'ticker': tk,
        'sector': sector,
        'pred_next_ret': pred_ret,
        'pred_next_price': pred_price,
        'prob_up': prob_up,
        'prob_down': prob_down,
        'sector_config': {
            'ensemble': config['use_ensemble'],
            'classification': config['use_classification'],
            'position_holding_days': config['position_holding_days']
        },
        # ... other return fields ...
    }
```

## 3. Updated app.py Display

```python
# In app.py, when displaying predictions:

if 'sector_config' in pred:
    sector = pred.get('sector', 'unknown').upper()
    st.markdown(f"**Sector**: {sector}")
    
    config = pred['sector_config']
    if sector == 'HEALTHCARE':
        st.success("🔧 Healthcare Config: Classification + 7-day holding (NO threshold)")
    elif sector in ['TECH', 'FINANCE']:
        st.warning(f"⚠️  {sector} Config: Simple baseline (no improvements)")
    else:
        st.info("✅ Stable sector: Full ensemble + threshold optimization")
```

## 4. For Batch Processing (auto_paper_trade.py)

```python
# In auto_paper_trade.py or runner.py:

SECTOR_CONFIG_MAP = {
    'energy': 'full_improvements',
    'industrials': 'full_improvements',
    'consumer': 'full_improvements',
    'tech': 'baseline_only',
    'finance': 'baseline_only',
    'healthcare': 'classification_7day',
}

# When processing trades:
for ticker in tickers:
    sector = get_sector(ticker)
    config_type = SECTOR_CONFIG_MAP[sector]
    
    # Process with appropriate configuration
    pred = predict_next_for_ticker(
        ticker,
        **get_sector_config(sector)
    )
    
    # Generate signals with sector config
    if config_type == 'baseline_only':
        # Stricter filters for tech/finance
        signal = build_signals_from_pred_df(pred, threshold=0.01)
    else:
        # Normal threshold for other sectors
        signal = build_signals_from_pred_df(pred, threshold=None)
```

## 5. Configuration Matrix (Copy-Paste Reference)

```
SECTOR          | ENSEMBLE | CLASSIF | THRESHOLD | VOL_WEIGHT | HOLDING | HOLD_DAYS
----------------|----------|---------|-----------|-----------|---------|----------
Energy          | YES      | NO      | YES       | YES       | YES     | 3
Industrials     | YES      | NO      | YES       | YES       | YES     | 3
Consumer        | YES      | NO      | YES       | YES       | YES     | 3
Tech            | NO       | NO      | NO        | NO        | NO      | 1
Finance         | NO       | NO      | NO        | NO        | NO      | 1
Healthcare      | YES      | YES     | NO        | YES       | YES     | 7
```

## 6. Validation Script

```python
# Test script to verify sector configs are working

def validate_sector_configs():
    """Validate all sector configs work correctly"""
    
    test_tickers = {
        'energy': ['CVX', 'XOM'],
        'industrials': ['CAT'],
        'consumer': ['PG', 'WMT', 'KO'],
        'tech': ['MSFT', 'META'],
        'finance': ['JPM', 'BAC'],
        'healthcare': ['JNJ', 'PFE']
    }
    
    for sector, tickers in test_tickers.items():
        print(f"\n{'='*60}")
        print(f"Testing {sector.upper()} sector: {tickers}")
        print(f"{'='*60}")
        
        for ticker in tickers:
            try:
                pred = predict_next_for_ticker(ticker)
                
                # Verify sector detection
                detected_sector = pred.get('sector')
                assert detected_sector == sector, f"Sector mismatch: expected {sector}, got {detected_sector}"
                
                # Verify config applied
                config = pred.get('sector_config', {})
                print(f"  {ticker}: ✅ Sector={sector}, Config={config}")
                
            except Exception as e:
                print(f"  {ticker}: ❌ ERROR - {str(e)}")
                
    print(f"\n{'='*60}")
    print("All sector configs validated!")

# Run validation
if __name__ == '__main__':
    validate_sector_configs()
```

## 7. Monitoring Dashboard Update

For Streamlit dashboard, add sector-specific monitoring:

```python
# In app.py monitoring section:

def show_sector_performance():
    """Display performance by sector"""
    
    st.subheader("Sector Performance Summary")
    
    sectors = ['energy', 'industrials', 'consumer', 'tech', 'finance', 'healthcare']
    sector_stats = {}
    
    for sector in sectors:
        tickers = [tk for tk, s in SECTOR_MAP.items() if s == sector]
        
        if tickers:
            sharpes = []
            for ticker in tickers:
                # Get recent backtest results
                try:
                    result = backtest_one_ticker(ticker, period='1y')
                    sharpes.append(result.get('sharpe', 0))
                except:
                    pass
            
            if sharpes:
                avg_sharpe = np.mean(sharpes)
                config = get_sector_config(sector)
                
                # Color code by performance
                if avg_sharpe > 0.5:
                    emoji = "✅"
                    color = "green"
                elif avg_sharpe > 0:
                    emoji = "⚠️"
                    color = "yellow"
                else:
                    emoji = "❌"
                    color = "red"
                
                st.write(f"{emoji} **{sector.upper()}**: Sharpe={avg_sharpe:.2f}, "
                        f"Tickers={len(tickers)}, Config={'Ensemble' if config['use_ensemble'] else 'Baseline'}")
```

## 8. Testing Checklist Before Deployment

```python
# Checklist items:

checklist = {
    'Sector Detection': [
        '[ ] get_sector() returns correct sector for all 20 tickers',
        '[ ] Unknown tickers default to energy (safe default)',
        '[ ] Case-insensitive (works with 'aapl', 'AAPL', 'Aapl')'
    ],
    'Configuration Application': [
        '[ ] Energy sector uses ensemble=True',
        '[ ] Tech sector uses ensemble=False',
        '[ ] Healthcare uses classification=True',
        '[ ] Tech/Finance don\'t use threshold optimization',
        '[ ] Healthcare uses 7-day holding'
    ],
    'Integration': [
        '[ ] predict_next_for_ticker() returns sector in response',
        '[ ] app.py displays sector config correctly',
        '[ ] auto_paper_trade.py uses sector configs',
        '[ ] Prediction accuracy reported by sector'
    ],
    'Performance': [
        '[ ] No errors on full 20-ticker suite',
        '[ ] Execution time < 5 seconds per ticker',
        '[ ] Memory usage stable (no leaks)',
        '[ ] Cache working correctly'
    ]
}
```

## 9. Rollback Plan

If sector-specific config causes issues:

```python
# Rollback to original (all sectors use Phase 2 config):
def use_universal_config():
    """Fallback: use single config for all sectors"""
    
    return {
        'use_ensemble': True,
        'use_classification': False,
        'enable_threshold_optimization': True,
        'enable_volatility_weighting': True,
        'enable_position_holding': True,
        'position_holding_days': 3,
    }

# Quick switch in code:
# config = get_sector_config(sector)  # <-- Comment this
# config = use_universal_config()      # <-- Use this instead
```

## 10. Monitoring Metrics by Sector

Track these metrics post-deployment:

```
ENERGY:
  - Target Sharpe: 0.7+
  - Alert if: < 0.3 or > 2.0 (overfitting)
  
INDUSTRIALS:
  - Target Sharpe: 1.4+
  - Alert if: < 0.8 or > 2.5
  
CONSUMER:
  - Target Sharpe: 0.3-0.5
  - Alert if: < -0.2 or > 1.0
  
TECH:
  - Target Sharpe: -1.4 to -1.0 (negative OK)
  - Alert if: < -2.0 (something wrong) or > 0.0 (luck?)
  
FINANCE:
  - Target Sharpe: -0.5 to 0.0
  - Alert if: < -1.5 (something wrong) or > 0.5 (luck?)
  
HEALTHCARE:
  - Target Sharpe: 0.0 to +0.5
  - Alert if: < -0.5 (revert) or > 1.0 (overfitting)
```

---

**Summary**: Implement sector detection → apply config → test → deploy → monitor by sector.

Priority: Healthcare (new config), then Tech/Finance (disable improvements), then monitor Energy/Industrials/Consumer for stability.
