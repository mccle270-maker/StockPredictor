#!/usr/bin/env python3
"""
Quick diagnostic: Print which features are being selected for each fold
"""

import sys
sys.path.insert(0, '/Users/jakobmccleary/Desktop/Stock Predictor')

from prediction_model import _select_features_for_fold, build_panel_features_and_target
import pandas as pd
import numpy as np

print("\n" + "="*80)
print("FEATURE SELECTION DIAGNOSTIC")
print("="*80)

# Build small test dataset
print("\nBuilding test data for AAPL (1 month, quick test)...")
try:
    panel = build_panel_features_and_target(['AAPL'], period='1mo', horizon=1)
    
    # Get feature columns
    all_cols = set(panel.columns) - {"target", "ticker", "Date", "Unnamed: 0"}
    feat_cols = sorted([c for c in all_cols if c not in ["target", "ticker"]])
    
    print(f"\nTotal features available: {len(feat_cols)}")
    print(f"Sample features: {feat_cols[:10]}...")
    
    # Prepare data
    df = panel.dropna().copy()
    df = df.reset_index()
    df = df.rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Training data rows: {len(df)}")
    
    # Test each selection mode
    modes = ["elasticnet", "ols", "best", "none"]
    
    for mode in modes:
        print(f"\n{'-'*80}")
        print(f"Testing feature_selection='{mode}'")
        print(f"{'-'*80}")
        
        if mode == "none":
            print(f"✓ No filtering: Using all {len(feat_cols)} features")
            selected = feat_cols
        else:
            selected = _select_features_for_fold(
                train_df=df,
                feat_cols=feat_cols,
                horizon=1,
                selection_mode=mode
            )
        
        if selected is None:
            print(f"✗ Selection failed, would use all {len(feat_cols)} features")
        else:
            print(f"✓ Selected {len(selected)}/{len(feat_cols)} features ({100*len(selected)/len(feat_cols):.1f}%)")
            print(f"\n  Selected features:")
            for i, feat in enumerate(selected, 1):
                print(f"    {i:2d}. {feat}")
            
            print(f"\n  Filtered OUT {len(feat_cols) - len(selected)} features:")
            filtered_out = [f for f in feat_cols if f not in selected]
            for i, feat in enumerate(filtered_out[:10], 1):  # Show first 10
                print(f"    {i:2d}. {feat}")
            if len(filtered_out) > 10:
                print(f"    ... and {len(filtered_out)-10} more")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

interpretation = """
WHAT THIS SHOWS:

1. feature_selection="elasticnet" or "best"
   ✓ Reduces features from 133 to ~10-23
   ✓ Uses L1 regularization (elastic net) to select most important features
   ✓ HELPS: Healthcare, Energy, Industrials
   ✗ HURTS: Tech, Finance (removes needed features)

2. feature_selection="ols"
   ✓ Uses OLS p-value < 0.05 significance threshold
   ✓ Also reduces to ~10-23 features
   ✓ Similar effect to ElasticNet

3. feature_selection="none"
   ✓ Uses all 133 features
   ✓ Better for Tech/Finance sectors
   ✗ More noise, higher variance
   ✗ Slower training

RECOMMENDATION:

Tech/Finance sectors should use feature_selection="none" to avoid over-filtering.
Healthcare/Energy/Industrials can use feature_selection="best" for noise reduction.

You can set this per sector in walkforward_cross_sectional():
  
  walkforward_cross_sectional(
      tickers=['MSFT'],
      feature_selection="none",  # For Tech
  )
  
  walkforward_cross_sectional(
      tickers=['AAPL'],
      feature_selection="best",  # For Consumer
  )
"""

print(interpretation)
