"""
Gated Feature Network - Neural network that learns which features to use based on market context.

The key innovation: A gate network looks at market conditions (VIX, regime, volatility)
and outputs a weight (0-1) for each feature. High gate = feature important now, low gate = ignored.

Created: 2026-01-08
Author: Jakob McCleary
"""

import sys
import warnings
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA PREPARATION
# ============================================================================

# Main features (20 - optimized set)
MAIN_FEATURES = [
    "gbm_exp_ret_5d", "gbm_prob_up_5d", "ret_5d", "gbm_exp_ret_1d",
    "vol_20d", "rsi14", "gbm_prob_up_1d", "macd", "atr_14", "adx_14",
    "ret_1d", "ret_10d", "vol_10d", "obv", "momentum", "williams_r",
    "cci", "stoch_k", "bb_width", "mfi"
]

# Context features (5 - tells the network what market condition we're in)
CONTEXT_FEATURES = [
    "vix_normalized",      # VIX / 20
    "vix_change_5d",       # 5-day VIX change
    "spy_vs_ma200",        # (SPY - MA200) / MA200
    "regime_encoded",      # -1 bear, 0 neutral, 1 bull
    "vol_regime"           # current_vol / long_term_vol
]

TICKERS = ["AAPL", "MSFT", "AMZN"]


def load_and_prepare_data():
    """Load data for all tickers, add context features, and prepare for training."""
    from src.data import get_price_history
    
    print("=" * 70)
    print("PART 1: DATA PREPARATION")
    print("=" * 70)
    
    all_data = []
    
    for ticker in TICKERS:
        print(f"\nLoading {ticker}...")
        
        try:
            # Get raw price data and build features ourselves
            df = get_price_history(ticker, period="2y")
            if df is None or len(df) < 100:
                print(f"  ❌ Not enough data for {ticker}")
                continue
            
            print(f"  Raw shape: {df.shape}")
            
            # Build our own features (avoids the buggy pipeline)
            df = build_simple_features(df)
            
            # Add context features
            df = add_context_features(df, ticker)
            
            # Create binary target: 1 if next day return > 0, else 0
            df["target"] = (df["Close"].pct_change().shift(-1) > 0).astype(float)
            
            # Add ticker column
            df["ticker"] = ticker
            
            all_data.append(df)
            print(f"  ✅ Loaded {len(df)} rows with {len([c for c in df.columns if c in MAIN_FEATURES])} main features")
            
        except Exception as e:
            print(f"  ❌ Error loading {ticker}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    # Pool all data together
    pooled_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Pooled data: {len(pooled_df)} rows from {len(all_data)} tickers")
    
    return pooled_df


def build_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 20 main features from raw OHLCV data."""
    df = df.copy()
    
    # Ensure numeric types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Returns
    df["ret_1d"] = df["Close"].pct_change(1)
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_10d"] = df["Close"].pct_change(10)
    
    # Volatility
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    
    # GBM-based features
    df["gbm_mu_60d"] = df["ret_1d"].rolling(60).mean()
    df["gbm_sig_60d"] = df["ret_1d"].rolling(60).std()
    
    # GBM probability up (simplified)
    mu = df["gbm_mu_60d"]
    sig = df["gbm_sig_60d"].replace(0, 1e-6)
    df["gbm_prob_up_1d"] = 0.5 + 0.5 * np.tanh(mu / sig)
    df["gbm_prob_up_5d"] = 0.5 + 0.5 * np.tanh(5 * mu / sig)
    df["gbm_exp_ret_1d"] = mu
    df["gbm_exp_ret_5d"] = 5 * mu
    
    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    
    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    
    # ADX (simplified)
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    plus_di = 100 * plus_dm.rolling(14).mean() / df["atr_14"].replace(0, 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / df["atr_14"].replace(0, 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    df["adx_14"] = dx.rolling(14).mean()
    
    # OBV
    obv = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["obv"] = obv / obv.abs().max() if obv.abs().max() > 0 else obv  # Normalize
    
    # Momentum
    df["momentum"] = df["Close"] / df["Close"].shift(10) - 1
    
    # Williams %R
    high_14 = df["High"].rolling(14).max()
    low_14 = df["Low"].rolling(14).min()
    df["williams_r"] = -100 * (high_14 - df["Close"]) / (high_14 - low_14).replace(0, 1e-10)
    
    # CCI
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma_tp = tp.rolling(20).mean()
    md_tp = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df["cci"] = (tp - ma_tp) / (0.015 * md_tp).replace(0, 1e-10)
    
    # Stochastic %K
    df["stoch_k"] = 100 * (df["Close"] - low_14) / (high_14 - low_14).replace(0, 1e-10)
    
    # Bollinger Band Width
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["bb_width"] = (upper - lower) / ma20
    
    # Money Flow Index (MFI)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_mf = typical_price * df["Volume"]
    mf_pos = raw_mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    mf_neg = raw_mf.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mf_ratio = mf_pos / mf_neg.replace(0, 1e-10)
    df["mfi"] = 100 - (100 / (1 + mf_ratio))
    
    # Lag all features by 1 day to prevent look-ahead bias
    feature_cols = [c for c in MAIN_FEATURES if c in df.columns]
    for col in feature_cols:
        df[col] = df[col].shift(1)
    
    return df


def add_context_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add the 5 context features that tell the network about market conditions."""
    from src.data import get_price_history
    
    df = df.copy()
    
    # 1. VIX normalized (VIX / 20)
    if "vix" in df.columns:
        df["vix_normalized"] = df["vix"] / 20.0
    else:
        # Try to get VIX data
        try:
            vix_df = get_price_history("^VIX", period="2y")
            if vix_df is not None and len(vix_df) > 0:
                vix_df = vix_df[["Close"]].rename(columns={"Close": "vix"})
                vix_df.index = pd.to_datetime(vix_df.index)
                df.index = pd.to_datetime(df.index)
                df = df.join(vix_df, how="left")
                df["vix"] = df["vix"].ffill().fillna(20.0)
                df["vix_normalized"] = df["vix"] / 20.0
            else:
                df["vix_normalized"] = 1.0  # Default to VIX=20
        except:
            df["vix_normalized"] = 1.0
    
    # 2. VIX change 5d
    if "vix" in df.columns:
        df["vix_change_5d"] = df["vix"].pct_change(5).fillna(0)
    else:
        df["vix_change_5d"] = 0.0
    
    # 3. SPY vs MA200
    try:
        spy_df = get_price_history("SPY", period="2y")
        if spy_df is not None and len(spy_df) > 0:
            spy_df["spy_ma200"] = spy_df["Close"].rolling(200).mean()
            spy_df["spy_vs_ma200"] = (spy_df["Close"] - spy_df["spy_ma200"]) / spy_df["spy_ma200"]
            spy_df = spy_df[["spy_vs_ma200"]]
            spy_df.index = pd.to_datetime(spy_df.index)
            df.index = pd.to_datetime(df.index)
            df = df.join(spy_df, how="left")
            df["spy_vs_ma200"] = df["spy_vs_ma200"].ffill().fillna(0)
        else:
            df["spy_vs_ma200"] = 0.0
    except:
        df["spy_vs_ma200"] = 0.0
    
    # 4. Regime encoded (-1 bear, 0 neutral, 1 bull)
    if "regime_bull" in df.columns and "regime_bear" in df.columns:
        df["regime_encoded"] = df["regime_bull"].astype(float) - df["regime_bear"].astype(float)
    elif "spy_vs_ma200" in df.columns:
        # Infer regime from SPY vs MA200
        df["regime_encoded"] = np.where(df["spy_vs_ma200"] > 0.05, 1.0,
                                np.where(df["spy_vs_ma200"] < -0.05, -1.0, 0.0))
    else:
        df["regime_encoded"] = 0.0
    
    # 5. Volatility regime (current vol / long-term vol)
    if "vol_20d" in df.columns:
        long_term_vol = df["vol_20d"].rolling(60).mean()
        df["vol_regime"] = (df["vol_20d"] / long_term_vol).fillna(1.0)
    else:
        df["vol_regime"] = 1.0
    
    return df


def prepare_tensors(df: pd.DataFrame):
    """Prepare PyTorch tensors with train/val/test split."""
    
    # Check which features exist
    available_main = [f for f in MAIN_FEATURES if f in df.columns]
    available_context = [f for f in CONTEXT_FEATURES if f in df.columns]
    
    print(f"\n📋 Feature availability:")
    print(f"   Main features: {len(available_main)}/{len(MAIN_FEATURES)}")
    print(f"   Context features: {len(available_context)}/{len(CONTEXT_FEATURES)}")
    
    missing_main = [f for f in MAIN_FEATURES if f not in df.columns]
    missing_context = [f for f in CONTEXT_FEATURES if f not in df.columns]
    
    if missing_main:
        print(f"   ⚠️  Missing main: {missing_main}")
    if missing_context:
        print(f"   ⚠️  Missing context: {missing_context}")
    
    # Use available features
    feature_cols = available_main
    context_cols = available_context
    
    # Drop rows with NaN in key columns
    required_cols = feature_cols + context_cols + ["target"]
    df_clean = df.dropna(subset=[c for c in required_cols if c in df.columns])
    
    print(f"\n📊 Clean data: {len(df_clean)} rows (dropped {len(df) - len(df_clean)} with NaN)")
    
    # Extract arrays
    X_main = df_clean[feature_cols].values.astype(np.float32)
    X_context = df_clean[context_cols].values.astype(np.float32)
    y = df_clean["target"].values.astype(np.float32)
    
    # Time-based split (70/15/15)
    n = len(df_clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_main_train, X_main_val, X_main_test = X_main[:train_end], X_main[train_end:val_end], X_main[val_end:]
    X_ctx_train, X_ctx_val, X_ctx_test = X_context[:train_end], X_context[train_end:val_end], X_context[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    
    print(f"\n📊 Split sizes:")
    print(f"   Train: {len(y_train)} ({len(y_train)/n*100:.1f}%)")
    print(f"   Val:   {len(y_val)} ({len(y_val)/n*100:.1f}%)")
    print(f"   Test:  {len(y_test)} ({len(y_test)/n*100:.1f}%)")
    
    # Normalize - fit on train only
    main_scaler = StandardScaler()
    context_scaler = StandardScaler()
    
    X_main_train = main_scaler.fit_transform(X_main_train)
    X_main_val = main_scaler.transform(X_main_val)
    X_main_test = main_scaler.transform(X_main_test)
    
    X_ctx_train = context_scaler.fit_transform(X_ctx_train)
    X_ctx_val = context_scaler.transform(X_ctx_val)
    X_ctx_test = context_scaler.transform(X_ctx_test)
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    train_data = {
        "features": torch.FloatTensor(X_main_train).to(device),
        "context": torch.FloatTensor(X_ctx_train).to(device),
        "target": torch.FloatTensor(y_train).unsqueeze(1).to(device)
    }
    val_data = {
        "features": torch.FloatTensor(X_main_val).to(device),
        "context": torch.FloatTensor(X_ctx_val).to(device),
        "target": torch.FloatTensor(y_val).unsqueeze(1).to(device)
    }
    test_data = {
        "features": torch.FloatTensor(X_main_test).to(device),
        "context": torch.FloatTensor(X_ctx_test).to(device),
        "target": torch.FloatTensor(y_test).unsqueeze(1).to(device)
    }
    
    return train_data, val_data, test_data, feature_cols, context_cols, main_scaler, context_scaler


# ============================================================================
# PART 2: THE GATED FEATURE NETWORK
# ============================================================================

class GatedFeatureNetwork(nn.Module):
    """
    Neural network that learns which features to focus on based on current market context.
    
    The gate network looks at market conditions (VIX, regime, etc.)
    and outputs a weight (0-1) for each feature.
    
    High gate = feature is important right now
    Low gate = feature is ignored right now
    """
    
    def __init__(self, n_features=20, n_context=5):
        super().__init__()
        
        self.n_features = n_features
        self.n_context = n_context
        
        # GATE NETWORK: Context → Feature Gates
        # This is the key innovation - learns WHEN to use each feature
        self.gate_network = nn.Sequential(
            nn.Linear(n_context, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_features),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # PREDICTION NETWORK: Gated Features → Prediction
        self.prediction_network = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Probability of UP
        )
    
    def forward(self, features, context):
        """
        Args:
            features: [batch, n_features] - the stock features
            context: [batch, n_context] - market context (VIX, regime, etc.)
        
        Returns:
            prediction: [batch, 1] - probability of UP
            gates: [batch, n_features] - learned importance of each feature
        """
        # Step 1: Learn which features matter given current context
        gates = self.gate_network(context)  # [batch, n_features]
        
        # Step 2: Apply gates to features (element-wise multiply)
        gated_features = features * gates  # [batch, n_features]
        
        # Step 3: Make prediction using gated features
        prediction = self.prediction_network(gated_features)
        
        return prediction, gates
    
    def get_feature_importance(self, context):
        """Get which features the network thinks are important right now."""
        with torch.no_grad():
            gates = self.gate_network(context)
        return gates


class BaselineNetwork(nn.Module):
    """Simple MLP baseline without gating for comparison."""
    
    def __init__(self, n_features=20, n_context=5):
        super().__init__()
        
        # Concatenate features + context and predict
        self.network = nn.Sequential(
            nn.Linear(n_features + n_context, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, context):
        x = torch.cat([features, context], dim=1)
        prediction = self.network(x)
        return prediction, None  # No gates


# ============================================================================
# PART 3: TRAINING
# ============================================================================

def train_model(model, train_data, val_data, epochs=100, lr=0.001, patience=15, sparsity_weight=0.01):
    """
    Train with early stopping, learning rate scheduling, and sparsity regularization.
    
    The sparsity_weight controls L1 penalty on gates:
    - Higher = fewer features used (more selective)
    - Lower = more features used
    """
    
    device = next(model.parameters()).device
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "gate_sparsity": []}
    
    print(f"\n{'='*70}")
    print(f"Training {model.__class__.__name__}")
    print(f"{'='*70}")
    print(f"Sparsity weight: {sparsity_weight} (L1 penalty on gates)")
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        pred, gates = model(train_data["features"], train_data["context"])
        
        # Main loss: prediction accuracy
        pred_loss = criterion(pred, train_data["target"])
        
        # Regularization: encourage sparse gates (not all features used)
        # L1 penalty pushes gates toward 0, making feature selection more selective
        if gates is not None:
            sparsity_loss = sparsity_weight * torch.mean(torch.abs(gates))
            total_loss = pred_loss + sparsity_loss
            gate_sparsity = (gates < 0.1).float().mean().item()  # % of gates "off"
        else:
            total_loss = pred_loss
            gate_sparsity = 0.0
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, val_gates = model(val_data["features"], val_data["context"])
            val_loss = criterion(val_pred, val_data["target"])
        
        scheduler.step(val_loss)
        
        # Calculate accuracies
        train_acc = ((pred > 0.5).float() == train_data["target"]).float().mean().item()
        val_acc = ((val_pred > 0.5).float() == val_data["target"]).float().mean().item()
        
        history["train_loss"].append(total_loss.item())
        history["val_loss"].append(val_loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["gate_sparsity"].append(gate_sparsity)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            sparsity_str = f" | Gate Sparsity: {gate_sparsity:.1%}" if gates is not None else ""
            print(f"Epoch {epoch+1:3d} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}{sparsity_str}")
        
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history


# ============================================================================
# PART 4: EVALUATION
# ============================================================================

def evaluate_model(model, test_data, feature_names, model_name="Model"):
    """Comprehensive evaluation on test set."""
    
    model.eval()
    with torch.no_grad():
        pred, gates = model(test_data["features"], test_data["context"])
    
    y_true = test_data["target"].cpu().numpy().flatten()
    y_prob = pred.cpu().numpy().flatten()
    y_pred = (y_prob > 0.5).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5
    
    print(f"\n{'='*70}")
    print(f"TEST RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"   AUC-ROC:   {auc:.3f}")
    
    # Class distribution
    up_pct = y_true.mean() * 100
    print(f"\n   Class balance: {up_pct:.1f}% UP days in test set")
    print(f"   Predicted UP:  {y_pred.mean()*100:.1f}%")
    
    # Gate analysis (for GatedFeatureNetwork only)
    gate_importance = None
    if gates is not None:
        gates_np = gates.cpu().numpy()
        gate_importance = gates_np.mean(axis=0)
        
        print(f"\n{'='*70}")
        print(f"FEATURE GATE ANALYSIS")
        print(f"{'='*70}")
        print(f"\nAverage gate values (higher = more important):")
        
        # Sort by importance
        sorted_idx = np.argsort(gate_importance)[::-1]
        for i, idx in enumerate(sorted_idx):
            if idx < len(feature_names):
                bar = "█" * int(gate_importance[idx] * 20)
                print(f"   {i+1:2d}. {feature_names[idx]:20s} {gate_importance[idx]:.3f} {bar}")
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "gate_importance": gate_importance
    }
    
    return results


def analyze_gates_by_regime(model, test_data, feature_names, context_names):
    """Analyze how gates change based on market regime."""
    
    if not hasattr(model, 'gate_network'):
        print("Model doesn't have gate network")
        return
    
    model.eval()
    with torch.no_grad():
        _, gates = model(test_data["features"], test_data["context"])
    
    gates_np = gates.cpu().numpy()
    context_np = test_data["context"].cpu().numpy()
    
    # Find regime column
    regime_idx = None
    for i, name in enumerate(context_names):
        if "regime" in name.lower():
            regime_idx = i
            break
    
    if regime_idx is None:
        print("No regime column found in context")
        return
    
    regimes = context_np[:, regime_idx]
    
    print(f"\n{'='*70}")
    print(f"GATE VALUES BY MARKET REGIME")
    print(f"{'='*70}")
    
    regime_labels = {-1: "🐻 BEAR", 0: "➖ NEUTRAL", 1: "🐂 BULL"}
    
    for regime_val, label in regime_labels.items():
        mask = np.abs(regimes - regime_val) < 0.5
        if mask.sum() == 0:
            continue
        
        regime_gates = gates_np[mask].mean(axis=0)
        
        print(f"\n{label} Market ({mask.sum()} samples):")
        sorted_idx = np.argsort(regime_gates)[::-1][:5]  # Top 5
        for idx in sorted_idx:
            if idx < len(feature_names):
                print(f"   {feature_names[idx]:20s} {regime_gates[idx]:.3f}")


def plot_results(gated_history, baseline_history, gated_results, baseline_results, feature_names, save_path=None):
    """Create visualization of results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training curves
    ax1 = axes[0, 0]
    ax1.plot(gated_history["val_loss"], label="Gated - Val Loss", color="blue")
    ax1.plot(baseline_history["val_loss"], label="Baseline - Val Loss", color="red", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Validation Loss During Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy comparison
    ax2 = axes[0, 1]
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x = np.arange(len(metrics))
    width = 0.35
    
    gated_vals = [gated_results[m] for m in metrics]
    baseline_vals = [baseline_results[m] for m in metrics]
    
    ax2.bar(x - width/2, gated_vals, width, label="Gated Network", color="blue", alpha=0.7)
    ax2.bar(x + width/2, baseline_vals, width, label="Baseline MLP", color="red", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in metrics])
    ax2.set_ylabel("Score")
    ax2.set_title("Test Set Performance Comparison")
    ax2.legend()
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Feature importance from gates
    ax3 = axes[1, 0]
    if gated_results["gate_importance"] is not None:
        importance = gated_results["gate_importance"]
        sorted_idx = np.argsort(importance)[::-1][:10]  # Top 10
        
        names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in sorted_idx]
        vals = [importance[i] for i in sorted_idx]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(vals)))
        ax3.barh(range(len(names)), vals, color=colors)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names)
        ax3.invert_yaxis()
        ax3.set_xlabel("Average Gate Value")
        ax3.set_title("Top 10 Features by Gate Importance")
        ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Training accuracy
    ax4 = axes[1, 1]
    ax4.plot(gated_history["val_acc"], label="Gated - Val Acc", color="blue")
    ax4.plot(baseline_history["val_acc"], label="Baseline - Val Acc", color="red", linestyle="--")
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="Random")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy")
    ax4.set_title("Validation Accuracy During Training")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the complete Gated Feature Network experiment."""
    
    print("\n" + "="*70)
    print("🧠 GATED FEATURE NETWORK EXPERIMENT")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tickers: {TICKERS}")
    print(f"Main features: {len(MAIN_FEATURES)}")
    print(f"Context features: {len(CONTEXT_FEATURES)}")
    
    # Load and prepare data
    df = load_and_prepare_data()
    train_data, val_data, test_data, feature_names, context_names, main_scaler, context_scaler = prepare_tensors(df)
    
    n_features = len(feature_names)
    n_context = len(context_names)
    
    print(f"\n✅ Data prepared:")
    print(f"   Features: {n_features}")
    print(f"   Context: {n_context}")
    
    device = train_data["features"].device
    
    # Create models
    gated_model = GatedFeatureNetwork(n_features=n_features, n_context=n_context).to(device)
    baseline_model = BaselineNetwork(n_features=n_features, n_context=n_context).to(device)
    
    print(f"\n📐 Model sizes:")
    print(f"   Gated:    {sum(p.numel() for p in gated_model.parameters()):,} parameters")
    print(f"   Baseline: {sum(p.numel() for p in baseline_model.parameters()):,} parameters")
    
    # Train both models
    gated_model, gated_history = train_model(gated_model, train_data, val_data, epochs=150, lr=0.001, patience=20)
    baseline_model, baseline_history = train_model(baseline_model, train_data, val_data, epochs=150, lr=0.001, patience=20)
    
    # Evaluate both models
    print("\n" + "="*70)
    print("EVALUATION ON HOLDOUT TEST SET")
    print("="*70)
    
    gated_results = evaluate_model(gated_model, test_data, feature_names, "Gated Feature Network")
    baseline_results = evaluate_model(baseline_model, test_data, feature_names, "Baseline MLP")
    
    # Analyze gates by regime
    analyze_gates_by_regime(gated_model, test_data, feature_names, context_names)
    
    # Summary comparison
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<15} {'Gated':>12} {'Baseline':>12} {'Winner':>12}")
    print("-" * 55)
    
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    gated_score = 0
    baseline_score = 0
    
    for m in metrics:
        g = gated_results[m]
        b = baseline_results[m]
        winner = "GATED" if g > b else ("BASELINE" if b > g else "TIE")
        if g > b:
            gated_score += 1
        elif b > g:
            baseline_score += 1
        print(f"{m.upper():<15} {g:>12.3f} {b:>12.3f} {winner:>12}")
    
    print("-" * 55)
    print(f"{'SCORE':<15} {gated_score:>12} {baseline_score:>12}")
    
    # Winner announcement
    print("\n" + "="*70)
    if gated_score > baseline_score:
        print("🏆 WINNER: Gated Feature Network")
        print("   The gating mechanism provides value by learning context-aware features!")
    elif baseline_score > gated_score:
        print("🏆 WINNER: Baseline MLP")
        print("   Simple concatenation outperforms gating on this dataset.")
    else:
        print("🏆 TIE: Both models perform similarly")
    print("="*70)
    
    # Save plot
    plot_path = Path(__file__).parent / "gated_network_results.png"
    try:
        plot_results(gated_history, baseline_history, gated_results, baseline_results, feature_names, save_path=str(plot_path))
    except Exception as e:
        print(f"\n⚠️  Could not create plot: {e}")
    
    # Save model
    model_path = Path(__file__).parent / "gated_feature_network.pt"
    torch.save({
        "model_state_dict": gated_model.state_dict(),
        "feature_names": feature_names,
        "context_names": context_names,
        "n_features": n_features,
        "n_context": n_context,
        "results": {k: v for k, v in gated_results.items() if not isinstance(v, np.ndarray)}
    }, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    print(f"\n✅ Experiment complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return gated_model, baseline_model, gated_results, baseline_results


if __name__ == "__main__":
    main()
