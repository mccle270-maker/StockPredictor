"""
Accuracy Diagnostics - Identify why accuracy is ~46% despite Sharpe 2.51

Key Questions:
1. How many features actually help? (feature importance)
2. Is the model just getting lucky? (confidence calibration)
3. Can we improve accuracy with different horizons? (2d, 5d vs 1d)
4. How much accuracy comes from position sizing vs directional calls?
5. Are there high-confidence predictions we can filter for? (60%+ accuracy subset)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediction_model import build_features_and_target, make_model, FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class AccuracyDiagnostics:
    def __init__(self, ticker, period="5y", model_type="rf", horizon=1):
        self.ticker = ticker
        self.period = period
        self.model_type = model_type
        self.horizon = horizon
        self.X = None
        self.y = None
        self.feature_names = FEATURE_COLUMNS
        self.model = None
        self.results = {}
        
    def load_data(self):
        """Load features and target"""
        logger.info(f"Loading {self.ticker} ({self.horizon}d horizon)...")
        try:
            X, y, _, _, _, _, _ = build_features_and_target(
                self.ticker, period=self.period, horizon=self.horizon, run_gaf=False
            )
            self.X = X
            self.y = y
            logger.info(f"✅ Loaded {len(X)} samples, {len(FEATURE_COLUMNS)} features")
            return True
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            return False
    
    def analyze_target(self):
        """Analyze target distribution"""
        logger.info("\n📊 TARGET ANALYSIS")
        logger.info("=" * 80)
        pos = (self.y > 0).sum()
        total = len(self.y)
        logger.info(f"  Total samples: {total}")
        logger.info(f"  Up moves: {pos} ({pos/total*100:.1f}%)")
        logger.info(f"  Down moves: {total-pos} ({(total-pos)/total*100:.1f}%)")
        logger.info(f"  Mean return: {self.y.mean():.6f}")
        logger.info(f"  Std return: {self.y.std():.6f}")
        return pos, total
    
    def train_evaluate(self):
        """Train and evaluate"""
        logger.info("\n🤖 MODEL TRAINING")
        logger.info("=" * 80)
        split = int(0.8 * len(self.X))
        X_tr, X_te = self.X[:split], self.X[split:]
        y_tr, y_te = self.y[:split], self.y[split:]
        
        logger.info(f"  Train: {len(X_tr)} | Test: {len(X_te)}")
        
        self.model = make_model(self.model_type, task='regression')
        self.model.fit(X_tr, y_tr)
        
        y_tr_pred = self.model.predict(X_tr)
        y_te_pred = self.model.predict(X_te)
        
        # Accuracy
        tr_acc = (np.sign(y_tr_pred) == np.sign(y_tr)).mean()
        te_acc = (np.sign(y_te_pred) == np.sign(y_te)).mean()
        
        logger.info(f"\n  Train Accuracy: {tr_acc*100:.1f}%")
        logger.info(f"  Test Accuracy: {te_acc*100:.1f}%")
        logger.info(f"  Overfit gap: {(tr_acc-te_acc)*100:.1f}%")
        
        # Prediction strength
        logger.info(f"\n  Pred Mean: {y_te_pred.mean():.6f}")
        logger.info(f"  Pred Std: {y_te_pred.std():.6f}")
        logger.info(f"  Positive preds: {(y_te_pred > 0).sum() / len(y_te_pred) * 100:.1f}%")
        
        return X_tr, X_te, y_tr, y_te, y_tr_pred, y_te_pred, te_acc
    
    def analyze_confidence(self, y_te, y_te_pred):
        """Accuracy vs confidence thresholds"""
        logger.info("\n💡 ACCURACY BY CONFIDENCE")
        logger.info("=" * 80)
        confidence = np.abs(y_te_pred)
        correct = (np.sign(y_te_pred) == np.sign(y_te))
        
        logger.info(f"{'Min Conf':<12} {'Count':<10} {'Accuracy':<12} {'% Data':<12}")
        logger.info("-" * 50)
        for thresh in [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]:
            mask = confidence >= thresh
            if mask.sum() > 0:
                acc = correct[mask].mean()
                pct = mask.sum() / len(mask) * 100
                logger.info(f"{thresh:<12.5f} {mask.sum():<10} {acc*100:>10.1f}% {pct:>10.1f}%")
        
        return confidence
    
    def analyze_importance(self, X_tr):
        """Feature importance"""
        logger.info("\n🔍 TOP FEATURES BY IMPORTANCE")
        logger.info("=" * 80)
        if hasattr(self.model, 'feature_importances_'):
            imp = self.model.feature_importances_
            # Use actual feature count from model, not pre-defined
            num_features = len(imp)
            feature_labels = [f"feat_{i}" for i in range(num_features)] if len(self.feature_names) != num_features else self.feature_names
            df_imp = pd.DataFrame({
                'feature': feature_labels,
                'importance': imp
            }).sort_values('importance', ascending=False)
            
            logger.info(f"{'Feature':<40} {'Importance':<15}")
            logger.info("-" * 55)
            for _, row in df_imp.head(15).iterrows():
                logger.info(f"{row['feature']:<40} {row['importance']:>14.6f}")
            
            top5_pct = df_imp.head(5)['importance'].sum() / df_imp['importance'].sum() * 100
            logger.info(f"\nTop 5 explain: {top5_pct:.1f}% of importance")
            
            return df_imp
        else:
            logger.warning("Model has no feature_importances_")
            return None
    
    def run(self):
        """Execute all diagnostics"""
        logger.info(f"\n{'='*80}")
        logger.info(f"DIAGNOSTICS: {self.ticker} | {self.model_type.upper()} | {self.horizon}d")
        logger.info(f"{'='*80}")
        
        if not self.load_data():
            return {}
        
        self.analyze_target()
        X_tr, X_te, y_tr, y_te, y_tr_pred, y_te_pred, te_acc = self.train_evaluate()
        self.analyze_confidence(y_te, y_te_pred)
        df_imp = self.analyze_importance(X_tr)
        
        self.results = {
            'ticker': self.ticker,
            'model': self.model_type,
            'horizon': self.horizon,
            'test_accuracy': float(te_acc),
        }
        
        if df_imp is not None:
            self.results['top_5_features'] = df_imp.head(5)['feature'].tolist()
        
        return self.results


def test_horizons():
    """Test if different prediction horizons improve accuracy"""
    logger.info("\n\n" + "#"*80)
    logger.info("# HORIZON OPTIMIZATION - Testing 1d, 2d, 5d horizons")
    logger.info("#"*80)
    
    for ticker in ["GLD", "SPY"]:
        logger.info(f"\n\n{ticker}")
        logger.info("=" * 50)
        for horizon in [1, 2, 5]:
            diag = AccuracyDiagnostics(ticker, model_type="rf", horizon=horizon)
            res = diag.run()
            if res:
                logger.info(f"✅ {horizon}d: Accuracy {res.get('test_accuracy', 0)*100:.1f}%")


def main():
    """Main diagnostic suite"""
    logger.info("\n" + "="*80)
    logger.info("ACCURACY IMPROVEMENT DIAGNOSTICS")
    logger.info("="*80)
    
    # Basic diagnostics
    for ticker in ["GLD", "SPY"]:
        for model in ["rf", "xgb"]:
            diag = AccuracyDiagnostics(ticker, model_type=model)
            diag.run()
    
    # Horizon optimization
    test_horizons()
    
    logger.info("\n\n" + "="*80)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
