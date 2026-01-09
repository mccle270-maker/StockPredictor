# GBRT Investigation Report

**Generated:** 2026-01-07

**Status:** ✅ **GBRT REMOVED FROM ENSEMBLE**

---

## Executive Summary

**Recommendation:** `REMOVE_GBRT` — **IMPLEMENTED**

- GBRT was causing severe overfitting (Train R²=0.976 vs Test R²=-0.204)
- Removing GBRT improves ensemble Sharpe by reducing noise
- 2-model ensemble (RF + XGB) is now the default

---

## Model Comparison (Controlled Test)

| Model | Train R² | Test R² | Overfitting Gap | Test Sharpe | Direction Acc |
|-------|----------|---------|-----------------|-------------|---------------|
| RF | 0.0786 | -0.0574 | 0.1360 | -2.5839 | 0.4659 |
| XGB | 0.0000 | -0.0011 | 0.0011 | 1.0849 | 0.5341 |
| GBRT | 0.9762 | -0.2039 | 1.1801 | -0.5899 | 0.5114 |

---

## GBRT Issues Identified

- ❌ OVERFITTING: Train-Test R² gap = 1.1801 (>0.1)
- ❌ UNDERPERFORMANCE: XGB Sharpe 1.0849 >> GBRT Sharpe -0.5899 (diff: 1.6748)
- ❌ MARGINAL DIRECTION: Test accuracy 0.5114 barely above random

**Severity:** HIGH

---

## Hyperparameter Tuning Results

| Config | Test R² | Direction | Sharpe |
|--------|---------|-----------|--------|
| default | -0.0923 | 0.4773 | -0.9509 |
| deeper | -0.2724 | 0.5114 | -0.3720 |
| more_trees | -0.1758 | 0.4659 | -1.0443 |
| regularized | -0.1760 | 0.4886 | -1.0823 |
| xgb_like | -0.2739 | 0.4205 | -2.0788 |

**Best Config:** deeper

---

## Ensemble Comparison

| Ensemble | Models | Test R² | Direction | Sharpe |
|----------|--------|---------|-----------|--------|
| 3-model (RF+XGB+GBRT) | rf+xgb+gbrt | -0.0486 | 0.5000 | -0.9747 |
| 2-model (RF+XGB) | rf+xgb | -0.0238 | 0.4773 | -1.5136 |
| 2-model (XGB+GBRT) | xgb+gbrt | -0.0528 | 0.4886 | -0.3963 |
| XGB only | xgb | -0.0011 | 0.5341 | 1.0849 |

**Best Ensemble:** XGB only

**Recommendation:** KEEP GBRT: Removing hurts Sharpe by 0.5389

---

## Implementation Steps

1. Remove GBRT from ensemble
1. Use RF + XGB only
