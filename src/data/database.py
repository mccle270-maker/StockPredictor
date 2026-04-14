"""
Database Module for Stock Predictor
====================================

SQLite-based storage for:
- Prediction history (queries, outputs, accuracy tracking)
- Model performance metrics over time
- Backtest results for comparison
- User query history for quick re-runs

This is an optional enhancement - the dashboard works without it.
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from contextlib import contextmanager

# Database path (in project root)
DB_PATH = Path(__file__).parent.parent.parent / "stock_predictor.db"


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """
    Initialize the database with required tables.
    Safe to call multiple times - uses IF NOT EXISTS.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Prediction history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                period TEXT NOT NULL,
                model_type TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                
                -- Prediction outputs
                pred_next_ret REAL,
                pred_next_price REAL,
                prob_up REAL,
                prob_down REAL,
                last_close REAL,
                confidence_score REAL,
                
                -- Actual outcome (filled in later for accuracy tracking)
                actual_ret REAL,
                actual_price REAL,
                outcome_date DATE,
                was_correct INTEGER,  -- 1 = direction correct, 0 = wrong, NULL = pending
                
                -- Full prediction JSON for detailed analysis
                prediction_json TEXT,
                
                -- Indexing
                UNIQUE(ticker, timestamp, model_type, horizon)
            )
        """)
        
        # Model performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                model_type TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                period TEXT NOT NULL,
                
                -- Performance metrics
                accuracy REAL,
                sharpe_ratio REAL,
                total_return REAL,
                max_drawdown REAL,
                win_rate REAL,
                num_predictions INTEGER,
                
                -- Backtest metadata
                train_start DATE,
                train_end DATE,
                test_start DATE,
                test_end DATE,
                
                -- Full results JSON
                results_json TEXT
            )
        """)
        
        # Query history for quick re-runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query_type TEXT NOT NULL,  -- 'prediction', 'backtest', 'walkforward'
                tickers TEXT NOT NULL,     -- JSON array of tickers
                period TEXT NOT NULL,
                model_type TEXT NOT NULL,
                horizon INTEGER NOT NULL,
                
                -- Optional parameters (JSON)
                extra_params TEXT,
                
                -- Quick access
                is_favorite INTEGER DEFAULT 0,
                label TEXT
            )
        """)
        
        # Live price snapshots (for live updates feature)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL,
                change_pct REAL
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pred_ticker_time 
            ON prediction_history(ticker, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_perf_ticker_model 
            ON model_performance(ticker, model_type, horizon)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_time 
            ON query_history(timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshot_ticker_time 
            ON price_snapshots(ticker, timestamp)
        """)
        
        conn.commit()


# =============================================================================
# PREDICTION HISTORY FUNCTIONS
# =============================================================================

def save_prediction(
    ticker: str,
    period: str,
    model_type: str,
    horizon: int,
    prediction: Dict[str, Any]
) -> int:
    """
    Save a prediction result to the database.
    
    Returns:
        The ID of the inserted row.
    """
    init_database()  # Ensure tables exist
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO prediction_history 
            (ticker, period, model_type, horizon, pred_next_ret, pred_next_price,
             prob_up, prob_down, last_close, confidence_score, prediction_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            period,
            model_type,
            horizon,
            prediction.get("pred_next_ret"),
            prediction.get("pred_next_price"),
            prediction.get("prob_up"),
            prediction.get("prob_down"),
            prediction.get("last_close"),
            prediction.get("confidence_score"),
            json.dumps(prediction, default=str)
        ))
        
        return cursor.lastrowid


def save_predictions_batch(predictions: List[Dict[str, Any]], period: str, model_type: str, horizon: int) -> int:
    """
    Save multiple predictions at once (more efficient for batch runs).
    
    Returns:
        Number of predictions saved.
    """
    init_database()
    count = 0
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        for pred in predictions:
            ticker = pred.get("ticker")
            if not ticker:
                continue
                
            cursor.execute("""
                INSERT OR REPLACE INTO prediction_history 
                (ticker, period, model_type, horizon, pred_next_ret, pred_next_price,
                 prob_up, prob_down, last_close, confidence_score, prediction_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                period,
                model_type,
                horizon,
                pred.get("pred_next_ret"),
                pred.get("pred_next_price"),
                pred.get("prob_up"),
                pred.get("prob_down"),
                pred.get("last_close"),
                pred.get("confidence_score"),
                json.dumps(pred, default=str)
            ))
            count += 1
    
    return count


def get_prediction_history(
    ticker: Optional[str] = None,
    model_type: Optional[str] = None,
    limit: int = 100,
    days_back: int = 30
) -> pd.DataFrame:
    """
    Retrieve prediction history from the database.
    
    Args:
        ticker: Filter by specific ticker (None = all)
        model_type: Filter by model type (None = all)
        limit: Maximum number of records to return
        days_back: How many days back to look
        
    Returns:
        DataFrame with prediction history
    """
    init_database()
    
    query = """
        SELECT 
            timestamp, ticker, period, model_type, horizon,
            pred_next_ret, pred_next_price, prob_up, prob_down,
            last_close, confidence_score, actual_ret, was_correct
        FROM prediction_history
        WHERE timestamp >= datetime('now', ?)
    """
    params = [f'-{days_back} days']
    
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    
    if model_type:
        query += " AND model_type = ?"
        params.append(model_type)
    
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    
    return df


def update_prediction_outcome(
    prediction_id: int,
    actual_ret: float,
    actual_price: float,
    outcome_date: str
) -> bool:
    """
    Update a prediction with the actual outcome for accuracy tracking.
    
    Returns:
        True if update was successful.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # First get the predicted direction
        cursor.execute("SELECT pred_next_ret FROM prediction_history WHERE id = ?", (prediction_id,))
        row = cursor.fetchone()
        
        if not row:
            return False
        
        pred_ret = row["pred_next_ret"]
        was_correct = None
        
        if pred_ret is not None and actual_ret is not None:
            # Direction was correct if both positive or both negative
            was_correct = 1 if (pred_ret > 0) == (actual_ret > 0) else 0
        
        cursor.execute("""
            UPDATE prediction_history
            SET actual_ret = ?, actual_price = ?, outcome_date = ?, was_correct = ?
            WHERE id = ?
        """, (actual_ret, actual_price, outcome_date, was_correct, prediction_id))
        
        return cursor.rowcount > 0


def validate_pending_predictions(max_validate: int = 500) -> Dict[str, Any]:
    """
    Validate pending predictions by fetching actual prices and calculating outcomes.
    
    This function:
    1. Finds predictions where actual_ret is NULL and enough time has passed
    2. Fetches actual prices for those dates
    3. Updates was_correct based on direction accuracy
    
    Returns:
        Dict with validation statistics
    """
    from datetime import datetime, timedelta
    from .aggregator import fetch_prices
    
    init_database()
    
    validated = 0
    errors = 0
    by_ticker = {}
    
    with get_db_connection() as conn:
        # Find unvalidated predictions where horizon has passed
        cursor = conn.execute("""
            SELECT id, ticker, timestamp, horizon, pred_next_ret, last_close
            FROM prediction_history
            WHERE was_correct IS NULL
              AND actual_ret IS NULL
              AND timestamp < datetime('now', '-1 day')
            ORDER BY timestamp DESC
            LIMIT ?
        """, (max_validate,))
        
        pending = cursor.fetchall()
        
        if not pending:
            return {"validated": 0, "errors": 0, "message": "No pending predictions to validate"}
        
        # Group by ticker for efficient fetching
        ticker_predictions = {}
        for row in pending:
            ticker = row["ticker"]
            if ticker not in ticker_predictions:
                ticker_predictions[ticker] = []
            ticker_predictions[ticker].append(dict(row))
        
        # Process each ticker
        for ticker, preds in ticker_predictions.items():
            try:
                # Fetch price history using our multi-source aggregator
                df = fetch_prices(ticker, period="6mo")
                
                if df is None or df.empty:
                    errors += len(preds)
                    continue
                
                # Normalize index
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Get close column
                close_col = None
                for col in ["Close", "close", "Adj Close", "adj_close"]:
                    if col in df.columns:
                        close_col = col
                        break
                
                if close_col is None:
                    errors += len(preds)
                    continue
                
                for pred in preds:
                    try:
                        pred_date = datetime.fromisoformat(pred["timestamp"].replace("Z", ""))
                        horizon = pred["horizon"] or 1
                        outcome_date = pred_date + timedelta(days=horizon)
                        
                        # Find the actual close price on or after outcome date
                        future_prices = df[df.index >= outcome_date]
                        if future_prices.empty:
                            continue  # Not enough data yet
                        
                        actual_price = float(future_prices.iloc[0][close_col])
                        last_close = pred["last_close"]
                        
                        if last_close and last_close > 0:
                            actual_ret = (actual_price - last_close) / last_close
                            pred_ret = pred["pred_next_ret"] or 0
                            
                            # Direction correct if both positive or both negative
                            was_correct = 1 if (pred_ret > 0) == (actual_ret > 0) else 0
                            
                            # Update the prediction
                            conn.execute("""
                                UPDATE prediction_history
                                SET actual_ret = ?, actual_price = ?, 
                                    outcome_date = ?, was_correct = ?
                                WHERE id = ?
                            """, (actual_ret, actual_price, outcome_date.isoformat(), 
                                  was_correct, pred["id"]))
                            
                            validated += 1
                            by_ticker[ticker] = by_ticker.get(ticker, 0) + 1
                    except Exception:
                        errors += 1
                        continue
                        
            except Exception:
                errors += len(preds)
                continue
        
        conn.commit()
    
    return {
        "validated": validated,
        "errors": errors,
        "by_ticker": by_ticker,
        "message": f"Validated {validated} predictions across {len(by_ticker)} tickers"
    }


def get_ticker_accuracy(ticker: str, days_back: int = 90) -> Dict[str, Any]:
    """
    Get accuracy statistics for a specific ticker.
    
    Returns:
        Dict with win_rate, total_predictions, correct_predictions, avg_error
    """
    init_database()
    
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(CASE WHEN was_correct IS NOT NULL THEN was_correct ELSE NULL END) as win_rate,
                AVG(ABS(pred_next_ret - actual_ret)) as avg_error
            FROM prediction_history
            WHERE ticker = ?
              AND was_correct IS NOT NULL
              AND timestamp >= datetime('now', ?)
        """, (ticker, f'-{days_back} days'))
        
        row = cursor.fetchone()
        
        if not row or row["total"] == 0:
            return {"total": 0, "win_rate": None, "correct": 0, "avg_error": None}
        
        return {
            "total": row["total"],
            "correct": row["correct"] or 0,
            "win_rate": row["win_rate"],
            "avg_error": row["avg_error"]
        }


def get_all_ticker_accuracy(days_back: int = 90) -> Dict[str, Dict[str, Any]]:
    """
    Get accuracy statistics for all tickers with validated predictions.
    
    Returns:
        Dict mapping ticker -> accuracy stats
    """
    init_database()
    
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT 
                ticker,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(CASE WHEN was_correct IS NOT NULL THEN was_correct ELSE NULL END) as win_rate,
                AVG(ABS(pred_next_ret - actual_ret)) as avg_error
            FROM prediction_history
            WHERE was_correct IS NOT NULL
              AND timestamp >= datetime('now', ?)
            GROUP BY ticker
            ORDER BY total DESC
        """, (f'-{days_back} days',))
        
        results = {}
        for row in cursor.fetchall():
            results[row["ticker"]] = {
                "total": row["total"],
                "correct": row["correct"] or 0,
                "win_rate": row["win_rate"],
                "avg_error": row["avg_error"]
            }
        
        return results


def get_accuracy_stats(
    ticker: Optional[str] = None,
    model_type: Optional[str] = None,
    days_back: int = 90
) -> Dict[str, Any]:
    """
    Calculate accuracy statistics from historical predictions.
    
    Returns:
        Dictionary with accuracy metrics.
    """
    init_database()
    
    query = """
        SELECT 
            model_type,
            horizon,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            AVG(CASE WHEN was_correct IS NOT NULL THEN was_correct ELSE NULL END) as accuracy,
            AVG(ABS(pred_next_ret - actual_ret)) as avg_error
        FROM prediction_history
        WHERE was_correct IS NOT NULL
          AND timestamp >= datetime('now', ?)
    """
    params = [f'-{days_back} days']
    
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    
    if model_type:
        query += " AND model_type = ?"
        params.append(model_type)
    
    query += " GROUP BY model_type, horizon"
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    
    if df.empty:
        return {"total": 0, "accuracy": None, "by_model": {}}
    
    return {
        "total": int(df["total_predictions"].sum()),
        "accuracy": float(df["accuracy"].mean()) if not df["accuracy"].isna().all() else None,
        "by_model": df.to_dict("records")
    }


# =============================================================================
# MODEL PERFORMANCE FUNCTIONS
# =============================================================================

def save_backtest_result(
    ticker: str,
    model_type: str,
    horizon: int,
    period: str,
    metrics: Dict[str, Any],
    results: Optional[Dict] = None
) -> int:
    """
    Save backtest/walk-forward results to the database.
    
    Returns:
        The ID of the inserted row.
    """
    init_database()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance 
            (ticker, model_type, horizon, period, accuracy, sharpe_ratio,
             total_return, max_drawdown, win_rate, num_predictions,
             train_start, train_end, test_start, test_end, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            model_type,
            horizon,
            period,
            metrics.get("accuracy"),
            metrics.get("sharpe_ratio") or metrics.get("sharpe"),
            metrics.get("total_return") or metrics.get("cumulative_return"),
            metrics.get("max_drawdown"),
            metrics.get("win_rate"),
            metrics.get("num_predictions") or metrics.get("n_trades"),
            metrics.get("train_start"),
            metrics.get("train_end"),
            metrics.get("test_start"),
            metrics.get("test_end"),
            json.dumps(results, default=str) if results else None
        ))
        
        return cursor.lastrowid


def get_model_performance_history(
    ticker: Optional[str] = None,
    model_type: Optional[str] = None,
    limit: int = 50
) -> pd.DataFrame:
    """
    Retrieve historical model performance metrics.
    """
    init_database()
    
    query = """
        SELECT 
            timestamp, ticker, model_type, horizon, period,
            accuracy, sharpe_ratio, total_return, max_drawdown, 
            win_rate, num_predictions
        FROM model_performance
        WHERE 1=1
    """
    params = []
    
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    
    if model_type:
        query += " AND model_type = ?"
        params.append(model_type)
    
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    
    return df


def get_best_model_for_ticker(ticker: str, horizon: int = 1) -> Optional[Dict[str, Any]]:
    """
    Get the best performing model for a ticker based on Sharpe ratio.
    
    Returns:
        Dictionary with best model info, or None if no data.
    """
    init_database()
    
    query = """
        SELECT model_type, horizon, sharpe_ratio, accuracy, win_rate
        FROM model_performance
        WHERE ticker = ? AND horizon = ?
          AND sharpe_ratio IS NOT NULL
        ORDER BY sharpe_ratio DESC
        LIMIT 1
    """
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (ticker, horizon))
        row = cursor.fetchone()
    
    if row:
        return dict(row)
    return None


# =============================================================================
# QUERY HISTORY FUNCTIONS
# =============================================================================

def save_query(
    query_type: str,
    tickers: List[str],
    period: str,
    model_type: str,
    horizon: int,
    extra_params: Optional[Dict] = None,
    label: Optional[str] = None
) -> int:
    """
    Save a query to history for quick re-runs.
    
    Returns:
        The ID of the inserted row.
    """
    init_database()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO query_history 
            (query_type, tickers, period, model_type, horizon, extra_params, label)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            query_type,
            json.dumps(tickers),
            period,
            model_type,
            horizon,
            json.dumps(extra_params) if extra_params else None,
            label
        ))
        
        return cursor.lastrowid


def get_recent_queries(limit: int = 10, query_type: Optional[str] = None) -> List[Dict]:
    """
    Get recent queries for quick re-run.
    """
    init_database()
    
    query = """
        SELECT id, timestamp, query_type, tickers, period, model_type, horizon, label, is_favorite
        FROM query_history
        WHERE 1=1
    """
    params = []
    
    if query_type:
        query += " AND query_type = ?"
        params.append(query_type)
    
    query += " ORDER BY is_favorite DESC, timestamp DESC LIMIT ?"
    params.append(limit)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
    
    results = []
    for row in rows:
        r = dict(row)
        r["tickers"] = json.loads(r["tickers"])
        results.append(r)
    
    return results


def toggle_favorite_query(query_id: int) -> bool:
    """Toggle the favorite status of a query."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE query_history 
            SET is_favorite = CASE WHEN is_favorite = 1 THEN 0 ELSE 1 END
            WHERE id = ?
        """, (query_id,))
        return cursor.rowcount > 0


def delete_query(query_id: int) -> bool:
    """Delete a query from history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM query_history WHERE id = ?", (query_id,))
        return cursor.rowcount > 0


# =============================================================================
# LIVE PRICE SNAPSHOT FUNCTIONS
# =============================================================================

def save_price_snapshot(ticker: str, price: float, volume: Optional[float] = None, change_pct: Optional[float] = None):
    """Save a price snapshot for live tracking."""
    init_database()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO price_snapshots (ticker, price, volume, change_pct)
            VALUES (?, ?, ?, ?)
        """, (ticker, price, volume, change_pct))


def get_price_snapshots(ticker: str, minutes_back: int = 60) -> pd.DataFrame:
    """Get recent price snapshots for a ticker."""
    init_database()
    
    query = """
        SELECT timestamp, price, volume, change_pct
        FROM price_snapshots
        WHERE ticker = ?
          AND timestamp >= datetime('now', ?)
        ORDER BY timestamp ASC
    """
    
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=[ticker, f'-{minutes_back} minutes'])
    
    return df


def cleanup_old_snapshots(days_to_keep: int = 7):
    """Remove old price snapshots to prevent database bloat."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM price_snapshots
            WHERE timestamp < datetime('now', ?)
        """, (f'-{days_to_keep} days',))
        return cursor.rowcount


# =============================================================================
# DATABASE MAINTENANCE
# =============================================================================

def get_database_stats() -> Dict[str, Any]:
    """Get statistics about the database."""
    init_database()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        stats = {}
        
        # Count records in each table
        for table in ["prediction_history", "model_performance", "query_history", "price_snapshots"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()[0]
        
        # Database file size
        if DB_PATH.exists():
            stats["db_size_mb"] = round(DB_PATH.stat().st_size / (1024 * 1024), 2)
        else:
            stats["db_size_mb"] = 0
        
        # Oldest and newest predictions
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM prediction_history")
        row = cursor.fetchone()
        stats["oldest_prediction"] = row[0]
        stats["newest_prediction"] = row[1]
        
    return stats


def vacuum_database():
    """Reclaim disk space after deleting records."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("VACUUM")
    conn.close()


def export_predictions_to_csv(output_path: str, days_back: int = 30) -> str:
    """Export prediction history to CSV file."""
    df = get_prediction_history(limit=10000, days_back=days_back)
    df.to_csv(output_path, index=False)
    return output_path


# Initialize database on module import
try:
    init_database()
except Exception as e:
    print(f"Warning: Could not initialize database: {e}")
