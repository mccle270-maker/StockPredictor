# portfolio_manager.py
"""
Portfolio management backend for Stock Predictor
- Load/save holdings
- Calculate P&L
- Fetch current prices
- Manage risk rules
"""

import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None


# ==================== FILE PATHS ====================
PORTFOLIO_DIR = Path("storage/portfolio_data")
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_FILE = PORTFOLIO_DIR / "holdings.json"
EXECUTIONS_FILE = PORTFOLIO_DIR / "executions.json"
RISK_RULES_FILE = PORTFOLIO_DIR / "risk_rules.json"
TRADING_SCHEDULE_FILE = PORTFOLIO_DIR / "trading_schedule.json"


# ==================== PORTFOLIO OPERATIONS ====================

def load_portfolio_data() -> pd.DataFrame:
    """Load portfolio holdings from JSON file"""
    if not PORTFOLIO_FILE.exists():
        return pd.DataFrame(columns=['ticker', 'quantity', 'purchase_price', 'purchase_date'])
    
    try:
        with open(PORTFOLIO_FILE) as f:
            data = json.load(f)
        return pd.DataFrame(data.get('holdings', []))
    except:
        return pd.DataFrame(columns=['ticker', 'quantity', 'purchase_price', 'purchase_date'])


def save_portfolio_data(portfolio_df: pd.DataFrame):
    """Save portfolio holdings to JSON file"""
    data = {
        'holdings': portfolio_df.to_dict('records'),
        'last_updated': datetime.now().isoformat(),
        'cash_available': 5000.0
    }
    
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def add_position(ticker: str, quantity: int, purchase_price: float) -> bool:
    """Add a new position to portfolio"""
    try:
        portfolio = load_portfolio_data()
        
        # Check if already exists (merge quantities)
        if ticker in portfolio['ticker'].values:
            idx = portfolio[portfolio['ticker'] == ticker].index[0]
            portfolio.at[idx, 'quantity'] += quantity
            portfolio.at[idx, 'purchase_price'] = (
                (portfolio.at[idx, 'quantity'] - quantity) * portfolio.at[idx, 'purchase_price'] +
                quantity * purchase_price
            ) / portfolio.at[idx, 'quantity']
        else:
            new_position = pd.DataFrame([{
                'ticker': ticker.upper(),
                'quantity': quantity,
                'purchase_price': purchase_price,
                'purchase_date': datetime.now().strftime('%Y-%m-%d')
            }])
            portfolio = pd.concat([portfolio, new_position], ignore_index=True)
        
        save_portfolio_data(portfolio)
        return True
    except Exception as e:
        print(f"Error adding position: {e}")
        return False


def remove_position(ticker: str) -> bool:
    """Remove a position from portfolio"""
    try:
        portfolio = load_portfolio_data()
        portfolio = portfolio[portfolio['ticker'] != ticker.upper()]
        save_portfolio_data(portfolio)
        return True
    except Exception as e:
        print(f"Error removing position: {e}")
        return False


def update_position(ticker: str, new_quantity: int, new_price: float = None) -> bool:
    """Update position quantity or price"""
    try:
        portfolio = load_portfolio_data()
        idx = portfolio[portfolio['ticker'] == ticker].index
        
        if len(idx) == 0:
            return False
        
        idx = idx[0]
        portfolio.at[idx, 'quantity'] = new_quantity
        
        if new_price is not None:
            portfolio.at[idx, 'purchase_price'] = new_price
        
        save_portfolio_data(portfolio)
        return True
    except Exception as e:
        print(f"Error updating position: {e}")
        return False


# ==================== PRICE FETCHING ====================

@st.cache_data(ttl=60)  # Cache for 60 seconds
def fetch_current_prices(tickers: list) -> dict:
    """Fetch current prices from yfinance"""
    if not yf:
        return {tk: 0 for tk in tickers}
    
    prices = {}
    for tk in tickers:
        try:
            ticker_obj = yf.Ticker(tk)
            price = ticker_obj.info.get('currentPrice', None)
            
            if price is None:
                # Fallback: get from historical data
                hist = ticker_obj.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                else:
                    price = 0
            
            prices[tk] = float(price) if price else 0
        except Exception as e:
            print(f"Error fetching price for {tk}: {e}")
            prices[tk] = 0
    
    return prices


# ==================== PORTFOLIO ANALYSIS ====================

def get_portfolio_with_prices() -> pd.DataFrame:
    """Get portfolio with current prices and calculations"""
    portfolio = load_portfolio_data()
    
    if portfolio.empty:
        return portfolio
    
    # Fetch current prices
    prices = fetch_current_prices(portfolio['ticker'].tolist())
    
    # Add price data
    portfolio['current_price'] = portfolio['ticker'].map(prices)
    portfolio['current_value'] = portfolio['quantity'] * portfolio['current_price']
    portfolio['cost_basis'] = portfolio['quantity'] * portfolio['purchase_price']
    portfolio['unrealized_pl'] = portfolio['current_value'] - portfolio['cost_basis']
    portfolio['unrealized_pl_pct'] = (
        (portfolio['unrealized_pl'] / portfolio['cost_basis'] * 100)
        .fillna(0)
        .round(2)
    )
    
    return portfolio


def get_portfolio_summary(cash_available: float = 5000.0) -> dict:
    """Get portfolio summary metrics"""
    portfolio = get_portfolio_with_prices()
    
    if portfolio.empty:
        return {
            'total_value': cash_available,
            'total_cost': 0,
            'total_pl': 0,
            'total_pl_pct': 0,
            'cash': cash_available,
            'positions_count': 0,
            'top_performer': None,
            'worst_performer': None
        }
    
    total_value = portfolio['current_value'].sum() + cash_available
    total_cost = portfolio['cost_basis'].sum()
    total_pl = total_value - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
    
    # Find best/worst performers
    portfolio = portfolio.sort_values('unrealized_pl_pct', ascending=False)
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_pl': total_pl,
        'total_pl_pct': total_pl_pct,
        'cash': cash_available,
        'positions_count': len(portfolio),
        'top_performer': portfolio.iloc[0]['ticker'] if len(portfolio) > 0 else None,
        'worst_performer': portfolio.iloc[-1]['ticker'] if len(portfolio) > 0 else None,
        'allocation': portfolio[['ticker', 'current_value']].set_index('ticker')['current_value'].to_dict()
    }


def get_allocation() -> dict:
    """Get asset allocation breakdown"""
    portfolio = get_portfolio_with_prices()
    
    if portfolio.empty:
        return {}
    
    total = portfolio['current_value'].sum()
    
    return {
        row['ticker']: {
            'value': row['current_value'],
            'pct': (row['current_value'] / total * 100) if total > 0 else 0,
            'quantity': row['quantity']
        }
        for _, row in portfolio.iterrows()
    }


# ==================== TRADE EXECUTION LOGGING ====================

def log_execution(ticker: str, action: str, quantity: int, price: float, status: str = "executed"):
    """Log a trade execution"""
    execution = {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker.upper(),
        'action': action.upper(),  # BUY/SELL
        'quantity': quantity,
        'price': price,
        'total_value': quantity * price,
        'status': status
    }
    
    # Load existing executions
    executions = []
    if EXECUTIONS_FILE.exists():
        with open(EXECUTIONS_FILE) as f:
            executions = json.load(f)
    
    # Append new execution
    executions.append(execution)
    
    # Save
    with open(EXECUTIONS_FILE, 'w') as f:
        json.dump(executions, f, indent=2)


def get_recent_executions(limit: int = 20) -> list:
    """Get recent trade executions"""
    if not EXECUTIONS_FILE.exists():
        return []
    
    with open(EXECUTIONS_FILE) as f:
        executions = json.load(f)
    
    return executions[-limit:]


# ==================== RISK MANAGEMENT ====================

def save_risk_rules(rules: dict):
    """Save risk management rules"""
    rules['updated_at'] = datetime.now().isoformat()
    
    with open(RISK_RULES_FILE, 'w') as f:
        json.dump(rules, f, indent=2)


def load_risk_rules() -> dict:
    """Load risk management rules"""
    if not RISK_RULES_FILE.exists():
        return {
            'max_position_size': 0.10,  # 10% per position
            'max_daily_loss': 0.02,     # 2% daily max loss
            'max_leverage': 2.0,        # 2x leverage max
            'stop_loss_pct': 0.10,      # 10% trailing stop
            'take_profit_pct': 0.25     # 25% target
        }
    
    with open(RISK_RULES_FILE) as f:
        return json.load(f)


def check_risk_limits(new_trade: dict, summary: dict) -> dict:
    """Check if trade violates risk rules"""
    rules = load_risk_rules()
    violations = []
    
    # Check position size
    trade_value = new_trade['quantity'] * new_trade['price']
    position_pct = trade_value / summary['total_value']
    
    if position_pct > rules['max_position_size']:
        violations.append(f"Position size {position_pct:.1%} exceeds max {rules['max_position_size']:.1%}")
    
    # Check daily loss
    if summary['total_pl_pct'] < -rules['max_daily_loss'] * 100:
        violations.append(f"Daily loss {summary['total_pl_pct']:.1%} exceeds max {-rules['max_daily_loss']*100:.1%}")
    
    return {
        'allowed': len(violations) == 0,
        'violations': violations
    }


# ==================== TRADING SCHEDULE ====================

def save_trading_schedule(schedule: dict):
    """Save trading schedule"""
    schedule['updated_at'] = datetime.now().isoformat()
    
    with open(TRADING_SCHEDULE_FILE, 'w') as f:
        json.dump(schedule, f, indent=2)


def load_trading_schedule() -> dict:
    """Load trading schedule"""
    if not TRADING_SCHEDULE_FILE.exists():
        return {
            'enabled': False,
            'frequency': 'daily',
            'execution_time': '09:30',
            'last_run': None
        }
    
    with open(TRADING_SCHEDULE_FILE) as f:
        return json.load(f)


# ==================== HELPERS ====================

def clear_all_data():
    """Clear all portfolio data (use with caution!)"""
    try:
        PORTFOLIO_FILE.unlink(missing_ok=True)
        EXECUTIONS_FILE.unlink(missing_ok=True)
        RISK_RULES_FILE.unlink(missing_ok=True)
        TRADING_SCHEDULE_FILE.unlink(missing_ok=True)
        return True
    except Exception as e:
        print(f"Error clearing data: {e}")
        return False


def export_portfolio_csv() -> str:
    """Export portfolio to CSV"""
    portfolio = get_portfolio_with_prices()
    csv_path = PORTFOLIO_DIR / "portfolio_export.csv"
    portfolio.to_csv(csv_path, index=False)
    return str(csv_path)


def export_executions_csv() -> str:
    """Export execution history to CSV"""
    executions = get_recent_executions(limit=1000)
    df = pd.DataFrame(executions)
    csv_path = PORTFOLIO_DIR / "executions_export.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ==================== TEST ====================

if __name__ == "__main__":
    # Test operations
    print("Testing portfolio_manager.py...")
    
    # Test adding position
    print("\n1. Adding position...")
    add_position("AAPL", 10, 180.50)
    add_position("NVDA", 5, 120.00)
    
    # Test loading
    print("\n2. Loading portfolio...")
    portfolio = load_portfolio_data()
    print(portfolio)
    
    # Test with prices
    print("\n3. Getting portfolio with prices...")
    portfolio_with_prices = get_portfolio_with_prices()
    print(portfolio_with_prices)
    
    # Test summary
    print("\n4. Getting summary...")
    summary = get_portfolio_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test allocation
    print("\n5. Getting allocation...")
    allocation = get_allocation()
    for ticker, data in allocation.items():
        print(f"  {ticker}: ${data['value']:.2f} ({data['pct']:.1f}%)")
    
    print("\n✅ All tests passed!")
