#!/usr/bin/env python3
"""
Quick test script to verify Alpaca Paper Trading API is working.
Tests: authentication, account access, positions, and trade execution readiness.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    print("=" * 60)
    print("ALPACA PAPER TRADING API TEST")
    print("=" * 60)
    
    # Try to get keys from streamlit secrets first
    try:
        import streamlit as st
        api_key = st.secrets.get("ALPACA_API_KEY") or st.secrets.get("APCA_API_KEY_ID")
        secret_key = st.secrets.get("ALPACA_SECRET_KEY") or st.secrets.get("APCA_API_SECRET_KEY")
        print("✅ Found credentials in Streamlit secrets")
    except Exception:
        api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        if api_key and secret_key:
            print("✅ Found credentials in environment variables")
        else:
            print("❌ No credentials found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
            return False
    
    if not api_key or not secret_key:
        print("❌ Missing API key or secret key")
        return False
    
    print(f"   API Key: {api_key[:6]}...{api_key[-4:]}")
    
    # Test 1: Import Alpaca SDK
    print("\n[1/5] Testing Alpaca SDK Import...")
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        print("   ✅ Alpaca SDK imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import Alpaca SDK: {e}")
        print("   Run: pip install alpaca-py")
        return False
    
    # Test 2: Create Trading Client (Paper Mode)
    print("\n[2/5] Creating Paper Trading Client...")
    try:
        client = TradingClient(api_key, secret_key, paper=True)
        print("   ✅ Trading client created (paper mode)")
    except Exception as e:
        print(f"   ❌ Failed to create client: {e}")
        return False
    
    # Test 3: Get Account Information
    print("\n[3/5] Fetching Account Information...")
    try:
        account = client.get_account()
        print(f"   ✅ Account accessed successfully")
        print(f"   Account ID: {account.id}")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        print(f"   Day Trade Count: {account.daytrade_count}")
        print(f"   Pattern Day Trader: {account.pattern_day_trader}")
    except Exception as e:
        print(f"   ❌ Failed to get account: {e}")
        return False
    
    # Test 4: Get Current Positions
    print("\n[4/5] Fetching Current Positions...")
    try:
        positions = client.get_all_positions()
        if positions:
            print(f"   ✅ Found {len(positions)} open position(s):")
            for pos in positions:
                pnl = float(pos.unrealized_pl) if pos.unrealized_pl else 0
                pnl_pct = float(pos.unrealized_plpc) * 100 if pos.unrealized_plpc else 0
                print(f"      {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
        else:
            print("   ✅ No open positions")
    except Exception as e:
        print(f"   ❌ Failed to get positions: {e}")
        return False
    
    # Test 5: Get Recent Orders
    print("\n[5/5] Fetching Recent Orders...")
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        
        request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=5)
        orders = client.get_orders(filter=request)
        if orders:
            print(f"   ✅ Found {len(orders)} recent order(s):")
            for order in orders[:5]:
                print(f"      {order.created_at.strftime('%Y-%m-%d %H:%M')} | {order.side} {order.qty} {order.symbol} | {order.status}")
        else:
            print("   ✅ No recent orders")
    except Exception as e:
        print(f"   ⚠️  Could not fetch orders: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ALPACA API IS WORKING!")
    print("=" * 60)
    print(f"\nReady for paper trading:")
    print(f"  - Account Status: {account.status}")
    print(f"  - Available Cash: ${float(account.buying_power):,.2f}")
    print(f"  - Trading Enabled: {account.trading_blocked == False}")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
