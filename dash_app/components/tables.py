"""
Table Components for QuantDesk Dashboard

DataTable configurations for signals, options, and P&L tables.
"""

from dash import dash_table, html
import dash_bootstrap_components as dbc
import pandas as pd


# Common dark theme styles for DataTables
DARK_HEADER_STYLE = {
    "backgroundColor": "#161b22",
    "fontWeight": "600",
    "border": "1px solid #30363d",
    "color": "#f0f6fc",
    "textTransform": "uppercase",
    "fontSize": "0.75rem",
    "letterSpacing": "0.5px",
}

DARK_CELL_STYLE = {
    "backgroundColor": "#0d1117",
    "color": "#c9d1d9",
    "border": "1px solid #21262d",
    "padding": "10px 12px",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "0.85rem",
}

DARK_DATA_CONDITIONAL = [
    {
        "if": {"row_index": "odd"},
        "backgroundColor": "#0f141a",
    },
    {
        "if": {"state": "active"},
        "backgroundColor": "#21262d",
        "border": "1px solid #388bfd",
    },
    {
        "if": {"state": "selected"},
        "backgroundColor": "#21262d",
        "border": "1px solid #388bfd",
    },
]


def create_signals_table(predictions: list) -> dash_table.DataTable:
    """
    Create a signals table from predictions.
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Dash DataTable component
    """
    if not predictions:
        return html.Div("No data available", className="text-muted text-center py-4")
    
    # Build table data
    data = []
    for p in predictions:
        signal = p.get("signal", "HOLD")
        pred_ret = (p.get("pred_next_ret", 0) or 0) * 100
        
        data.append({
            "ticker": p.get("ticker", "N/A"),
            "signal": signal,
            "pred_pct": f"{pred_ret:+.2f}%",
            "prob_up": f"{(p.get('prob_up', 0.5) or 0.5)*100:.0f}%",
            "confidence": f"{(p.get('confidence', 0) or 0)*100:.0f}%",
            "z_score": f"{p.get('pred_zscore', p.get('prediction_zscore', 0)) or 0:+.2f}",
            "target": f"${p.get('pred_next_price', 0) or 0:.2f}",
            "last_close": f"${p.get('last_close', 0) or 0:.2f}",
        })
    
    columns = [
        {"name": "Ticker", "id": "ticker"},
        {"name": "Signal", "id": "signal"},
        {"name": "Pred %", "id": "pred_pct"},
        {"name": "P(Up)", "id": "prob_up"},
        {"name": "Conf", "id": "confidence"},
        {"name": "Z-Score", "id": "z_score"},
        {"name": "Target", "id": "target"},
        {"name": "Last", "id": "last_close"},
    ]
    
    # Conditional styling for signals
    style_data_conditional = DARK_DATA_CONDITIONAL + [
        {
            "if": {"filter_query": '{signal} = "BUY" || {signal} = "STRONG BUY"'},
            "color": "#2ecc71",
        },
        {
            "if": {"filter_query": '{signal} = "SELL" || {signal} = "STRONG SELL"'},
            "color": "#e74c3c",
        },
        {
            "if": {
                "column_id": "pred_pct",
                "filter_query": '{pred_pct} contains "+"',
            },
            "color": "#2ecc71",
        },
        {
            "if": {
                "column_id": "pred_pct",
                "filter_query": '{pred_pct} contains "-"',
            },
            "color": "#e74c3c",
        },
    ]
    
    return dash_table.DataTable(
        id="signals-table",
        columns=columns,
        data=data,
        style_header=DARK_HEADER_STYLE,
        style_cell=DARK_CELL_STYLE,
        style_data_conditional=style_data_conditional,
        sort_action="native",
        filter_action="native",
        page_size=15,
        page_action="native",
        style_table={"overflowX": "auto"},
    )


def create_pnl_table(backtest_results: list) -> dash_table.DataTable:
    """
    Create a P&L summary table from backtest results.
    
    Args:
        backtest_results: List of backtest result dictionaries
        
    Returns:
        Dash DataTable component
    """
    if not backtest_results:
        return html.Div("No backtest data available", className="text-muted text-center py-4")
    
    data = []
    for r in backtest_results:
        data.append({
            "ticker": r.get("ticker", "N/A"),
            "sharpe": f"{r.get('sharpe', 0):.2f}",
            "total_return": f"{r.get('total_return', 0)*100:+.1f}%",
            "max_drawdown": f"{r.get('max_drawdown', 0)*100:.1f}%",
            "win_rate": f"{r.get('win_rate', r.get('accuracy', 0))*100:.0f}%",
            "num_trades": r.get("num_trades", 0),
            "test_period": r.get("test_period", "N/A"),
        })
    
    columns = [
        {"name": "Ticker", "id": "ticker"},
        {"name": "Sharpe", "id": "sharpe"},
        {"name": "Total Return", "id": "total_return"},
        {"name": "Max DD", "id": "max_drawdown"},
        {"name": "Win Rate", "id": "win_rate"},
        {"name": "Trades", "id": "num_trades"},
        {"name": "Test Period", "id": "test_period"},
    ]
    
    # Conditional styling
    style_data_conditional = DARK_DATA_CONDITIONAL + [
        {
            "if": {
                "column_id": "sharpe",
                "filter_query": '{sharpe} > 1',
            },
            "color": "#2ecc71",
            "fontWeight": "600",
        },
        {
            "if": {
                "column_id": "sharpe",
                "filter_query": '{sharpe} < 0',
            },
            "color": "#e74c3c",
        },
        {
            "if": {
                "column_id": "total_return",
                "filter_query": '{total_return} contains "+"',
            },
            "color": "#2ecc71",
        },
        {
            "if": {
                "column_id": "total_return",
                "filter_query": '{total_return} contains "-"',
            },
            "color": "#e74c3c",
        },
    ]
    
    return dash_table.DataTable(
        id="pnl-table",
        columns=columns,
        data=data,
        style_header=DARK_HEADER_STYLE,
        style_cell=DARK_CELL_STYLE,
        style_data_conditional=style_data_conditional,
        sort_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
    )


def create_options_table(options_data: list) -> dash_table.DataTable:
    """
    Create an options strategies table.
    
    Args:
        options_data: List of options strategy dictionaries
        
    Returns:
        Dash DataTable component
    """
    if not options_data:
        return html.Div("No options data available", className="text-muted text-center py-4")
    
    data = []
    for opt in options_data:
        data.append({
            "ticker": opt.get("ticker", "N/A"),
            "strategy": opt.get("strategy", "N/A"),
            "direction": opt.get("direction", "N/A"),
            "strike": f"${opt.get('strike', 0):.2f}",
            "expiry": opt.get("expiry", "N/A"),
            "premium": f"${opt.get('premium', 0):.2f}",
            "iv": f"{(opt.get('iv', 0) or 0)*100:.1f}%",
            "delta": f"{opt.get('delta', 0):.2f}",
        })
    
    columns = [
        {"name": "Ticker", "id": "ticker"},
        {"name": "Strategy", "id": "strategy"},
        {"name": "Direction", "id": "direction"},
        {"name": "Strike", "id": "strike"},
        {"name": "Expiry", "id": "expiry"},
        {"name": "Premium", "id": "premium"},
        {"name": "IV", "id": "iv"},
        {"name": "Delta", "id": "delta"},
    ]
    
    return dash_table.DataTable(
        id="options-table",
        columns=columns,
        data=data,
        style_header=DARK_HEADER_STYLE,
        style_cell=DARK_CELL_STYLE,
        style_data_conditional=DARK_DATA_CONDITIONAL,
        sort_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
    )


def create_portfolio_positions_table(positions: list) -> dash_table.DataTable:
    """
    Create a portfolio positions table.
    
    Args:
        positions: List of position dictionaries
        
    Returns:
        Dash DataTable component
    """
    if not positions:
        return html.Div("No positions", className="text-muted text-center py-4")
    
    data = []
    for pos in positions:
        pnl = pos.get("pnl", 0)
        pnl_pct = pos.get("pnl_pct", 0)
        
        data.append({
            "ticker": pos.get("ticker", "N/A"),
            "qty": pos.get("quantity", 0),
            "avg_cost": f"${pos.get('avg_cost', 0):.2f}",
            "current": f"${pos.get('current_price', 0):.2f}",
            "market_value": f"${pos.get('market_value', 0):,.2f}",
            "pnl": f"${pnl:+,.2f}",
            "pnl_pct": f"{pnl_pct:+.2f}%",
        })
    
    columns = [
        {"name": "Ticker", "id": "ticker"},
        {"name": "Qty", "id": "qty"},
        {"name": "Avg Cost", "id": "avg_cost"},
        {"name": "Current", "id": "current"},
        {"name": "Market Value", "id": "market_value"},
        {"name": "P&L", "id": "pnl"},
        {"name": "P&L %", "id": "pnl_pct"},
    ]
    
    style_data_conditional = DARK_DATA_CONDITIONAL + [
        {
            "if": {
                "column_id": "pnl",
                "filter_query": '{pnl} contains "+"',
            },
            "color": "#2ecc71",
        },
        {
            "if": {
                "column_id": "pnl",
                "filter_query": '{pnl} contains "-"',
            },
            "color": "#e74c3c",
        },
        {
            "if": {
                "column_id": "pnl_pct",
                "filter_query": '{pnl_pct} contains "+"',
            },
            "color": "#2ecc71",
        },
        {
            "if": {
                "column_id": "pnl_pct",
                "filter_query": '{pnl_pct} contains "-"',
            },
            "color": "#e74c3c",
        },
    ]
    
    return dash_table.DataTable(
        id="positions-table",
        columns=columns,
        data=data,
        style_header=DARK_HEADER_STYLE,
        style_cell=DARK_CELL_STYLE,
        style_data_conditional=style_data_conditional,
        sort_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
    )


def create_trade_history_table(trades: list) -> dash_table.DataTable:
    """
    Create a trade history table.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dash DataTable component
    """
    if not trades:
        return html.Div("No trade history", className="text-muted text-center py-4")
    
    data = []
    for trade in trades:
        pnl = trade.get("pnl", 0)
        
        data.append({
            "date": trade.get("date", "N/A"),
            "ticker": trade.get("ticker", "N/A"),
            "side": trade.get("side", "N/A"),
            "qty": trade.get("quantity", 0),
            "price": f"${trade.get('price', 0):.2f}",
            "pnl": f"${pnl:+,.2f}" if pnl else "—",
            "status": trade.get("status", "N/A"),
        })
    
    columns = [
        {"name": "Date", "id": "date"},
        {"name": "Ticker", "id": "ticker"},
        {"name": "Side", "id": "side"},
        {"name": "Qty", "id": "qty"},
        {"name": "Price", "id": "price"},
        {"name": "P&L", "id": "pnl"},
        {"name": "Status", "id": "status"},
    ]
    
    style_data_conditional = DARK_DATA_CONDITIONAL + [
        {
            "if": {"filter_query": '{side} = "BUY"'},
            "color": "#2ecc71",
        },
        {
            "if": {"filter_query": '{side} = "SELL"'},
            "color": "#e74c3c",
        },
    ]
    
    return dash_table.DataTable(
        id="trade-history-table",
        columns=columns,
        data=data,
        style_header=DARK_HEADER_STYLE,
        style_cell=DARK_CELL_STYLE,
        style_data_conditional=style_data_conditional,
        sort_action="native",
        page_size=20,
        style_table={"overflowX": "auto"},
    )
