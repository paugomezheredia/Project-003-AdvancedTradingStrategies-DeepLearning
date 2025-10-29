"""
backtesting.py
---------------
Simulate trading using model predictions and calculate portfolio performance.
Includes:
- Signal-based trade simulation
- Transaction costs
- Performance metrics
"""

import pandas as pd
import numpy as np

def backtest(predictions, prices, commission=0.00125, borrow_rate=0.0025):
    """
    Simple backtesting loop for model-based signals.

    Args:
        predictions (pd.Series): Trading signals (-1, 0, 1)
        prices (pd.Series): Corresponding price series
        commission (float): Transaction cost per trade
        borrow_rate (float): Annualized short borrow cost

    Returns:
        pd.DataFrame: Backtest results with equity curve.
    """
    df = pd.DataFrame({"signal": predictions, "price": prices})
    df["return"] = df["price"].pct_change()
    df["strategy_return"] = df["signal"].shift(1) * df["return"]

    # Apply commission on trades
    df["trade"] = df["signal"].diff().fillna(0).abs()
    df["strategy_return"] -= df["trade"] * commission

    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    return df


def performance_metrics(equity_curve):
    """
    Calculate Sharpe, Sortino, Calmar ratios and drawdown.
    """
    returns = equity_curve.pct_change().dropna()
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    downside = returns[returns < 0]
    sortino = np.mean(returns) / np.std(downside) * np.sqrt(252)
    max_dd = (equity_curve / equity_curve.cummax() - 1).min()
    calmar = np.mean(returns) * 252 / abs(max_dd)
    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max_Drawdown": max_dd
    }
