"""
Module for creating trading signal labels (target feature) based on
future returns. The target encodes as following list:
    1  -> long (buy)
    0  -> hold (neutral)
   -1  -> short (sell)
"""

import pandas as pd

def generate_trading_signals(df: pd.DataFrame, threshold: float = 0.002, horizon: int = 1) -> pd.DataFrame:
    """
    Generate trading signal labels based on future returns.

    Args:
        df (pd.DataFrame): DataFrame with at least a 'Close' column.
        threshold (float): Minimum return magnitude to trigger buy/sell signal.
        horizon (int): Number of periods ahead to compute return.

    Returns:
        pd.DataFrame: Same DataFrame with added 'signal' column.
    """
    df['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['signal'] = 0  # Default: hold
    df.loc[df['future_return'] > threshold, 'signal'] = 1
    df.loc[df['future_return'] < -threshold, 'signal'] = -1
    df.dropna(inplace=True)
    return df
