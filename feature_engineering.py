"""
Creates engineered time series features using technical indicators such as
momentum, volatility, and volume-based metrics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a set of technical indicators as predictive features.

    Args:
        df (pd.DataFrame): OHLCV dataset.

    Returns:
        pd.DataFrame: Dataset with added features.
    """
    df['returns'] = df['Close'].pct_change()
    df['rolling_mean_5'] = df['Close'].rolling(5).mean()
    df['rolling_mean_20'] = df['Close'].rolling(20).mean()
    df['rolling_std_20'] = df['Close'].rolling(20).std()
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['ema_12'] = df['Close'].ewm(span=12).mean()
    df['ema_26'] = df['Close'].ewm(span=26).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['signal_line'] = df['macd'].ewm(span=9).mean()
    df['rsi'] = compute_rsi(df['Close'])
    df['volatility_10'] = df['returns'].rolling(10).std()
    df['volume_ma_10'] = df['Volume'].rolling(10).mean()
    df['price_volume_trend'] = (df['returns'] * df['Volume']).cumsum()
    df['high_low_ratio'] = df['High'] / df['Low']
    df['close_open_ratio'] = df['Close'] / df['Open']
    df['log_volume'] = np.log(df['Volume'] + 1)
    df['rolling_skew_10'] = df['returns'].rolling(10).skew()
    df['rolling_kurt_10'] = df['returns'].rolling(10).kurt()
    df['ema_ratio'] = df['ema_12'] / df['ema_26']
    df.dropna(inplace=True)
    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all numeric columns between 0 and 1 using MinMaxScaler.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)
