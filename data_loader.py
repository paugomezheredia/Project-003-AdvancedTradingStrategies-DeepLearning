"""
Module for loading and preparing historical price data.
Handles loading, cleaning, and splitting the dataset chronologically.
"""

import pandas as pd
from typing import Tuple

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load historical market data from a CSV file.

    Args:
        filepath (str): Path to the CSV file containing data.

    Returns:
        pd.DataFrame: DataFrame with datetime index.
    """
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill and backward-fill missing data, ensuring no NaN remains.

    Args:
        df (pd.DataFrame): Raw market data.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = df.ffill().bfill()
    df = df.dropna()
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into 60% train, 20% validation, 20% test chronologically.

    Args:
        df (pd.DataFrame): Complete cleaned dataset.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_end = int(0.6 * len(df))
    val_end = int(0.8 * len(df))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df
