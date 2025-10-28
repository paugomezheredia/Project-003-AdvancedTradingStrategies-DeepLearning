"""
Splits the processed dataset chronologically into training, validation,
and test sets. Ensures no look-ahead bias for financial time series.

Output files:
    - data/train_split.csv
    - data/val_split.csv
    - data/test_split.csv
"""

import pandas as pd
from typing import Tuple

def split_data(df: pd.DataFrame, train_size: float = 0.6, val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets while preserving chronological order.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing features and target column.
    train_size : float, optional
        Fraction of data for training (default=0.6).
    val_size : float, optional
        Fraction of data for validation (default=0.2).
        Remaining fraction will be used for testing.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames for train, validation, and test sets.
    """
    if not 0 < train_size < 1 or not 0 < val_size < 1:
        raise ValueError("train_size and val_size must be between 0 and 1")

    df = df.sort_index()  # Sort chronologically

    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"✅ Chronological data split complete:")
    print(f"   Train: {train.shape[0]} samples ({train_size*100:.0f}%)")
    print(f"   Validation: {val.shape[0]} samples ({val_size*100:.0f}%)")
    print(f"   Test: {test.shape[0]} samples ({(1-train_size-val_size)*100:.0f}%)")

    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Save split datasets as CSV files.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Data splits to save.
    """
    train.to_csv("data/train_split.csv", index=False)
    val.to_csv("data/val_split.csv", index=False)
    test.to_csv("data/test_split.csv", index=False)
    print("💾 Saved splits to 'data/' directory.")


if __name__ == "__main__":
    df = pd.read_csv("data/processed_dataset.csv")
    train, val, test = split_data(df)
    save_splits(train, val, test)
