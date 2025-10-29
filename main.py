"""
main.py
-------
Main execution pipeline for the systematic deep learning trading project.
"""

import torch
import pandas as pd
from data_loader import load_data, preprocess_data, split_data
from target_labeling import generate_trading_signals
from feature_engineering import add_technical_features, normalize_features
from mlp_model import MLPModel
from train import train_model
from data_drift import detect_data_drift
from backtesting import backtest, performance_metrics

def main():
    # 1. Load and preprocess data
    df = load_data("data/historical_prices.csv")
    df = preprocess_data(df)
    df = generate_trading_signals(df)
    df = add_technical_features(df)
    df = normalize_features(df)

    print("Class distribution:", df['signal'].value_counts(normalize=True))

    # 2. Split data
    train_df, val_df, test_df = split_data(df)

    # 3. Separate features and labels
    X_train, y_train = train_df.drop("signal", axis=1).values, train_df["signal"].values
    X_val, y_val = val_df.drop("signal", axis=1).values, val_df["signal"].values
    X_test, y_test = test_df.drop("signal", axis=1).values, test_df["signal"].values

    # 4. Train model
    model = MLPModel(input_dim=X_train.shape[1])
    model = train_model(model, X_train, y_train, X_val, y_val, epochs=25, lr=1e-3)

    # 5. Data drift detection
    detect_data_drift(train_df, test_df)

    # 6. Backtesting
    preds = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).detach().numpy()
    results = backtest(preds, test_df["Close"])
    metrics = performance_metrics(results["equity_curve"])

    print("\nFinal Backtest Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
