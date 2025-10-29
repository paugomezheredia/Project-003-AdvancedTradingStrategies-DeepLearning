"""
train.py
--------
Training script for deep learning trading signal models (MLP, CNN, LSTM).
Handles:
- Dataset preparation
- Model training loop
- Class weighting
- Logging metrics to MLFlow
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from mlflow_tracking import log_experiment
from evaluation import evaluate_model


def create_dataloader(X, y, batch_size=32):
    """
    Convert numpy arrays to PyTorch DataLoader.
    """
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=1e-3):
    """
    Train model with class weighting and validation monitoring.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Compute class weights for imbalanced dataset
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_val, y_val)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc, val_f1 = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

    # Log model metrics and parameters to MLFlow
    log_experiment(model, {"epochs": epochs, "lr": lr}, {"val_acc": val_acc, "val_f1": val_f1})

    return model
