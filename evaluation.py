"""
evaluation.py
--------------
Model evaluation utilities for accuracy, F1, and confusion matrix.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on validation/test data.
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return acc, f1


def evaluate_test_performance(model, X_test, y_test):
    """
    Convenience function for evaluating on final test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        ),
        batch_size=32,
        shuffle=False
    )
    acc, f1 = evaluate_model(model, dataloader, device)
    print(f"Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}")
    return acc, f1
