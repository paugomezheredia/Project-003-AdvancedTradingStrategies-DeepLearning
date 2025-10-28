"""
mlflow_tracking.py
------------------
Handles MLFlow experiment logging:
- Model versioning
- Metric and parameter tracking
"""

import mlflow
import mlflow.pytorch


def log_experiment(model, params, metrics):
    """
    Log model parameters, metrics, and weights to MLFlow.

    Args:
        model (torch.nn.Module): Trained model.
        params (dict): Hyperparameters used during training.
        metrics (dict): Final evaluation metrics.
    """
    with mlflow.start_run():
        for k, v in params.items():
            mlflow.log_param(k, v)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.pytorch.log_model(model, "model")
        print("✅ Model logged successfully to MLFlow.")
