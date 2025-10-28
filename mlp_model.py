"""
Defines a Multi-Layer Perceptron for trading signal prediction.
"""

import torch
import torch.nn as nn

class MLPModel(nn.Module):
    """
    Simple feedforward neural network for feature-based prediction.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        """
        return self.network(x)
