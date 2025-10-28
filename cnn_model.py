"""
Defines a 1D CNN model to learn local temporal dependencies in sequential data.
"""

import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    Convolutional Neural Network for time series classification.
    """
    def __init__(self, input_channels: int, output_dim: int = 3):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Forward pass for CNN model.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
