"""
Recurrent Baseline Models: LSTM, GRU, and BiLSTM.

Consume the 24-step temporal sequence representations extracted from the 120-minute
Module 2 window to establish an empirical benchmark against the proposed GCN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from module3.models.base import BaseModel
from module3.config import ModelConfig


class LSTMModel(BaseModel):
    """
    Standard Long Short-Term Memory (LSTM) baseline.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.hidden_dim = config.temporal_hidden_dim
        self.num_layers = config.num_temporal_layers
        self.dropout_rate = config.dropout if config.num_temporal_layers > 1 else 0.0

        # Input feature projection
        self.input_proj = nn.Linear(self.in_channels, self.hidden_dim)

        # Recurrent Core
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
        )

        # Regression Head
        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
            nn.Softplus(beta=1.0),  # Strictly non-negative dosage, non-zero gradient everywhere
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: [B, 24, in_channels]
        Returns:
            y_pred: [B]
        """
        # Project features: [B, 24, hidden_dim]
        h_proj = F.relu(self.input_proj(x_seq))

        # LSTM forward: out is [B, 24, hidden_dim]
        out, (h_n, c_n) = self.lstm(h_proj)

        # Take representation at last time step (prediction instant t = step 23)
        h_last = out[:, -1, :]

        # Regression head: [B, 1] -> [B]
        y_pred = self.head(h_last).squeeze(-1)
        return y_pred


class GRUModel(BaseModel):
    """
    Gated Recurrent Unit (GRU) baseline.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.hidden_dim = config.temporal_hidden_dim
        self.num_layers = config.num_temporal_layers
        self.dropout_rate = config.dropout if config.num_temporal_layers > 1 else 0.0

        self.input_proj = nn.Linear(self.in_channels, self.hidden_dim)

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
        )

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
            nn.Softplus(beta=1.0),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h_proj = F.relu(self.input_proj(x_seq))
        out, h_n = self.gru(h_proj)
        h_last = out[:, -1, :]
        y_pred = self.head(h_last).squeeze(-1)
        return y_pred


class BiLSTMModel(BaseModel):
    """
    Bidirectional Long Short-Term Memory (BiLSTM) baseline.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.hidden_dim = config.temporal_hidden_dim
        self.num_layers = config.num_temporal_layers
        self.dropout_rate = config.dropout if config.num_temporal_layers > 1 else 0.0

        self.input_proj = nn.Linear(self.in_channels, self.hidden_dim)

        self.bilstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate,
        )

        # Head consumes concatenated forward + backward hidden states (2 * hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
            nn.Softplus(beta=1.0),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h_proj = F.relu(self.input_proj(x_seq))
        out, (h_n, c_n) = self.bilstm(h_proj)
        h_last = out[:, -1, :]  # [B, 2 * hidden_dim]
        y_pred = self.head(h_last).squeeze(-1)
        return y_pred
