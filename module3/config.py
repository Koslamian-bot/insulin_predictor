"""
Configuration classes for Module 3 model architectures, training, and evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Hyperparameters for deep learning architectures."""
    model_type: str = "gcn_lstm"  # "lstm", "gru", "bilstm", "gcn_lstm", "gcn_gru", "gcn_bilstm"
    in_channels: int = 15          # Node feature dimension from Module 2
    num_steps: int = 24            # Number of 5-min steps in 120-min window
    gcn_hidden_dim: int = 64       # GCN feature channels
    num_gcn_layers: int = 2        # Number of GCNConv layers
    temporal_hidden_dim: int = 64  # Hidden dimension of recurrent layer
    num_temporal_layers: int = 1   # Stacked recurrent layers
    dropout: float = 0.2           # Dropout probability
    bidirectional: bool = False    # Whether recurrent layer is bidirectional
    head_hidden_dim: int = 32      # MLP regression head intermediate dimension


@dataclass
class TrainConfig:
    """Hyperparameters and settings for model training."""
    batch_size: int = 16
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    loss_fn: str = "weighted_smooth_l1"  # "weighted_smooth_l1", "smooth_l1", or "mse"
    pos_weight: float = 5.0              # Scientifically balanced positive-sample weight
    seed: int = 42
    device: str = "cpu"            # "cuda" if available else "cpu"
    clip_grad_norm: float = 1.0


@dataclass
class ExperimentConfig:
    """Paths and global settings for experiment execution."""
    data_dir: str = "data/graphs"
    models_dir: str = "data/models"
    results_dir: str = "data/results"
    reports_dir: str = "data/reports"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
