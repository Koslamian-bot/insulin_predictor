"""
Loss functions for bolus insulin regression with imbalance compensation.
"""

from typing import Callable, Optional
import torch
import torch.nn as nn


class WeightedSmoothL1Loss(nn.Module):
    """
    Weighted Smooth L1 (Huber) Loss for severely imbalanced drug dosage regression.

    Rationale:
    In Type 1 Diabetes, >98% of 5-minute continuous windows have zero bolus injection.
    An unweighted loss mathematically encourages models to predict 0.0 everywhere.
    This loss assigns an amplified weight (default: 8.0x) to active bolus samples (y > 0.1 U)
    so that predicting meal bolus doses receives appropriate gradient priority.
    """

    def __init__(self, pos_weight: float = 8.0, beta: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.beta = beta
        self.base_loss = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raw_loss = self.base_loss(pred, target)
        weights = torch.ones_like(target)
        weights = torch.where(target > 0.1, weights * self.pos_weight, weights)
        return (raw_loss * weights).mean()


def get_loss_fn(loss_name: str = "weighted_smooth_l1", pos_weight: float = 8.0) -> nn.Module:
    """
    Returns the configured regression loss function.
    """
    lname = loss_name.lower()
    if lname in ["weighted_smooth_l1", "weighted_huber", "imbalanced"]:
        return WeightedSmoothL1Loss(pos_weight=pos_weight, beta=1.0)
    elif lname in ["smooth_l1", "huber"]:
        return nn.SmoothL1Loss(beta=1.0)
    elif lname == "mse":
        return nn.MSELoss()
    elif lname == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss function: '{loss_name}'. Choose 'weighted_smooth_l1', 'smooth_l1', 'mse', or 'mae'.")
