"""
Training and evaluation routines for Module 3.
"""

from module3.training.loss import get_loss_fn
from module3.training.metrics import compute_metrics
from module3.training.trainer import Trainer, set_seed

__all__ = [
    "get_loss_fn",
    "compute_metrics",
    "Trainer",
    "set_seed",
]
