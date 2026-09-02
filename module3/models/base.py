"""
Abstract Base Model Class for all Module 3 neural architectures.
"""

import os
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Common base class providing parameter counting and checkpointing.
    """

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str) -> None:
        """Save model weights to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str, map_location: str = "cpu") -> None:
        """Load model weights from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        state = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state)
