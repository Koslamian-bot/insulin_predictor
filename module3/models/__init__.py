"""
Model definitions and factory for Module 3.
"""

from module3.models.base import BaseModel
from module3.models.baselines import LSTMModel, GRUModel, BiLSTMModel
from module3.models.gcn_temporal import GCNTemporalModel
from module3.config import ModelConfig


def build_model(config: ModelConfig) -> BaseModel:
    """
    Factory function instantiating the requested model architecture.
    """
    mtype = config.model_type.lower()
    if mtype == "lstm":
        return LSTMModel(config)
    elif mtype == "gru":
        return GRUModel(config)
    elif mtype == "bilstm":
        config.bidirectional = True
        return BiLSTMModel(config)
    elif mtype in ["gcn_lstm", "gcn_gru", "gcn_bilstm"]:
        if "bilstm" in mtype:
            config.bidirectional = True
        return GCNTemporalModel(config)
    else:
        raise ValueError(
            f"Unknown model_type '{config.model_type}'. Expected one of: "
            "['lstm', 'gru', 'bilstm', 'gcn_lstm', 'gcn_gru', 'gcn_bilstm']"
        )


__all__ = [
    "BaseModel",
    "LSTMModel",
    "GRUModel",
    "BiLSTMModel",
    "GCNTemporalModel",
    "build_model",
]
