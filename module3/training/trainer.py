"""
Model Training & Validation Engine.

Manages the training loop, validation loop, early stopping,
checkpoint saving, and zero-leakage verification during training.
"""

import os
import json
import time
import random
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from module3.models.base import BaseModel
from module3.models.gcn_temporal import GCNTemporalModel
from module3.config import TrainConfig
from module3.training.loss import get_loss_fn
from module3.training.metrics import compute_metrics
from module3.data.graph_dataset import extract_temporal_sequence_from_batch
from module2.ontology.schema_types import NodeType

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Ensure exact reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Orchestrates neural network training and validation.
    """

    def __init__(
        self,
        model: BaseModel,
        config: TrainConfig,
        model_name: str = "model",
        save_dir: str = "data/models",
    ):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.save_dir = save_dir
        self.device = torch.device(config.device if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        self.model.to(self.device)

        pos_w = getattr(config, "pos_weight", 8.0)
        self.loss_fn = get_loss_fn(config.loss_fn, pos_weight=pos_w)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)

        set_seed(config.seed)
        self.is_gcn = isinstance(model, GCNTemporalModel)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch over the dataset."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_trues = []

        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Zero-leakage verification assertion on batch
            self._assert_zero_leakage(batch)

            if self.is_gcn:
                preds = self.model(batch)
                y_true = batch.y.view(-1).to(self.device)
            else:
                x_seq, y_true = extract_temporal_sequence_from_batch(batch)
                preds = self.model(x_seq)

            loss = self.loss_fn(preds, y_true)
            loss.backward()

            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

            self.optimizer.step()

            total_loss += loss.item() * len(y_true)
            all_preds.extend(preds.detach().cpu().numpy())
            all_trues.extend(y_true.detach().cpu().numpy())

        avg_loss = total_loss / max(1, len(all_trues))
        metrics = compute_metrics(np.array(all_trues), np.array(all_preds))
        return avg_loss, metrics

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
        """Run validation or evaluation without gradients."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                self._assert_zero_leakage(batch)

                if self.is_gcn:
                    preds = self.model(batch)
                    y_true = batch.y.view(-1).to(self.device)
                else:
                    x_seq, y_true = extract_temporal_sequence_from_batch(batch)
                    preds = self.model(x_seq)

                loss = self.loss_fn(preds, y_true)
                total_loss += loss.item() * len(y_true)
                all_preds.extend(preds.detach().cpu().numpy())
                all_trues.extend(y_true.detach().cpu().numpy())

        avg_loss = total_loss / max(1, len(all_trues))
        y_true_np = np.array(all_trues)
        y_pred_np = np.array(all_preds)
        metrics = compute_metrics(y_true_np, y_pred_np)
        return avg_loss, metrics, y_true_np, y_pred_np

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """
        Execute full training loop with early stopping and checkpoint saving.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.save_dir, f"{self.model_name}_best.pt")

        best_val_loss = float("inf")
        best_val_metrics: Dict[str, float] = {}
        patience_counter = 0

        history = {
            "epoch": [],
            "train_loss": [],
            "train_mae": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_r2": [],
            "epoch_duration_sec": [],
            "learning_rate": [],
        }

        print(f"\nTraining Model: {self.model_name.upper()} (Parameters: {self.model.count_parameters():,})")
        print("-" * 75)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _ = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            epoch_duration = round(time.time() - t0, 2)
            current_lr = self.optimizer.param_groups[0]['lr']

            history["epoch"].append(epoch)
            history["train_loss"].append(float(train_loss))
            history["train_mae"].append(float(train_metrics.get("mae", 0.0)))
            history["val_loss"].append(float(val_loss))
            history["val_mae"].append(float(val_metrics.get("mae", 0.0)))
            history["val_rmse"].append(float(val_metrics.get("rmse", 0.0)))
            history["val_r2"].append(float(val_metrics.get("r2", 0.0)))
            history["epoch_duration_sec"].append(epoch_duration)
            history["learning_rate"].append(float(current_lr))

            print(
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({epoch_duration}s) "
                f"Train Loss: {train_loss:.4f} | Train MAE: {train_metrics.get('mae', 0.0):.3f}U | "
                f"Val Loss: {val_loss:.4f} | Val MAE: {val_metrics.get('mae', 0.0):.3f}U | Val RMSE: {val_metrics.get('rmse', 0.0):.3f}U | LR: {current_lr:.6f}"
            )

            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_metrics = val_metrics
                patience_counter = 0
                self.model.save_checkpoint(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch} (patience={self.config.early_stopping_patience}).")
                    break

        print(f"[Best Checkpoint Saved]: {checkpoint_path} (Val MAE: {best_val_metrics.get('mae', 0.0):.3f} U)")

        # Persist full training history as verifiable proof of training
        results_dir = os.path.join(os.path.dirname(os.path.abspath(self.save_dir)), "results")
        os.makedirs(results_dir, exist_ok=True)
        history_path = os.path.join(results_dir, f"{self.model_name}_training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "parameters": self.model.count_parameters(),
                "best_val_loss": float(best_val_loss),
                "best_val_metrics": best_val_metrics,
                "epochs_executed": len(history["epoch"]),
                "checkpoint_path": checkpoint_path,
                "history": history,
            }, f, indent=2)
        print(f"[Training History Log Saved]: {history_path}")

        return {
            "best_val_metrics": best_val_metrics,
            "best_val_loss": best_val_loss,
            "checkpoint_path": checkpoint_path,
            "history": history,
            "history_path": history_path,
        }

    @staticmethod
    def _assert_zero_leakage(batch) -> None:
        """
        Runtime integrity check asserting that NO Bolus node exists at prediction instant t.
        """
        if hasattr(batch, "node_types") and hasattr(batch, "x"):
            delta_tau = batch.x[:, 1]
            at_pred_time = torch.isclose(delta_tau, torch.tensor(0.0, device=delta_tau.device), atol=1e-4)
            is_bolus = batch.node_types == int(NodeType.BOLUS_INSULIN)
            leaked = at_pred_time & is_bolus
            if leaked.any():
                raise RuntimeError(
                    f"FATAL LEAKAGE IN MODULE 3: Found {leaked.sum().item()} Bolus nodes at delta_tau=0.0!"
                )
