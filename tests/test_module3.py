"""
Unit & Integration Test Suite for Module 3.

Verifies:
1. Dataset loads correctly from data/graphs/*.pt
2. Batch dimensions and shape assertions
3. Model forward pass on all 4 architectures (LSTM, GRU, BiLSTM, GCN+LSTM)
4. Output shape correctness ([B])
5. Loss calculation and backward pass
6. Single training step completes
7. Zero-leakage assertion (target bolus absent from input features)
8. Train/val/test data separation
"""

import unittest
import os
import torch
import numpy as np

from module3.config import ModelConfig, TrainConfig
from module3.data.graph_dataset import (
    load_pyg_split,
    create_dataloader,
    extract_temporal_sequence_from_batch,
)
from module3.models import build_model
from module3.training.loss import get_loss_fn
from module3.training.trainer import Trainer, set_seed
from module2.ontology.schema_types import NodeType


class TestModule3(unittest.TestCase):
    """Test suite ensuring all Module 3 components satisfy specification."""

    @classmethod
    def setUpClass(cls):
        set_seed(42)
        cls.data_dir = "data/graphs"
        cls.train_graphs = load_pyg_split(cls.data_dir, "train")
        cls.val_graphs = load_pyg_split(cls.data_dir, "val")
        cls.test_graphs = load_pyg_split(cls.data_dir, "test")

    def test_01_dataset_loading(self):
        """1. Verify dataset loads correctly from data/graphs/*.pt."""
        self.assertGreater(len(self.train_graphs), 0)
        self.assertGreater(len(self.val_graphs), 0)
        self.assertGreater(len(self.test_graphs), 0)

        # Check PyG Data object attributes
        sample = self.train_graphs[0]
        self.assertTrue(hasattr(sample, "x"))
        self.assertTrue(hasattr(sample, "edge_index"))
        self.assertTrue(hasattr(sample, "y"))
        self.assertEqual(sample.x.shape[1], 15)  # 15 node features

    def test_02_batch_dimensions(self):
        """2. Verify batch dimensions and dataloader shapes."""
        loader = create_dataloader(self.train_graphs, batch_size=8, shuffle=False)
        batch = next(iter(loader))

        self.assertEqual(batch.num_graphs, 8)
        self.assertEqual(batch.edge_index.shape[0], 2)
        self.assertEqual(batch.x.shape[1], 15)

        # Temporal sequence extraction for baselines
        x_seq, y_target = extract_temporal_sequence_from_batch(batch, num_steps=24)
        self.assertEqual(x_seq.shape, (8, 24, 15))
        self.assertEqual(y_target.shape, (8,))

    def test_03_and_04_model_forward_passes_and_shapes(self):
        """3 & 4. Verify forward passes and output shapes [B] across all models."""
        loader = create_dataloader(self.train_graphs, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        x_seq, _ = extract_temporal_sequence_from_batch(batch, num_steps=24)

        # A. LSTM
        cfg_lstm = ModelConfig(model_type="lstm", in_channels=15, temporal_hidden_dim=32)
        m_lstm = build_model(cfg_lstm)
        out_lstm = m_lstm(x_seq)
        self.assertEqual(out_lstm.shape, (4,))
        self.assertTrue(torch.all(out_lstm >= 0.0))  # Non-negative dosage

        # B. GRU
        cfg_gru = ModelConfig(model_type="gru", in_channels=15, temporal_hidden_dim=32)
        m_gru = build_model(cfg_gru)
        out_gru = m_gru(x_seq)
        self.assertEqual(out_gru.shape, (4,))

        # C. BiLSTM
        cfg_bilstm = ModelConfig(model_type="bilstm", in_channels=15, temporal_hidden_dim=32)
        m_bilstm = build_model(cfg_bilstm)
        out_bilstm = m_bilstm(x_seq)
        self.assertEqual(out_bilstm.shape, (4,))

        # D. Proposed GCN + LSTM
        cfg_gcn = ModelConfig(model_type="gcn_lstm", in_channels=15, gcn_hidden_dim=32, temporal_hidden_dim=32)
        m_gcn = build_model(cfg_gcn)
        out_gcn = m_gcn(batch)
        self.assertEqual(out_gcn.shape, (4,))
        self.assertTrue(torch.all(out_gcn >= 0.0))

    def test_05_loss_calculation(self):
        """5. Verify loss functions compute valid non-nan scalar gradients."""
        loss_smooth = get_loss_fn("smooth_l1")
        loss_mse = get_loss_fn("mse")

        y_pred = torch.tensor([2.5, 0.0, 4.0], requires_grad=True)
        y_true = torch.tensor([3.0, 0.0, 3.5])

        l1 = loss_smooth(y_pred, y_true)
        self.assertFalse(torch.isnan(l1))
        self.assertGreater(l1.item(), 0.0)

        l2 = loss_mse(y_pred, y_true)
        self.assertFalse(torch.isnan(l2))

    def test_06_single_training_step(self):
        """6. Verify one full optimization step updates model weights."""
        loader = create_dataloader(self.train_graphs, batch_size=4, shuffle=False)
        cfg = ModelConfig(model_type="gcn_lstm", in_channels=15, gcn_hidden_dim=16, temporal_hidden_dim=16)
        model = build_model(cfg)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = get_loss_fn("smooth_l1")

        batch = next(iter(loader))
        init_weights = model.conv1.lin.weight.clone()

        model.train()
        optimizer.zero_grad()
        preds = model(batch)
        loss = loss_fn(preds, batch.y.view(-1))
        loss.backward()
        optimizer.step()

        # Check weights updated
        self.assertFalse(torch.equal(init_weights, model.conv1.lin.weight))

    def test_07_zero_leakage_assertion(self):
        """7. Verify zero-leakage constraint: current bolus NEVER in input nodes."""
        for g in self.train_graphs[:20]:
            x_np = g.x.cpu().numpy()
            node_types = g.node_types.cpu().numpy()

            # Find nodes at prediction time delta_tau == 0.0
            zero_tau_nodes = [
                i for i in range(g.num_nodes)
                if np.isclose(x_np[i, 1], 0.0, atol=1e-4) and node_types[i] not in [int(NodeType.PATIENT), int(NodeType.TIME_WINDOW)]
            ]

            # Assert NO node at delta_tau == 0.0 is a Bolus node!
            bolus_at_zero = [
                i for i in zero_tau_nodes
                if node_types[i] == int(NodeType.BOLUS_INSULIN)
            ]
            self.assertEqual(
                len(bolus_at_zero), 0,
                f"FATAL: Found {len(bolus_at_zero)} bolus nodes at prediction instant delta_tau=0.0!"
            )

    def test_08_train_val_test_separation(self):
        """8. Verify strict separation of train, val, and test splits."""
        train_ids = {getattr(g, "window_id", "") for g in self.train_graphs}
        val_ids = {getattr(g, "window_id", "") for g in self.val_graphs}
        test_ids = {getattr(g, "window_id", "") for g in self.test_graphs}

        # Sets must be mutually disjoint
        self.assertTrue(train_ids.isdisjoint(val_ids), "Contamination: train and val share window IDs!")
        self.assertTrue(train_ids.isdisjoint(test_ids), "Contamination: train and test share window IDs!")
        self.assertTrue(val_ids.isdisjoint(test_ids), "Contamination: val and test share window IDs!")


if __name__ == "__main__":
    unittest.main()
