"""
Proposed Model: GCN + Temporal Deep Learning Architecture.

Combines graph representation learning over the patient Knowledge Graph
with chronological recurrent modeling (LSTM, GRU, or BiLSTM) over the 24-step window.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from module3.models.base import BaseModel
from module3.config import ModelConfig
from module3.data.graph_dataset import extract_node_step_indices


class GCNTemporalModel(BaseModel):
    """
    Proposed Architecture: GCNConv layers -> Step-wise Temporal Pooling -> Recurrent Core -> Regression Head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.gcn_dim = config.gcn_hidden_dim
        self.temp_dim = config.temporal_hidden_dim
        self.num_steps = config.num_steps
        self.bidirectional = config.bidirectional

        # -------------------------------------------------------------
        # 1. Graph Convolutional Encoder (GCNConv)
        # -------------------------------------------------------------
        self.conv1 = GCNConv(self.in_channels, self.gcn_dim)
        self.bn1 = nn.BatchNorm1d(self.gcn_dim)
        self.conv2 = GCNConv(self.gcn_dim, self.gcn_dim)
        self.bn2 = nn.BatchNorm1d(self.gcn_dim)
        self.dropout = nn.Dropout(config.dropout)

        # -------------------------------------------------------------
        # 2. Recurrent Temporal Core (LSTM, GRU, or BiLSTM)
        # -------------------------------------------------------------
        recurrent_type = config.model_type.lower()
        if "gru" in recurrent_type:
            self.rnn = nn.GRU(
                input_size=self.gcn_dim,
                hidden_size=self.temp_dim,
                num_layers=config.num_temporal_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=config.dropout if config.num_temporal_layers > 1 else 0.0,
            )
            self.rnn_type = "gru"
        else:
            # Default to LSTM / BiLSTM
            self.rnn = nn.LSTM(
                input_size=self.gcn_dim,
                hidden_size=self.temp_dim,
                num_layers=config.num_temporal_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
                dropout=config.dropout if config.num_temporal_layers > 1 else 0.0,
            )
            self.rnn_type = "lstm"

        rnn_out_dim = self.temp_dim * (2 if self.bidirectional else 1)

        # -------------------------------------------------------------
        # 3. Multilayer Perceptron Regression Head
        # -------------------------------------------------------------
        # Consumes concatenated temporal state + global graph context
        combined_dim = rnn_out_dim + self.gcn_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
            nn.Softplus(beta=1.0),  # Strictly non-negative, non-zero gradient everywhere
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Args:
            batch: PyTorch Geometric Batch object
        Returns:
            y_pred: [B] predicted bolus dosage
        """
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        device = x.device
        num_graphs = batch.num_graphs

        # Step 1: Graph Convolutions
        h = F.relu(self.bn1(self.conv1(x, edge_index)))
        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h, edge_index)))
        h = self.dropout(h)  # [N_total, gcn_dim]

        # Step 2: Extract 24-step temporal sequence via step-wise pooling
        step_indices = extract_node_step_indices(batch.x, batch.node_types, num_steps=self.num_steps)

        # Separate temporal observations from static anchors
        temp_mask = step_indices >= 0
        anchor_mask = ~temp_mask

        # Dense 3D sequence tensor: [B, 24, gcn_dim]
        seq_tensor = torch.zeros((num_graphs, self.num_steps, self.gcn_dim), dtype=torch.float32, device=device)
        counts = torch.zeros((num_graphs, self.num_steps, 1), dtype=torch.float32, device=device)

        if temp_mask.any():
            v_graphs = batch_idx[temp_mask]
            v_steps = step_indices[temp_mask]
            v_h = h[temp_mask]

            seq_tensor.index_put_((v_graphs, v_steps), v_h, accumulate=True)
            counts.index_put_((v_graphs, v_steps), torch.ones((v_graphs.shape[0], 1), device=device), accumulate=True)

            mask = counts > 0
            seq_tensor = torch.where(mask, seq_tensor / torch.clamp(counts, min=1.0), seq_tensor)

        # Global anchor context (Patient + TimeWindow)
        if anchor_mask.any():
            anchor_h = h[anchor_mask]
            anchor_batch = batch_idx[anchor_mask]
            global_context = global_mean_pool(anchor_h, anchor_batch, size=num_graphs)
        else:
            global_context = global_mean_pool(h, batch_idx, size=num_graphs)

        # Step 3: Recurrent Temporal Processing over the 24 chronological steps
        if self.rnn_type == "lstm":
            out_seq, (h_n, c_n) = self.rnn(seq_tensor)
        else:
            out_seq, h_n = self.rnn(seq_tensor)

        # State at final time step (prediction instant t = step 23)
        h_last = out_seq[:, -1, :]  # [B, rnn_out_dim]

        # Step 4: Concatenate with global anchor context and feed to regression head
        combined = torch.cat([h_last, global_context], dim=-1)
        y_pred = self.head(combined).squeeze(-1)  # [B]

        return y_pred
