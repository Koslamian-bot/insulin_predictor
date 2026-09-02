"""
Ontology-Grounded Dynamic Graph Builder.

Constructs patient-specific temporal subgraphs from time-window slices,
enforcing strict zero-leakage target isolation, dynamic missing-modality pruning,
and sparse sequential edge connectivity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import pandas as pd

from module2.ontology.schema_types import (
    NodeType,
    EdgeType,
    TOTAL_NODE_TYPES,
)
from module2.graph.window_slicer import TimeWindowSlice

logger = logging.getLogger(__name__)


@dataclass
class GraphDataDict:
    """
    In-memory graph container ready for PyTorch Geometric conversion.
    """
    x: np.ndarray             # Shape: [num_nodes, 15] (value, delta_tau, one-hot[13])
    edge_index: np.ndarray    # Shape: [2, num_edges]
    edge_attr: np.ndarray     # Shape: [num_edges]
    y: float                  # Target bolus dose at time t
    node_types: List[NodeType]# Type of each node
    window_id: str
    patient_id: str
    timestamp: str


class GraphBuilder:
    """
    Transforms a TimeWindowSlice into a GraphDataDict using ontology rules.
    """

    def __init__(self, normalize_values: bool = True):
        self.normalize_values = normalize_values
        self.feature_dim = 1 + 1 + TOTAL_NODE_TYPES  # 1 (val) + 1 (tau) + 13 (one-hot) = 15

    def build_graph(self, window_slice: TimeWindowSlice) -> GraphDataDict:
        """
        Build a graph for a single time window.
        """
        df = window_slice.data
        n_steps = len(df)
        pred_idx = n_steps - 1  # Index of prediction instant t (typically 23)

        nodes: List[Dict[str, Any]] = []
        edges: List[Tuple[int, int, EdgeType]] = []  # (src, dst, type)

        # -------------------------------------------------------------
        # 1. Structural Anchor Nodes: Patient & TimeWindow
        # -------------------------------------------------------------
        # Node 0: Patient Hub Node
        patient_node_idx = 0
        nodes.append({
            "type": NodeType.PATIENT,
            "value": 0.0,
            "delta_tau": 0.0,
            "step": None,
        })

        # Node 1: TimeWindow Node
        window_node_idx = 1
        nodes.append({
            "type": NodeType.TIME_WINDOW,
            "value": 1.0,
            "delta_tau": 0.0,
            "step": None,
        })

        # Edge: Patient -> TimeWindow (hasTimeWindow)
        edges.append((patient_node_idx, window_node_idx, EdgeType.HAS_TIME_WINDOW))

        # Track observation nodes per modality for sequential temporal edges
        modality_nodes: Dict[NodeType, List[int]] = {nt: [] for nt in NodeType}
        # Track observation nodes per time step for contemporaneous edges
        step_nodes: Dict[int, List[int]] = {k: [] for k in range(n_steps)}

        # -------------------------------------------------------------
        # 2. Iterate Chronologically Across Time Steps [0, pred_idx]
        # -------------------------------------------------------------
        for k in range(n_steps):
            row = df.iloc[k]
            # Normalized relative time offset: tau in [-1.0, 0.0]
            delta_tau = float((k - pred_idx) / max(1, pred_idx))

            # A. Glucose Observation (CGM)
            cgm_val = row.get("cgm_glucose")
            if pd.notna(cgm_val) and row.get("cgm_missing_mask", 0) == 0:
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.GLUCOSE_OBSERVATION,
                    "value": float(cgm_val),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.GLUCOSE_OBSERVATION].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

            # B. Meal Observation (Carbohydrates) - Created only if carbs present
            meal_val = row.get("meal_carbs", 0.0)
            if pd.notna(meal_val) and float(meal_val) > 0.0:
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.MEAL_OBSERVATION,
                    "value": float(meal_val),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.MEAL_OBSERVATION].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

            # C. Scheduled Basal Insulin
            basal_val = row.get("basal_rate")
            if pd.notna(basal_val):
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.BASAL_INSULIN,
                    "value": float(basal_val),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.BASAL_INSULIN].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

            # D. Temporary Basal Rate Override (if active)
            temp_basal = row.get("temp_basal_rate")
            if pd.notna(temp_basal):
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.TEMP_BASAL_INSULIN,
                    "value": float(temp_basal),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.TEMP_BASAL_INSULIN].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

            # E. Bolus Insulin Delivery - STRICT ZERO-LEAKAGE ENFORCEMENT
            bolus_val = row.get("bolus_dose", 0.0)
            if pd.notna(bolus_val) and float(bolus_val) > 0.0:
                # Rule: ONLY historical boluses (strictly before prediction instant k < pred_idx)
                # are permitted to become input nodes in the graph!
                if k < pred_idx:
                    idx = len(nodes)
                    nodes.append({
                        "type": NodeType.BOLUS_INSULIN,
                        "value": float(bolus_val),
                        "delta_tau": delta_tau,
                        "step": k,
                    })
                    modality_nodes[NodeType.BOLUS_INSULIN].append(idx)
                    step_nodes[k].append(idx)
                    edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))
                else:
                    # At k == pred_idx (time t), DO NOT create a Bolus node!
                    # The bolus at t is the prediction target.
                    pass

            # F. Continuous Wearables (Heart Rate) - 2018 cohort
            hr_val = row.get("heart_rate")
            if pd.notna(hr_val) and row.get("has_heart_rate", 1) == 1 and row.get("heart_rate_missing_mask", 0) == 0:
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.HEART_RATE_OBSERVATION,
                    "value": float(hr_val),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.HEART_RATE_OBSERVATION].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

            # G. Skin Temperature
            st_val = row.get("skin_temperature")
            if pd.notna(st_val) and row.get("has_skin_temperature", 1) == 1 and row.get("skin_temperature_missing_mask", 0) == 0:
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.SKIN_TEMP_OBSERVATION,
                    "value": float(st_val),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.SKIN_TEMP_OBSERVATION].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

            # H. Air Temperature - 2018 cohort
            at_val = row.get("air_temperature")
            if pd.notna(at_val) and row.get("has_air_temperature", 1) == 1 and row.get("air_temperature_missing_mask", 0) == 0:
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.AIR_TEMP_OBSERVATION,
                    "value": float(at_val),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.AIR_TEMP_OBSERVATION].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

            # I. Galvanic Skin Response (GSR)
            gsr_val = row.get("gsr")
            if pd.notna(gsr_val) and row.get("has_gsr", 1) == 1 and row.get("gsr_missing_mask", 0) == 0:
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.GSR_OBSERVATION,
                    "value": float(gsr_val),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.GSR_OBSERVATION].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

            # J. 3-Axis Acceleration - 2020 cohort
            acc_val = row.get("acceleration")
            if pd.notna(acc_val) and row.get("has_acceleration", 1) == 1 and row.get("acceleration_missing_mask", 0) == 0:
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.ACCELERATION_OBSERVATION,
                    "value": float(acc_val),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.ACCELERATION_OBSERVATION].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

            # K. Steps - 2018 cohort (if steps > 0)
            step_val = row.get("steps", 0.0)
            if pd.notna(step_val) and float(step_val) > 0.0:
                idx = len(nodes)
                nodes.append({
                    "type": NodeType.STEP_OBSERVATION,
                    "value": float(step_val),
                    "delta_tau": delta_tau,
                    "step": k,
                })
                modality_nodes[NodeType.STEP_OBSERVATION].append(idx)
                step_nodes[k].append(idx)
                edges.append((window_node_idx, idx, EdgeType.HAS_OBSERVATION))

        # -------------------------------------------------------------
        # 3. Sparse Consecutive Temporal Edges: t_i -> t_(i+1)
        # -------------------------------------------------------------
        for mod_type, node_indices in modality_nodes.items():
            if len(node_indices) > 1:
                for i in range(len(node_indices) - 1):
                    src = node_indices[i]
                    dst = node_indices[i + 1]
                    edges.append((src, dst, EdgeType.SEQUENTIAL_TEMPORAL))

        # -------------------------------------------------------------
        # 4. Contemporaneous Cross-Modal Edges at Step k
        # -------------------------------------------------------------
        for k, step_node_list in step_nodes.items():
            # If both a meal/bolus and glucose exist at this step, connect them
            if len(step_node_list) > 1:
                # Find glucose node if present
                cgm_nodes_at_k = [
                    n_idx for n_idx in step_node_list
                    if nodes[n_idx]["type"] == NodeType.GLUCOSE_OBSERVATION
                ]
                other_nodes_at_k = [
                    n_idx for n_idx in step_node_list
                    if nodes[n_idx]["type"] != NodeType.GLUCOSE_OBSERVATION
                ]
                if cgm_nodes_at_k:
                    g_idx = cgm_nodes_at_k[0]
                    for other_idx in other_nodes_at_k:
                        edges.append((other_idx, g_idx, EdgeType.CONTEMPORANEOUS))

        # -------------------------------------------------------------
        # 5. Construct Node Feature Matrix X [num_nodes, 15]
        # -------------------------------------------------------------
        num_nodes = len(nodes)
        x_matrix = np.zeros((num_nodes, self.feature_dim), dtype=np.float32)

        for i, n in enumerate(nodes):
            val = n["value"]
            tau = n["delta_tau"]
            ntype = n["type"]

            x_matrix[i, 0] = val
            x_matrix[i, 1] = tau
            x_matrix[i, 2 + int(ntype)] = 1.0  # One-hot encoding

        # -------------------------------------------------------------
        # 6. Construct Edge Indices and Attributes
        # -------------------------------------------------------------
        if edges:
            edge_index = np.array([[e[0] for e in edges], [e[1] for e in edges]], dtype=np.int64)
            edge_attr = np.array([int(e[2]) for e in edges], dtype=np.int64)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0,), dtype=np.int64)

        pred_time_str = window_slice.prediction_time.strftime("%Y-%m-%d %H:%M:%S")

        return GraphDataDict(
            x=x_matrix,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=float(window_slice.target_dose),
            node_types=[n["type"] for n in nodes],
            window_id=window_slice.window_id,
            patient_id=window_slice.patient_id,
            timestamp=pred_time_str,
        )
