"""
Automated Graph Integrity & Leakage Validator.

Performs strict automated audits on constructed graph instances:
- Verifies zero-leakage constraints (ensures target bolus is absent from input X).
- Verifies graph sparsity (ensures sequential edges rather than dense transitive explosion).
- Validates tensor dimensions and boundary indices.
"""

from typing import List, Tuple, Dict, Any, Union
import numpy as np

from module2.ontology.schema_types import NodeType, TOTAL_NODE_TYPES
from module2.graph.graph_builder import GraphDataDict


class GraphValidator:
    """
    Validates graph structures before feeding into PyTorch Geometric or Module 3.
    """

    def __init__(self, expected_feature_dim: int = 15):
        self.expected_feature_dim = expected_feature_dim

    def validate(self, graph: GraphDataDict) -> Tuple[bool, List[str]]:
        """
        Run complete audit on a GraphDataDict.
        Returns:
            (is_valid, list_of_issues)
        """
        issues: List[str] = []
        num_nodes = len(graph.node_types)

        # 1. Dimension & Shape Checks
        if graph.x.shape[0] != num_nodes:
            issues.append(f"Node count mismatch: x rows={graph.x.shape[0]} vs node_types={num_nodes}")

        if graph.x.shape[1] != self.expected_feature_dim:
            issues.append(f"Feature dim mismatch: expected {self.expected_feature_dim}, got {graph.x.shape[1]}")

        if graph.edge_index.shape[0] != 2:
            issues.append(f"Edge index format invalid: shape={graph.edge_index.shape}, expected [2, E]")

        num_edges = graph.edge_index.shape[1]
        if len(graph.edge_attr) != num_edges:
            issues.append(f"Edge attribute count mismatch: len(edge_attr)={len(graph.edge_attr)} vs num_edges={num_edges}")

        # 2. Edge Index Boundary Checks
        if num_edges > 0:
            if graph.edge_index.min() < 0:
                issues.append("Edge index contains negative indices.")
            if graph.edge_index.max() >= num_nodes:
                issues.append(f"Edge index out of bounds: max={graph.edge_index.max()} >= num_nodes={num_nodes}")

        # 3. Structural Anchor Checks
        if num_nodes < 2:
            issues.append("Graph lacks mandatory Patient and TimeWindow anchor nodes.")
        else:
            if graph.node_types[0] != NodeType.PATIENT:
                issues.append(f"Node 0 is not Patient: got {graph.node_types[0]}")
            if graph.node_types[1] != NodeType.TIME_WINDOW:
                issues.append(f"Node 1 is not TimeWindow: got {graph.node_types[1]}")

        # 4. STRICT ZERO-LEAKAGE AUDIT
        # Rule: No Bolus node may exist at prediction instant t (where delta_tau == 0.0)
        for i, ntype in enumerate(graph.node_types):
            delta_tau = graph.x[i, 1]
            if ntype == NodeType.BOLUS_INSULIN:
                # If delta_tau is 0.0, this is a fatal data leakage violation!
                if np.isclose(delta_tau, 0.0, atol=1e-4):
                    issues.append(
                        f"CRITICAL LEAKAGE: Bolus node {i} exists at prediction instant delta_tau={delta_tau:.4f}!"
                    )
                # Ensure bolus value in historical nodes is positive
                if graph.x[i, 0] <= 0.0:
                    issues.append(f"Historical bolus node {i} has non-positive dose {graph.x[i, 0]}")

        # 5. Sparsity Audit
        # Check that edge density does not explode to dense O(N^2) transitive closure
        if num_nodes > 5:
            max_possible_edges = num_nodes * (num_nodes - 1)
            density = num_edges / max(1, max_possible_edges)
            if density > 0.40:
                issues.append(
                    f"Graph over-density warning: density={density:.3f} > 0.40 (possible transitive edge explosion)"
                )

        # 6. Target Validity
        if graph.y < 0.0:
            issues.append(f"Negative target dosage: y={graph.y}")

        is_valid = len(issues) == 0
        return is_valid, issues
