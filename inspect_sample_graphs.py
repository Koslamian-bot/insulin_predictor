"""
Interactive Sanity Check & Graph Inspection Script.

Inspects 3 real patient windows from canonical_train.csv, builds their
ontology graphs, prints their structure in detail, and verifies zero-leakage.
"""

import os
import pandas as pd
import numpy as np

from module2.ontology.schema_types import NodeType, EdgeType
from module2.graph.window_slicer import WindowSlicer
from module2.graph.graph_builder import GraphBuilder
from module2.graph.graph_validator import GraphValidator
from module2.dataset.pyg_converter import PyGConverter


def inspect_graphs(sample_count: int = 3):
    print("\n==================================================")
    print("MANUAL GRAPH ANATOMY INSPECTION (3 SAMPLE WINDOWS)")
    print("==================================================")

    data_path = os.path.join("data", "processed", "canonical_train.csv")
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        return

    df = pd.read_csv(data_path, nrows=500)
    slicer = WindowSlicer(window_minutes=120, stride_minutes=15)
    builder = GraphBuilder()
    validator = GraphValidator()

    windows = list(slicer.slice_dataframe(df))
    print(f"Total candidate windows extracted from first 500 rows: {len(windows)}")

    # Pick windows: one with bolus target if available
    bolus_windows = [w for w in windows if w.target_dose > 0]
    normal_windows = [w for w in windows if w.target_dose == 0]

    selected = (bolus_windows[:1] + normal_windows[:2]) if bolus_windows else windows[:sample_count]

    for idx, w in enumerate(selected[:sample_count], 1):
        print(f"\n--- [SAMPLE {idx}] WINDOW: {w.window_id} ---")
        print(f"Patient ID: {w.patient_id} | Split: {w.split_label}")
        print(f"Timeline:   {w.start_time}  -->  {w.prediction_time} (Duration: 120m)")
        print(f"Ground-Truth Bolus Target (y): {w.target_dose:.2f} Units")

        g_dict = builder.build_graph(w)
        is_valid, issues = validator.validate(g_dict)

        print(f"Validator Result: {'[PASS]' if is_valid else '[FAIL]'}")
        if issues:
            print(f"Issues: {issues}")

        # Breakdown of nodes
        node_type_counts = {}
        for nt in g_dict.node_types:
            node_type_counts[nt.name] = node_type_counts.get(nt.name, 0) + 1

        print(f"Total Nodes: {len(g_dict.node_types)}")
        for name, count in sorted(node_type_counts.items()):
            print(f"  • {name}: {count}")

        # Breakdown of edges
        edge_type_counts = {}
        for et_val in g_dict.edge_attr:
            et = EdgeType(et_val)
            edge_type_counts[et.name] = edge_type_counts.get(et.name, 0) + 1

        print(f"Total Edges: {g_dict.edge_index.shape[1]}")
        for name, count in sorted(edge_type_counts.items()):
            print(f"  • {name}: {count}")

        # PyG conversion check
        pyg = PyGConverter.to_pyg_data(g_dict)
        print("PyTorch Geometric Data Object:")
        print(f"  x shape:          {list(pyg.x.shape)} (dtype: {pyg.x.dtype})")
        print(f"  edge_index shape: {list(pyg.edge_index.shape)} (dtype: {pyg.edge_index.dtype})")
        print(f"  edge_attr shape:  {list(pyg.edge_attr.shape)}")
        print(f"  y tensor:         {pyg.y} (target={pyg.y.item():.2f} U)")

        # Target leakage proof
        delta_tau_zero_nodes = [i for i, tau in enumerate(pyg.x[:, 1]) if np.isclose(tau.item(), 0.0)]
        bolus_nodes_at_zero = [
            i for i in delta_tau_zero_nodes
            if pyg.node_types[i].item() == int(NodeType.BOLUS_INSULIN)
        ]
        print(f"  Leakage Audit: Bolus nodes at delta_tau=0.0: {len(bolus_nodes_at_zero)} (Expected: 0)")
        assert len(bolus_nodes_at_zero) == 0, "FATAL: Leakage found!"

    print("\n==================================================")
    print("ALL 3 SAMPLE WINDOWS AUDITED AND VERIFIED LEAK-FREE")
    print("==================================================\n")


if __name__ == "__main__":
    inspect_graphs(3)
