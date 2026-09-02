"""
Automated Verification Suite for Demo Module.

Verifies:
- Data loader functions on real disk artifacts
- Dynamic zero-leakage audit logic on real PyG Data objects
- Vis-network translation logic
- Plotly timeline generation
- Schema loading from diabetes_insulin_ontology.owl
"""

import unittest
import os
import pandas as pd
import torch

from demo.data_loader import (
    load_canonical_df,
    get_dataset_metadata,
    load_graph_split,
    load_ontology_schema,
    load_quality_report,
    load_graph_report_metrics,
    get_raw_to_canonical_example,
)
from demo.components.graph_visualizer import (
    parse_pyg_to_network,
    audit_target_isolation,
)
from demo.components.timeline_chart import render_patient_timeline
from module2.ontology.schema_types import NodeType


class TestDemoModule(unittest.TestCase):
    """Test suite ensuring Demo Module functions without error."""

    def test_data_loader_metadata(self):
        meta = get_dataset_metadata()
        self.assertIn("train", meta)
        self.assertGreater(meta["train"]["row_count"], 0)

    def test_canonical_df_loading(self):
        df = load_canonical_df("train", max_rows=50)
        self.assertFalse(df.empty)
        self.assertIn("cgm_glucose", df.columns)
        self.assertIn("timestamp", df.columns)

    def test_ontology_loading(self):
        schema = load_ontology_schema()
        self.assertIsNotNone(schema)
        self.assertGreaterEqual(len(schema.classes), 18)
        self.assertIn("hasTimeWindow", schema.object_properties)

    def test_real_pyg_graph_loading_and_audit(self):
        graphs = load_graph_split("train")
        self.assertGreater(len(graphs), 0)

        # Audit first graph
        g = graphs[0]
        audit = audit_target_isolation(g)
        self.assertTrue(audit["is_leak_free"])
        self.assertEqual(audit["bolus_nodes_at_zero_count"], 0)
        self.assertGreater(audit["total_nodes"], 0)
        self.assertGreater(audit["total_edges"], 0)

        # Vis network translation
        nodes, edges = parse_pyg_to_network(g)
        self.assertEqual(len(nodes), g.num_nodes)
        self.assertEqual(len(edges), g.edge_index.shape[1])

    def test_patient_timeline_chart(self):
        df = load_canonical_df("train", max_rows=100)
        p_id = df["patient_id"].iloc[0]
        fig = render_patient_timeline(df, str(p_id))
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_raw_to_canonical_example(self):
        ex = get_raw_to_canonical_example()
        self.assertIn("raw_xml_glucose", ex)
        self.assertIn("synchronized_5min_row", ex)


if __name__ == "__main__":
    unittest.main()
