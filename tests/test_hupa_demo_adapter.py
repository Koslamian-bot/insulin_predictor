"""
Unit tests for the isolated HUPA demonstration adapter.
"""

import os
import unittest
import torch
import numpy as np

from module1.adapters.hupa_demo_adapter import (
    get_hupa_demo_patient_info,
    extract_hupa_demo_window,
    build_hupa_patient_graph,
    run_hupa_inference,
)
from module2.ontology.schema_types import NodeType


class TestHUPADemoAdapter(unittest.TestCase):

    def test_get_patient_info(self):
        info = get_hupa_demo_patient_info()
        self.assertEqual(info["patient_id"], "HUPA0001P")
        self.assertEqual(info["dataset_source"], "HUPA-UCM")
        self.assertEqual(info["sampling_frequency"], "5 minutes")
        self.assertEqual(info["total_bolus_events"], 76)
        self.assertEqual(info["total_meal_events"], 40)
        self.assertIn("CGM Glucose (glucose)", info["available_modalities"])
        self.assertIn("Skin Temperature (not recorded in HUPA)", info["unavailable_modalities"])

    def test_extract_demo_window_and_zero_leakage(self):
        raw_sub, canonical_sub, slice_obj = extract_hupa_demo_window()
        self.assertEqual(len(raw_sub), 25)
        self.assertEqual(len(canonical_sub), 25)
        self.assertEqual(slice_obj.patient_id, "HUPA0001P")
        self.assertEqual(slice_obj.target_dose, 4.0)

        # Confirm target leakage masking: bolus_dose in feature matrix at prediction instant must be 0.0
        self.assertEqual(canonical_sub.iloc[-1]["bolus_dose"], 0.0)

        # Confirm unavailable modalities are flagged missing without fabrication
        self.assertEqual(canonical_sub.iloc[0]["has_skin_temperature"], 0)
        self.assertEqual(canonical_sub.iloc[0]["skin_temperature_missing_mask"], 1)

    def test_build_knowledge_graph_and_audit(self):
        _, _, slice_obj = extract_hupa_demo_window()
        pyg_data, audit = build_hupa_patient_graph(slice_obj)

        self.assertEqual(pyg_data.num_nodes, 91)
        self.assertEqual(pyg_data.edge_index.shape[1], 237)
        self.assertEqual(audit["is_leak_free"], True)
        self.assertEqual(audit["bolus_nodes_at_zero"], 0)
        self.assertEqual(audit["actual_target_dose"], 4.0)

        # Assert zero bolus at delta_tau == 0.0
        x_np = pyg_data.x.cpu().numpy()
        nt_np = pyg_data.node_types.cpu().numpy()
        zero_tau_bolus = [
            i for i in range(pyg_data.num_nodes)
            if nt_np[i] == int(NodeType.BOLUS_INSULIN) and np.isclose(x_np[i, 1], 0.0, atol=1e-4)
        ]
        self.assertEqual(len(zero_tau_bolus), 0)

    def test_inference_with_existing_checkpoints(self):
        _, _, slice_obj = extract_hupa_demo_window()
        pyg_data, _ = build_hupa_patient_graph(slice_obj)

        for m_name in ["gcn_lstm", "gcn_gru", "gcn_bilstm"]:
            ckpt_path = os.path.join("data", "models", f"{m_name}_best.pt")
            if os.path.exists(ckpt_path):
                res = run_hupa_inference(pyg_data, model_name=m_name)
                self.assertEqual(res["model_name"], m_name.upper())
                self.assertEqual(res["actual_dose"], 4.0)
                self.assertGreater(res["predicted_dose"], 0.0)
                self.assertLess(res["absolute_error"], 5.0)


if __name__ == "__main__":
    unittest.main()
