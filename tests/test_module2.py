"""
Unit & Integration Test Suite for Module 2.

Covers:
- Ontology parsing and schema validation
- Window slicing and patient boundary safety
- Dynamic graph construction and missing-modality pruning
- STRICT zero-leakage target isolation
- Graph sparsity and connectivity validation
- PyTorch Geometric Data conversion
"""

import unittest
import os
import pandas as pd
import numpy as np

from module2.ontology.schema_types import NodeType, EdgeType, TOTAL_NODE_TYPES, ONTOLOGY_CLASSES
from module2.ontology.ontology_parser import OntologyParser
from module2.graph.window_slicer import WindowSlicer, TimeWindowSlice
from module2.graph.graph_builder import GraphBuilder, GraphDataDict
from module2.graph.graph_validator import GraphValidator
from module2.dataset.pyg_converter import PyGConverter


class TestOntologyParser(unittest.TestCase):
    """Test RDF/OWL parsing of diabetes_insulin_ontology.owl."""

    def test_parse_schema(self):
        parser = OntologyParser()
        schema = parser.parse()

        # Verify all classes
        for expected_cls in [
            "Patient", "TimeWindow", "Observation", "GlucoseObservation",
            "MealObservation", "BasalInsulinAdministration", "BolusInsulinAdministration",
            "AccelerationObservation", "StepObservation", "HeartRateObservation"
        ]:
            self.assertIn(expected_cls, schema.classes)

        # Verify object properties
        self.assertIn("hasTimeWindow", schema.object_properties)
        self.assertIn("hasObservation", schema.object_properties)
        self.assertIn("temporallyPrecedes", schema.object_properties)

        # Verify datatype properties
        self.assertIn("glucoseValue", schema.datatype_properties)
        self.assertIn("carbohydrateAmount", schema.datatype_properties)
        self.assertIn("basalRate", schema.datatype_properties)
        self.assertIn("insulinDose", schema.datatype_properties)


class TestWindowSlicer(unittest.TestCase):
    """Test sliding window generation and boundary isolation."""

    def setUp(self):
        # Create synthetic multi-patient DataFrame
        dates_p1 = pd.date_range("2022-01-01 00:00:00", periods=50, freq="5min")
        dates_p2 = pd.date_range("2022-01-02 00:00:00", periods=50, freq="5min")

        df1 = pd.DataFrame({
            "timestamp": dates_p1,
            "patient_id": "559",
            "cgm_glucose": np.random.uniform(100, 180, 50),
            "cgm_missing_mask": 0,
            "target_current_bolus_dose": 0.0,
            "split_label": "train",
        })
        df2 = pd.DataFrame({
            "timestamp": dates_p2,
            "patient_id": "563",
            "cgm_glucose": np.random.uniform(110, 190, 50),
            "cgm_missing_mask": 0,
            "target_current_bolus_dose": 1.5,
            "split_label": "train",
        })
        self.df = pd.concat([df1, df2], ignore_index=True)

    def test_window_boundaries_and_size(self):
        slicer = WindowSlicer(window_minutes=120, step_minutes=5, stride_minutes=5)
        slices = list(slicer.slice_dataframe(self.df))

        self.assertGreater(len(slices), 0)
        for s in slices:
            # Each window must have exactly 24 steps (120 min / 5 min)
            self.assertEqual(len(s.data), 24)
            # Patient boundary: only one unique patient per window
            self.assertEqual(len(s.data["patient_id"].unique()), 1)
            self.assertIn(s.patient_id, ["559", "563"])
            # Duration must be 115 minutes (23 steps of 5m between start and end)
            diff_mins = (s.prediction_time - s.start_time).total_seconds() / 60
            self.assertEqual(diff_mins, 115)


class TestGraphBuilderAndLeakage(unittest.TestCase):
    """Test graph construction, sparse edges, and strict target isolation."""

    def setUp(self):
        # Build 24-step slice
        dates = pd.date_range("2022-01-01 00:00:00", periods=24, freq="5min")
        df_slice = pd.DataFrame({
            "timestamp": dates,
            "patient_id": "559",
            "cgm_glucose": [120.0 + i * 2 for i in range(24)],
            "cgm_missing_mask": [0] * 24,
            "basal_rate": [0.8] * 24,
            "meal_carbs": [0.0] * 24,
            "bolus_dose": [0.0] * 24,
            "heart_rate": [70.0] * 24,
            "has_heart_rate": [1] * 24,
            "heart_rate_missing_mask": [0] * 24,
            "skin_temperature": [91.5] * 24,
            "has_skin_temperature": [1] * 24,
            "skin_temperature_missing_mask": [0] * 24,
            "air_temperature": [72.0] * 24,
            "has_air_temperature": [1] * 24,
            "air_temperature_missing_mask": [0] * 24,
            "gsr": [0.0002] * 24,
            "has_gsr": [1] * 24,
            "gsr_missing_mask": [0] * 24,
            "acceleration": [0.0] * 24,
            "has_acceleration": [0] * 24,  # 2018 cohort: no accel
            "steps": [0.0] * 24,
            "target_current_bolus_dose": 0.0,
            "split_label": "train",
        })
        # Add a meal and historical bolus at step 5
        df_slice.loc[5, "meal_carbs"] = 45.0
        df_slice.loc[5, "bolus_dose"] = 3.5

        # Add target bolus at prediction instant (step 23)
        df_slice.loc[23, "bolus_dose"] = 5.2
        df_slice.loc[23, "target_current_bolus_dose"] = 5.2

        self.w_slice = TimeWindowSlice(
            window_id="test_window_001",
            patient_id="559",
            start_time=dates[0],
            prediction_time=dates[-1],
            data=df_slice,
            target_dose=5.2,
            split_label="train",
        )
        self.builder = GraphBuilder()
        self.validator = GraphValidator()

    def test_strict_zero_leakage(self):
        graph = self.builder.build_graph(self.w_slice)

        # 1. Verify target value
        self.assertEqual(graph.y, 5.2)

        # 2. Verify NO Bolus node exists at prediction instant delta_tau == 0.0
        for i, ntype in enumerate(graph.node_types):
            delta_tau = graph.x[i, 1]
            if ntype == NodeType.BOLUS_INSULIN:
                self.assertLess(delta_tau, 0.0, "FATAL: Bolus node found at prediction instant!")
                self.assertEqual(graph.x[i, 0], 3.5, "Historical bolus dose should be 3.5")

        # 3. Verify validator passes
        is_valid, issues = self.validator.validate(graph)
        self.assertTrue(is_valid, f"Validation failed: {issues}")

    def test_missing_modality_pruning(self):
        graph = self.builder.build_graph(self.w_slice)

        # Acceleration was marked has_acceleration = 0
        self.assertNotIn(NodeType.ACCELERATION_OBSERVATION, graph.node_types)

        # Steps was 0.0 everywhere
        self.assertNotIn(NodeType.STEP_OBSERVATION, graph.node_types)

        # Meal was only at step 5 -> exactly 1 meal node
        meal_count = sum(1 for nt in graph.node_types if nt == NodeType.MEAL_OBSERVATION)
        self.assertEqual(meal_count, 1)

    def test_sparse_edges(self):
        graph = self.builder.build_graph(self.w_slice)
        num_nodes = len(graph.node_types)
        num_edges = graph.edge_index.shape[1]

        # In dense transitive all-to-all, edges would be ~ num_nodes*(num_nodes-1)
        max_dense_edges = num_nodes * (num_nodes - 1)
        self.assertLess(num_edges, max_dense_edges * 0.25, "Graph is too dense! Not sparse sequential.")


class TestPyGConverter(unittest.TestCase):
    """Test PyTorch Geometric Data object construction."""

    def test_conversion(self):
        dates = pd.date_range("2022-01-01 00:00:00", periods=24, freq="5min")
        df_slice = pd.DataFrame({
            "timestamp": dates,
            "patient_id": "559",
            "cgm_glucose": [130.0] * 24,
            "cgm_missing_mask": [0] * 24,
            "basal_rate": [0.7] * 24,
            "meal_carbs": [0.0] * 24,
            "bolus_dose": [0.0] * 24,
            "heart_rate": [65.0] * 24,
            "has_heart_rate": [1] * 24,
            "heart_rate_missing_mask": [0] * 24,
            "skin_temperature": [90.0] * 24,
            "has_skin_temperature": [1] * 24,
            "skin_temperature_missing_mask": [0] * 24,
            "air_temperature": [70.0] * 24,
            "has_air_temperature": [1] * 24,
            "air_temperature_missing_mask": [0] * 24,
            "gsr": [0.00015] * 24,
            "has_gsr": [1] * 24,
            "gsr_missing_mask": [0] * 24,
            "acceleration": [0.0] * 24,
            "has_acceleration": [0] * 24,
            "steps": [0.0] * 24,
            "target_current_bolus_dose": 2.0,
            "split_label": "test",
        })
        w_slice = TimeWindowSlice(
            window_id="test_pyg_window",
            patient_id="559",
            start_time=dates[0],
            prediction_time=dates[-1],
            data=df_slice,
            target_dose=2.0,
            split_label="test",
        )

        builder = GraphBuilder()
        g_dict = builder.build_graph(w_slice)
        pyg_data = PyGConverter.to_pyg_data(g_dict)

        # Test tensor dimensions
        self.assertEqual(pyg_data.x.ndim, 2)
        self.assertEqual(pyg_data.x.shape[1], 15)  # 15 features
        self.assertEqual(pyg_data.edge_index.shape[0], 2)
        self.assertEqual(pyg_data.edge_attr.ndim, 1)
        self.assertEqual(pyg_data.y.item(), 2.0)
        self.assertEqual(pyg_data.patient_id, "559")
        self.assertEqual(pyg_data.window_id, "test_pyg_window")


if __name__ == "__main__":
    unittest.main()
