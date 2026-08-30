"""
Integration test for OhioT1DMAdapter against real OhioT1DM files.
"""

import os
import unittest
from module1.adapters.registry import AdapterRegistry
from module1.adapters.ohio_t1dm_adapter import OhioT1DMAdapter
from module1.schema.canonical_schema import ModalityType, CanonicalPatientRecord


class TestOhioT1DMAdapter(unittest.TestCase):

    def setUp(self):
        self.raw_data_dir = r"c:\Final year\Insulin prediction\OhioT1DM"
        self.adapter = AdapterRegistry.get_adapter("ohiot1dm")

    def test_registry_resolution(self):
        self.assertIsInstance(self.adapter, OhioT1DMAdapter)
        self.assertIn("ohiot1dm", AdapterRegistry.list_adapters())
        self.assertIn("hupa_ucm", AdapterRegistry.list_adapters())

    def test_discover_files(self):
        files = self.adapter.discover_files(self.raw_data_dir)
        self.assertEqual(len(files), 24, f"Expected 24 XML files, found {len(files)}")

    def test_parse_2018_patient(self):
        sample_file = os.path.join(self.raw_data_dir, "2018", "train", "559-ws-training.xml")
        record = self.adapter.parse_patient(sample_file)

        self.assertIsInstance(record, CanonicalPatientRecord)
        self.assertEqual(record.metadata.patient_id, "559")
        self.assertEqual(record.metadata.cohort, "2018")
        self.assertEqual(record.metadata.split_label, "train")
        self.assertEqual(record.metadata.weight_kg, 99.0)
        self.assertEqual(record.metadata.insulin_type, "Novalog")

        modalities = record.get_modalities_present()
        self.assertIn(ModalityType.CGM_GLUCOSE, modalities)
        self.assertIn(ModalityType.BOLUS_DOSE, modalities)
        self.assertIn(ModalityType.BASAL_RATE, modalities)
        self.assertIn(ModalityType.MEAL_CARBS, modalities)
        self.assertIn(ModalityType.HEART_RATE, modalities)
        self.assertIn(ModalityType.GSR, modalities)
        self.assertIn(ModalityType.SKIN_TEMPERATURE, modalities)

        # Check total events > 10,000
        counts = record.get_event_counts()
        self.assertEqual(counts["cgm_glucose"], 10796)
        self.assertEqual(counts["bolus_dose"], 152)
        self.assertEqual(counts["meal_carbs"], 150)

        # Verify sorted
        timestamps = [e.timestamp for e in record.events]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_parse_2020_patient(self):
        sample_file = os.path.join(self.raw_data_dir, "2020", "train", "540-ws-training.xml")
        record = self.adapter.parse_patient(sample_file)

        self.assertEqual(record.metadata.patient_id, "540")
        self.assertEqual(record.metadata.cohort, "2020")
        self.assertEqual(record.metadata.split_label, "train")

        modalities = record.get_modalities_present()
        self.assertIn(ModalityType.CGM_GLUCOSE, modalities)
        self.assertIn(ModalityType.ACCELERATION, modalities)
        self.assertIn(ModalityType.BOLUS_DOSE, modalities)

        counts = record.get_event_counts()
        self.assertEqual(counts["cgm_glucose"], 11947)
        self.assertEqual(counts["acceleration"], 28353)


if __name__ == "__main__":
    unittest.main()
