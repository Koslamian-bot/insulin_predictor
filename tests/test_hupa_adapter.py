"""
Integration test for HUPAUCMAdapter against real HUPA-UCM files.
"""

import os
import unittest
from module1.adapters.registry import AdapterRegistry
from module1.adapters.hupa_ucm_adapter import HUPAUCMAdapter
from module1.schema.canonical_schema import ModalityType, CanonicalPatientRecord


class TestHUPAUCMAdapter(unittest.TestCase):

    def setUp(self):
        self.raw_data_dir = r"c:\Final year\Insulin prediction\HUPA-UCM Diabetes Dataset\Preprocessed"
        self.adapter = AdapterRegistry.get_adapter("hupa_ucm")

    def test_discover_hupa_files(self):
        if os.path.exists(self.raw_data_dir):
            files = self.adapter.discover_files(self.raw_data_dir)
            self.assertEqual(len(files), 25)

    def test_parse_hupa_patient(self):
        sample_file = os.path.join(self.raw_data_dir, "HUPA0001P.csv")
        if os.path.exists(sample_file):
            record = self.adapter.parse_patient(sample_file)
            self.assertIsInstance(record, CanonicalPatientRecord)
            self.assertEqual(record.metadata.patient_id, "HUPA0001P")
            self.assertEqual(record.metadata.dataset_source, "HUPA-UCM")

            modalities = record.get_modalities_present()
            self.assertIn(ModalityType.CGM_GLUCOSE, modalities)
            self.assertIn(ModalityType.HEART_RATE, modalities)
            self.assertIn(ModalityType.BASAL_RATE, modalities)
            self.assertGreater(len(record.events), 1000)


if __name__ == "__main__":
    unittest.main()
