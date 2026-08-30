"""
Unit and Integration tests for DataValidator.
"""

import os
import unittest
from datetime import datetime
from module1.adapters.registry import AdapterRegistry
from module1.schema.canonical_schema import (
    CanonicalPatientRecord,
    PatientMetadata,
    CanonicalEvent,
    ModalityType,
)
from module1.validation.validator import DataValidator
from module1.validation.validation_report import ValidationSeverity


class TestDataValidator(unittest.TestCase):

    def setUp(self):
        self.validator = DataValidator()

    def test_validation_clean_record(self):
        meta = PatientMetadata(patient_id="101", dataset_source="TestDataset")
        rec = CanonicalPatientRecord(metadata=meta)
        now = datetime(2026, 1, 1, 12, 0, 0)
        rec.add_event(CanonicalEvent(now, ModalityType.CGM_GLUCOSE, 120.0))
        rec.add_event(CanonicalEvent(now, ModalityType.BOLUS_DOSE, 4.5))

        report = self.validator.validate(rec)
        self.assertEqual(report.overall_status, ValidationSeverity.PASS)
        self.assertEqual(report.error_count, 0)

    def test_validation_detects_errors(self):
        meta = PatientMetadata(patient_id="102", dataset_source="TestDataset")
        rec = CanonicalPatientRecord(metadata=meta)
        now = datetime(2026, 1, 1, 12, 0, 0)
        # Impossible negative glucose
        rec.add_event(CanonicalEvent(now, ModalityType.CGM_GLUCOSE, -10.0))
        # Negative bolus
        rec.add_event(CanonicalEvent(now, ModalityType.BOLUS_DOSE, -5.0))

        report = self.validator.validate(rec)
        self.assertEqual(report.overall_status, ValidationSeverity.ERROR)
        self.assertGreaterEqual(report.error_count, 1)

    def test_validation_on_real_ohio_patient(self):
        adapter = AdapterRegistry.get_adapter("ohiot1dm")
        sample_file = r"c:\Final year\Insulin prediction\OhioT1DM\2018\train\559-ws-training.xml"
        rec = adapter.parse_patient(sample_file)

        report = self.validator.validate(rec)
        # Real patient data might have minor warnings (e.g. gaps), but must not have severe system failure
        self.assertIn(report.overall_status, [ValidationSeverity.PASS, ValidationSeverity.WARNING])
        self.assertEqual(report.error_count, 0)
        md = report.to_markdown()
        self.assertIn("Validation Report", md)


if __name__ == "__main__":
    unittest.main()
