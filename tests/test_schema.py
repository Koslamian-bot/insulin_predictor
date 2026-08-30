"""
Unit tests for Module 1 Canonical Schema using standard unittest.
"""

from datetime import datetime
import unittest
from module1.schema.canonical_schema import (
    ModalityType,
    CanonicalEvent,
    PatientMetadata,
    CanonicalPatientRecord,
    CANONICAL_UNITS,
    PHYSIOLOGICAL_BOUNDS,
)


class TestCanonicalSchema(unittest.TestCase):

    def test_modality_enum_and_bounds(self):
        self.assertEqual(ModalityType.CGM_GLUCOSE.value, "cgm_glucose")
        self.assertEqual(ModalityType.BOLUS_DOSE.value, "bolus_dose")
        self.assertEqual(CANONICAL_UNITS[ModalityType.CGM_GLUCOSE], "mg/dL")
        self.assertEqual(CANONICAL_UNITS[ModalityType.BOLUS_DOSE], "Units")
        self.assertEqual(PHYSIOLOGICAL_BOUNDS[ModalityType.CGM_GLUCOSE], (20.0, 600.0))

    def test_canonical_event_serialization(self):
        now = datetime(2026, 8, 30, 12, 0, 0)
        event = CanonicalEvent(
            timestamp=now,
            modality=ModalityType.CGM_GLUCOSE,
            value=142.5,
            attributes={"source": "Dexcom G4"}
        )
        d = event.to_dict()
        self.assertEqual(d["timestamp"], "2026-08-30T12:00:00")
        self.assertEqual(d["modality"], "cgm_glucose")
        self.assertEqual(d["value"], 142.5)
        self.assertEqual(d["attributes"]["source"], "Dexcom G4")

        reconstructed = CanonicalEvent.from_dict(d)
        self.assertEqual(reconstructed.timestamp, now)
        self.assertEqual(reconstructed.modality, ModalityType.CGM_GLUCOSE)
        self.assertEqual(reconstructed.value, 142.5)
        self.assertEqual(reconstructed.attributes["source"], "Dexcom G4")

    def test_canonical_patient_record(self):
        meta = PatientMetadata(
            patient_id="559",
            dataset_source="OhioT1DM",
            cohort="2018",
            split_label="train",
            weight_kg=99.0,
            insulin_type="Novalog",
        )
        record = CanonicalPatientRecord(metadata=meta)
        
        t1 = datetime(2022, 1, 1, 10, 5, 0)
        t0 = datetime(2022, 1, 1, 10, 0, 0)
        record.add_event(CanonicalEvent(t1, ModalityType.CGM_GLUCOSE, 120.0))
        record.add_event(CanonicalEvent(t0, ModalityType.CGM_GLUCOSE, 110.0))
        record.add_event(CanonicalEvent(t0, ModalityType.BOLUS_DOSE, 5.0, attributes={"type": "normal"}))

        # Verify sorting
        record.sort_events()
        self.assertEqual(record.events[0].timestamp, t0)
        self.assertEqual(record.events[-1].timestamp, t1)

        # Verify counts and modalities
        counts = record.get_event_counts()
        self.assertEqual(counts["cgm_glucose"], 2)
        self.assertEqual(counts["bolus_dose"], 1)
        self.assertIn(ModalityType.CGM_GLUCOSE, record.get_modalities_present())
        self.assertIn(ModalityType.BOLUS_DOSE, record.get_modalities_present())

        # Verify DataFrame conversion
        df = record.to_dataframe()
        self.assertEqual(len(df), 3)
        self.assertIn("cgm_glucose", df["modality"].values)
        self.assertEqual(df["patient_id"].iloc[0], "559")


if __name__ == "__main__":
    unittest.main()
