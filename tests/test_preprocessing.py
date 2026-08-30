"""
Unit and integration tests for all preprocessing engine components.
"""

from datetime import datetime, timedelta
import unittest
import pandas as pd
import numpy as np

from module1.schema.canonical_schema import (
    CanonicalPatientRecord,
    PatientMetadata,
    CanonicalEvent,
    ModalityType,
)
from module1.preprocessing.cleaner import DataCleaner
from module1.preprocessing.unit_normalizer import UnitNormalizer
from module1.preprocessing.timestamp_sync import TimestampSynchronizer
from module1.preprocessing.missing_data import MissingDataHandler
from module1.preprocessing.feature_engineering import FeatureEngineer
from module1.preprocessing.target_builder import TargetBuilder
from module1.preprocessing.splitter_scaler import DatasetSplitter, LeakageSafeScaler


class TestPreprocessingEngine(unittest.TestCase):

    def setUp(self):
        self.meta = PatientMetadata(patient_id="559", dataset_source="OhioT1DM", cohort="2018", split_label="train")
        self.record = CanonicalPatientRecord(metadata=self.meta)
        
        # Build synthetic 1-day timeline of multi-modal events
        base_time = datetime(2026, 1, 1, 8, 0, 0)
        # CGM every 5 min
        for step in range(288):
            t = base_time + timedelta(minutes=5 * step)
            val = 120.0 + 30.0 * np.sin(step / 10.0)
            self.record.add_event(CanonicalEvent(t, ModalityType.CGM_GLUCOSE, float(val)))
            self.record.add_event(CanonicalEvent(t, ModalityType.HEART_RATE, 72.0))
            self.record.add_event(CanonicalEvent(t, ModalityType.SKIN_TEMPERATURE, 92.0))

        # Add discrete basal
        self.record.add_event(CanonicalEvent(base_time, ModalityType.BASAL_RATE, 0.8))

        # Add discrete meal and bolus at 08:30 (step 6)
        meal_t = base_time + timedelta(minutes=30)
        self.record.add_event(CanonicalEvent(meal_t, ModalityType.MEAL_CARBS, 50.0, attributes={"type": "Breakfast"}))
        self.record.add_event(CanonicalEvent(meal_t, ModalityType.BOLUS_DOSE, 5.0, attributes={"type": "normal"}))

        # Add duplicate and out of bounds to test cleaner
        self.record.add_event(CanonicalEvent(meal_t, ModalityType.BOLUS_DOSE, 5.0, attributes={"type": "normal"}))
        self.record.add_event(CanonicalEvent(meal_t, ModalityType.CGM_GLUCOSE, -99.0))

    def test_cleaner_and_audit(self):
        cleaner = DataCleaner()
        cleaned_rec, audit = cleaner.clean(self.record)

        self.assertEqual(audit.initial_event_count, len(self.record.events))
        self.assertGreater(audit.removed_event_count, 0)
        self.assertEqual(audit.final_event_count, len(cleaned_rec.events))
        self.assertIn("DEDUPLICATE", [e.action for e in audit.entries])
        self.assertIn("FILTER_OUT_OF_BOUNDS", [e.action for e in audit.entries])

    def test_unit_normalizer(self):
        normalizer = UnitNormalizer(target_temperature_unit="degC")
        norm_rec, log = normalizer.normalize(self.record, source_temperature_unit="degF")

        self.assertEqual(len(norm_rec.events), len(self.record.events))
        self.assertGreater(len(log.entries), 0)
        # Check temperature converted: 92 F -> ~33.33 C
        skin_events = [e for e in norm_rec.events if e.modality == ModalityType.SKIN_TEMPERATURE]
        self.assertAlmostEqual(skin_events[0].value, (92.0 - 32.0) * (5.0 / 9.0), places=2)

    def test_timestamp_sync(self):
        cleaner = DataCleaner()
        cleaned_rec, _ = cleaner.clean(self.record)

        synchronizer = TimestampSynchronizer(grid_step_minutes=5)
        df_grid = synchronizer.synchronize(cleaned_rec)

        self.assertFalse(df_grid.empty)
        self.assertIn("cgm_glucose", df_grid.columns)
        self.assertIn("bolus_dose", df_grid.columns)
        self.assertIn("meal_carbs", df_grid.columns)
        self.assertIn("effective_basal_rate", df_grid.columns)

        # Check bolus dose conserved (5.0 U)
        self.assertAlmostEqual(df_grid["bolus_dose"].sum(), 5.0)
        self.assertAlmostEqual(df_grid["meal_carbs"].sum(), 50.0)

    def test_missing_data_and_features(self):
        cleaner = DataCleaner()
        cleaned_rec, _ = cleaner.clean(self.record)
        synchronizer = TimestampSynchronizer(grid_step_minutes=5)
        df_grid = synchronizer.synchronize(cleaned_rec)

        # Inject a small gap in CGM
        df_grid.loc[10:11, "cgm_glucose"] = np.nan

        missing_handler = MissingDataHandler(max_cgm_interp_steps=3)
        df_imputed = missing_handler.process(df_grid)

        self.assertEqual(df_imputed["cgm_interpolated"].sum(), 2)
        self.assertFalse(df_imputed["cgm_glucose"].isna().any())

        # Feature engineering
        fe = FeatureEngineer(grid_step_minutes=5, dia_hours=3.5)
        df_feat = fe.compute_features(df_imputed)

        self.assertIn("glucose_roc_5min", df_feat.columns)
        self.assertIn("glucose_mean_30m", df_feat.columns)
        self.assertIn("carbs_past_60m", df_feat.columns)
        self.assertIn("bolus_past_60m", df_feat.columns)
        self.assertIn("insulin_on_board", df_feat.columns)
        self.assertIn("hour_sin", df_feat.columns)

        # IOB should peak after bolus and then decrease
        bolus_idx = df_feat[df_feat["bolus_dose"] > 0].index[0]
        iob_at_bolus = df_feat.loc[bolus_idx, "insulin_on_board"]
        iob_1h_later = df_feat.loc[bolus_idx + 12, "insulin_on_board"]
        self.assertGreater(iob_at_bolus, iob_1h_later)
        self.assertGreater(iob_1h_later, 0.0)

    def test_target_builder_and_scaling(self):
        cleaner = DataCleaner()
        cleaned_rec, _ = cleaner.clean(self.record)
        synchronizer = TimestampSynchronizer(grid_step_minutes=5)
        df_grid = synchronizer.synchronize(cleaned_rec)
        missing_handler = MissingDataHandler()
        df_imputed = missing_handler.process(df_grid)
        fe = FeatureEngineer()
        df_feat = fe.compute_features(df_imputed)

        tb = TargetBuilder(grid_step_minutes=5, forward_horizons_mins=[30, 60])
        df_targets = tb.build_targets(df_feat)

        self.assertIn("target_current_bolus_dose", df_targets.columns)
        self.assertIn("target_future_bolus_30m", df_targets.columns)
        self.assertIn("target_next_step_total_insulin", df_targets.columns)

        # Test Splitting and Scaling
        df_train, df_val, _ = DatasetSplitter.chronological_split(df_targets, val_fraction=0.2)
        self.assertGreater(len(df_train), 0)
        self.assertGreater(len(df_val), 0)
        self.assertEqual(len(df_train) + len(df_val), len(df_targets))

        # Check Scaler
        scaler = LeakageSafeScaler(method="standard")
        cols_to_scale = ["cgm_glucose", "glucose_mean_30m", "insulin_on_board"]
        df_train_scaled = scaler.fit_transform(df_train, cols_to_scale)
        df_val_scaled = scaler.transform(df_val)

        # Train mean should be close to 0
        self.assertAlmostEqual(df_train_scaled["cgm_glucose"].mean(), 0.0, places=1)
        self.assertTrue(scaler.is_fitted)


if __name__ == "__main__":
    unittest.main()
