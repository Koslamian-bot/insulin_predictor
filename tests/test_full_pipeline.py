"""
End-to-end integration test for Module 1 Preprocessing Pipeline.
"""

import os
import unittest
import pandas as pd
from module1.pipeline.preprocessing_pipeline import PreprocessingPipeline, preprocess


class TestFullPipeline(unittest.TestCase):

    def setUp(self):
        self.raw_data_dir = r"c:\Final year\Insulin prediction\OhioT1DM"
        self.output_dir = r"c:\Final year\Insulin prediction\data"

    def test_end_to_end_pipeline_execution(self):
        result = preprocess(
            dataset="ohiot1dm",
            input_path=self.raw_data_dir,
            output_path=self.output_dir,
        )

        # 1. Verify file counts and rows
        self.assertEqual(result["patient_files_count"], 24)
        self.assertGreater(result["total_grid_rows"], 150000)
        self.assertGreater(result["train_rows"], 100000)
        self.assertGreater(result["val_rows"], 15000)
        self.assertGreater(result["test_rows"], 30000)

        # 2. Verify Output Files Exist
        proc_dir = os.path.join(self.output_dir, "processed")
        reports_dir = os.path.join(self.output_dir, "reports")
        can_dir = os.path.join(self.output_dir, "canonical")
        scalers_dir = os.path.join(self.output_dir, "scalers")

        self.assertTrue(os.path.exists(os.path.join(proc_dir, "canonical_timeseries_all.csv")))
        self.assertTrue(os.path.exists(os.path.join(proc_dir, "canonical_train.csv")))
        self.assertTrue(os.path.exists(os.path.join(proc_dir, "canonical_val.csv")))
        self.assertTrue(os.path.exists(os.path.join(proc_dir, "canonical_test.csv")))
        self.assertTrue(os.path.exists(os.path.join(reports_dir, "data_quality_report.md")))
        self.assertTrue(os.path.exists(os.path.join(reports_dir, "modality_availability.md")))
        self.assertTrue(os.path.exists(os.path.join(reports_dir, "quality_report.json")))
        self.assertTrue(os.path.exists(os.path.join(scalers_dir, "scaler_params.json")))

        # Check canonical records count
        can_files = [f for f in os.listdir(can_dir) if f.endswith(".json")]
        self.assertEqual(len(can_files), 24)

        # 3. Inspect Processed DataFrame Columns and Values
        df_train = pd.read_csv(os.path.join(proc_dir, "canonical_train.csv"))
        
        # Verify required columns exist
        expected_cols = [
            "timestamp", "patient_id", "cgm_glucose", "bolus_dose", "meal_carbs",
            "effective_basal_rate", "total_insulin_step", "insulin_on_board",
            "glucose_roc_5min", "glucose_mean_30m", "carbs_past_60m", "bolus_past_60m",
            "hour_sin", "hour_cos", "target_current_bolus_dose", "target_future_bolus_30m",
            "cgm_missing_mask", "cgm_interpolated",
        ]
        for col in expected_cols:
            self.assertIn(col, df_train.columns, f"Missing expected column: {col}")

        # Check targets are genuine non-synthetic
        bolus_targets = df_train[df_train["target_current_bolus_dose"] > 0]["target_current_bolus_dose"]
        self.assertGreater(len(bolus_targets), 1000)
        self.assertLessEqual(bolus_targets.max(), 50.0)


if __name__ == "__main__":
    unittest.main()
