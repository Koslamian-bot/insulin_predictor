"""
Module 1 Preprocessing Pipeline Orchestrator.

Implements the end-to-end, dataset-agnostic preprocessing workflow:
Adapter -> Canonical Data -> Validation -> Cleaning -> Unit Normalization
-> Synchronization -> Missing Data -> Feature Engineering -> Target Generation
-> Leakage-Safe Splitting/Scaling -> Quality Reporting -> Processed Dataset.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from module1.adapters.registry import AdapterRegistry
from module1.schema.canonical_schema import CanonicalPatientRecord
from module1.validation.validator import DataValidator
from module1.preprocessing.cleaner import DataCleaner
from module1.preprocessing.unit_normalizer import UnitNormalizer
from module1.preprocessing.timestamp_sync import TimestampSynchronizer
from module1.preprocessing.missing_data import MissingDataHandler
from module1.preprocessing.feature_engineering import FeatureEngineer
from module1.preprocessing.target_builder import TargetBuilder
from module1.preprocessing.splitter_scaler import DatasetSplitter, LeakageSafeScaler
from module1.reports.quality_report import QualityReporter
from module1.config import load_config

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    End-to-end Preprocessing Pipeline Orchestrator.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = config or load_config()

        # Instantiate pipeline components from config
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.normalizer = UnitNormalizer(
            target_temperature_unit=self.cfg.get("units", {}).get("target_temperature_unit", "degF")
        )
        self.synchronizer = TimestampSynchronizer(
            grid_step_minutes=self.cfg.get("synchronization", {}).get("grid_step_minutes", 5),
            tolerance_minutes=self.cfg.get("synchronization", {}).get("tolerance_minutes", 2.5),
        )
        self.missing_handler = MissingDataHandler(
            max_cgm_interp_steps=self.cfg.get("missing_data", {}).get("max_cgm_interp_steps", 3),
            max_sensor_interp_steps=self.cfg.get("missing_data", {}).get("max_sensor_interp_steps", 3),
        )
        self.feature_engineer = FeatureEngineer(
            grid_step_minutes=self.cfg.get("synchronization", {}).get("grid_step_minutes", 5),
            dia_hours=self.cfg.get("pharmacokinetics", {}).get("dia_hours", 3.5),
        )
        self.target_builder = TargetBuilder(
            grid_step_minutes=self.cfg.get("synchronization", {}).get("grid_step_minutes", 5),
            forward_horizons_mins=self.cfg.get("targets", {}).get("forward_horizons_mins", [30, 60]),
        )
        self.scaler = LeakageSafeScaler(
            method=self.cfg.get("splitting", {}).get("scaling_method", "standard")
        )

    def run(
        self,
        dataset_name: str,
        raw_data_dir: str,
        output_base_dir: str = "data",
        save_intermediate_canonical: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full pipeline on the specified dataset.
        
        Args:
            dataset_name: Registered adapter name (e.g. 'ohiot1dm', 'hupa_ucm').
            raw_data_dir: Directory holding raw files.
            output_base_dir: Base directory for canonical, processed, reports, and scalers.
            save_intermediate_canonical: Whether to persist canonical JSON records.
            
        Returns:
            Dictionary with processing metadata, record counts, and output file paths.
        """
        print(f"\n==================================================")
        print(f"[START] INITIATING MODULE 1 PIPELINE: {dataset_name.upper()}")
        print(f"==================================================")

        # 1. Output directory structure setup
        canonical_dir = os.path.join(output_base_dir, "canonical")
        processed_dir = os.path.join(output_base_dir, "processed")
        reports_dir = os.path.join(output_base_dir, "reports")
        scalers_dir = os.path.join(output_base_dir, "scalers")

        for d in [canonical_dir, processed_dir, reports_dir, scalers_dir]:
            os.makedirs(d, exist_ok=True)

        # 2. Get Adapter and Discover Files
        adapter = AdapterRegistry.get_adapter(dataset_name)
        file_paths = adapter.discover_files(raw_data_dir)
        print(f"[OK] Discovered {len(file_paths)} raw patient files via {adapter.__class__.__name__}")

        quality_reporter = QualityReporter(dataset_name=dataset_name)
        processed_dataframes: List[pd.DataFrame] = []
        all_audit_trails = []

        # 3. Process Each Patient File Through Generic Engine
        for idx, file_path in enumerate(file_paths, 1):
            file_name = os.path.basename(file_path)
            print(f"[{idx:02d}/{len(file_paths):02d}] Ingesting & Preprocessing: {file_name} ...", end=" ", flush=True)

            # Step A: Adapter -> Canonical Record
            raw_record = adapter.parse_patient(file_path)

            # Step B: Save Canonical Intermediate if enabled
            if save_intermediate_canonical:
                pid_clean = f"{raw_record.metadata.patient_id}_{raw_record.metadata.split_label}_{raw_record.metadata.cohort}"
                can_file = os.path.join(canonical_dir, f"patient_{pid_clean}.json")
                raw_record.save_json(can_file)

            # Step C: Validation
            val_report = self.validator.validate(raw_record)

            # Step D: Cleaning & Audit Trail
            cleaned_record, audit = self.cleaner.clean(raw_record)
            all_audit_trails.append(audit.to_dict())

            # Step E: Unit Normalization
            norm_record, unit_log = self.normalizer.normalize(cleaned_record)

            # Step F: 5-Minute Grid Timestamp Synchronization
            df_synced = self.synchronizer.synchronize(norm_record)

            # Step G: Missing Data Handling & Masking
            df_missing = self.missing_handler.process(df_synced)

            # Step H: Feature Engineering
            df_features = self.feature_engineer.compute_features(df_missing)

            # Step I: Target Generation
            df_processed_patient = self.target_builder.build_targets(df_features)

            # Ingest into Quality Reporter
            quality_reporter.add_patient_summary(
                raw_record=raw_record,
                cleaned_record=cleaned_record,
                df_processed=df_processed_patient,
                validation_report=val_report,
                audit_trail=audit,
            )

            processed_dataframes.append(df_processed_patient)
            print(f"Done ({len(raw_record.events):,} raw events -> {len(df_processed_patient):,} grid steps)")

        # 4. Aggregate Processed Data
        df_all = pd.concat(processed_dataframes, ignore_index=True)
        print(f"\n[OK] Combined Multi-Patient Processed Grid: {df_all.shape[0]:,} rows x {df_all.shape[1]} columns")

        # 5. Leakage-Safe Train / Validation / Test Splitting and Scaling
        # Separate official train vs test splits
        df_train_raw = df_all[df_all["split_label"] == "train"].copy()
        df_test_raw = df_all[df_all["split_label"] == "test"].copy()

        val_frac = self.cfg.get("splitting", {}).get("val_fraction", 0.15)

        train_dfs = []
        val_dfs = []

        # Chronologically carve validation set from the tail of each training patient
        for pid in df_train_raw["patient_id"].unique():
            p_train = df_train_raw[df_train_raw["patient_id"] == pid]
            tr, val, _ = DatasetSplitter.chronological_split(p_train, val_fraction=val_frac)
            train_dfs.append(tr)
            val_dfs.append(val)

        df_train_split = pd.concat(train_dfs, ignore_index=True)
        df_val_split = pd.concat(val_dfs, ignore_index=True)

        print(f"[OK] Split Allocations: Train={len(df_train_split):,} | Val={len(df_val_split):,} | Test={len(df_test_raw):,}")

        # Feature columns to scale (continuous numerical features, excluding targets/masks/metadata)
        manifest = FeatureEngineer.get_feature_manifest()
        cols_to_scale = [
            col for col in (manifest["raw_measurements"] + manifest["derived_glucose_features"] + manifest["derived_nutrition_features"] + manifest["derived_insulin_features"])
            if col in df_train_split.columns and not col.startswith("target_") and not col.endswith("_flag") and not col.startswith("is_")
        ]

        # Fit scaler exclusively on Train, transform Val and Test
        self.scaler.fit(df_train_split, cols_to_scale)
        df_train_scaled = self.scaler.transform(df_train_split)
        df_val_scaled = self.scaler.transform(df_val_split)
        df_test_scaled = self.scaler.transform(df_test_raw) if not df_test_raw.empty else pd.DataFrame()

        # Save Scaler parameters
        scaler_param_path = os.path.join(scalers_dir, "scaler_params.json")
        self.scaler.export_params_json(scaler_param_path)

        # 6. Save Processed Outputs
        full_csv_path = os.path.join(processed_dir, "canonical_timeseries_all.csv")
        train_csv_path = os.path.join(processed_dir, "canonical_train.csv")
        val_csv_path = os.path.join(processed_dir, "canonical_val.csv")
        test_csv_path = os.path.join(processed_dir, "canonical_test.csv")

        df_all.to_csv(full_csv_path, index=False)
        df_train_scaled.to_csv(train_csv_path, index=False)
        df_val_scaled.to_csv(val_csv_path, index=False)
        if not df_test_scaled.empty:
            df_test_scaled.to_csv(test_csv_path, index=False)

        # Save parquet versions for fast downstream loading if pyarrow is available
        try:
            df_all.to_parquet(os.path.join(processed_dir, "canonical_timeseries_all.parquet"), index=False)
            df_train_scaled.to_parquet(os.path.join(processed_dir, "canonical_train.parquet"), index=False)
            df_val_scaled.to_parquet(os.path.join(processed_dir, "canonical_val.parquet"), index=False)
            if not df_test_scaled.empty:
                df_test_scaled.to_parquet(os.path.join(processed_dir, "canonical_test.parquet"), index=False)
        except Exception:
            pass  # CSV is primary standard

        # 7. Generate Quality & Modality Reports
        report_paths = quality_reporter.save_reports(reports_dir)

        # Save audit trails
        with open(os.path.join(reports_dir, "cleaning_audit_trail.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(all_audit_trails, f, indent=2)

        print(f"\n==================================================")
        print(f"[COMPLETE] MODULE 1 PREPROCESSING FINISHED SUCCESSFULLY")
        print(f"==================================================")
        print(f"Quality Report: {report_paths['full_report']}")
        print(f"Modality Matrix: {report_paths['modality_matrix']}")
        print(f"Processed Datasets: {processed_dir}")
        print(f"==================================================\n")

        return {
            "dataset_name": dataset_name,
            "patient_files_count": len(file_paths),
            "total_grid_rows": len(df_all),
            "total_features": df_all.shape[1],
            "train_rows": len(df_train_scaled),
            "val_rows": len(df_val_scaled),
            "test_rows": len(df_test_scaled),
            "processed_dir": processed_dir,
            "reports_dir": reports_dir,
        }


def preprocess(
    dataset: str = "ohiot1dm",
    input_path: str = r"c:\Final year\Insulin prediction\OhioT1DM",
    output_path: str = r"c:\Final year\Insulin prediction\data",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Top-level convenience entry point.
    
    Example usage:
        preprocess(dataset="ohiot1dm", input_path="OhioT1DM", output_path="data")
    """
    cfg = load_config(config_path) if config_path else None
    pipeline = PreprocessingPipeline(config=cfg)
    return pipeline.run(
        dataset_name=dataset,
        raw_data_dir=input_path,
        output_base_dir=output_path,
    )


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "ohiot1dm"
    input_dir = sys.argv[2] if len(sys.argv) > 2 else r"c:\Final year\Insulin prediction\OhioT1DM"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else r"c:\Final year\Insulin prediction\data"
    preprocess(dataset=dataset, input_path=input_dir, output_path=out_dir)
