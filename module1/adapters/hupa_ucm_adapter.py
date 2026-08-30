"""
HUPA-UCM Dataset Adapter.

Translates HUPA-UCM tabular format (semicolon/comma-separated CSV files)
into standardized CanonicalPatientRecord structures.
"""

import os
import glob
import logging
import pandas as pd
from typing import List, Optional
from datetime import datetime

from module1.adapters.base_adapter import BaseDatasetAdapter
from module1.adapters.registry import AdapterRegistry
from module1.schema.canonical_schema import (
    ModalityType,
    CanonicalEvent,
    PatientMetadata,
    CanonicalPatientRecord,
)

logger = logging.getLogger(__name__)


@AdapterRegistry.register("hupa_ucm")
@AdapterRegistry.register("hupaucm")
class HUPAUCMAdapter(BaseDatasetAdapter):
    """
    Adapter for the HUPA-UCM diabetes dataset.
    Supports preprocessed CSV files with time, glucose, heart_rate, steps, basal_rate,
    bolus_volume_delivered, carb_input.
    """

    @property
    def dataset_name(self) -> str:
        return "hupa_ucm"

    def discover_files(self, raw_data_dir: str) -> List[str]:
        """Discover CSV files for HUPA-UCM."""
        pattern = os.path.join(raw_data_dir, "**", "*.csv")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(raw_data_dir, "*.csv")))
        logger.info("HUPAUCMAdapter discovered %d CSV files in %s", len(files), raw_data_dir)
        return files

    def parse_patient(self, file_path: str, **kwargs) -> CanonicalPatientRecord:
        """
        Parse HUPA-UCM tabular format into CanonicalPatientRecord.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HUPA-UCM file not found: {file_path}")

        # Detect delimiter (semicolon vs comma)
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            sep = ";" if ";" in first_line else ","

        df = pd.read_csv(file_path, sep=sep)
        pid = kwargs.get("patient_id", os.path.splitext(os.path.basename(file_path))[0])

        metadata = PatientMetadata(
            patient_id=str(pid),
            dataset_source="HUPA-UCM",
            cohort="standard",
            split_label=kwargs.get("split_label", "all"),
            extra_info={"file_path": file_path},
        )

        record = CanonicalPatientRecord(metadata=metadata)

        # Standard column mapping
        ts_col = "time" if "time" in df.columns else ("timestamp" if "timestamp" in df.columns else df.columns[0])
        
        for _, row in df.iterrows():
            ts = self.parse_timestamp(str(row.get(ts_col)))
            if not ts:
                continue

            # CGM Glucose
            if "glucose" in row and pd.notna(row["glucose"]):
                record.add_event(CanonicalEvent(ts, ModalityType.CGM_GLUCOSE, float(row["glucose"])))

            # Bolus insulin (bolus_volume_delivered or bolus)
            bolus_col = "bolus_volume_delivered" if "bolus_volume_delivered" in row else ("bolus" if "bolus" in row else None)
            if bolus_col and pd.notna(row[bolus_col]):
                b_val = float(row[bolus_col])
                if b_val > 0:
                    record.add_event(CanonicalEvent(ts, ModalityType.BOLUS_DOSE, b_val))

            # Basal insulin (basal_rate or basal)
            basal_col = "basal_rate" if "basal_rate" in row else ("basal" if "basal" in row else None)
            if basal_col and pd.notna(row[basal_col]):
                record.add_event(CanonicalEvent(ts, ModalityType.BASAL_RATE, float(row[basal_col])))

            # Carbohydrates (carb_input or carbs)
            carb_col = "carb_input" if "carb_input" in row else ("carbs" if "carbs" in row else None)
            if carb_col and pd.notna(row[carb_col]):
                c_val = float(row[carb_col])
                if c_val > 0:
                    record.add_event(CanonicalEvent(ts, ModalityType.MEAL_CARBS, c_val))

            # Heart Rate
            if "heart_rate" in row and pd.notna(row["heart_rate"]):
                record.add_event(CanonicalEvent(ts, ModalityType.HEART_RATE, float(row["heart_rate"])))

            # Steps
            if "steps" in row and pd.notna(row["steps"]):
                s_val = float(row["steps"])
                if s_val > 0:
                    record.add_event(CanonicalEvent(ts, ModalityType.STEPS, s_val))

        record.sort_events()
        return record
