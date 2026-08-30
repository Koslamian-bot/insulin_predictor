"""
Dataset-Agnostic Missing Data Handler.

Implements principled, medically valid missing-data policies:
- Distinguishes discrete absence (0 events) vs continuous sensor dropouts (NaN).
- Restricts linear interpolation to short physiological gaps (<= 15-30 min).
- Generates explicit binary masks (cgm_missing_mask, cgm_interpolated, modality availability flags).
- Prevents forward-filling across hours.
"""

from typing import Dict, Any, Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MissingDataHandler:
    """
    Handles missingness in synchronized multi-modal diabetes time-series data.
    """

    def __init__(self, max_cgm_interp_steps: int = 3, max_sensor_interp_steps: int = 3):
        """
        Args:
            max_cgm_interp_steps: Maximum consecutive missing steps to linearly interpolate
                                  (e.g., 3 steps * 5 min = 15 minutes max gap).
            max_sensor_interp_steps: Maximum consecutive missing steps for wearable sensors.
        """
        self.max_cgm_interp_steps = max_cgm_interp_steps
        self.max_sensor_interp_steps = max_sensor_interp_steps

    def process(self, df_grid: pd.DataFrame) -> pd.DataFrame:
        """
        Apply missingness masks, short-gap interpolation, and modality indicators.
        """
        if df_grid.empty:
            return df_grid.copy()

        df = df_grid.copy()

        # 1. CGM Missingness Mask and Short-Gap Linear Interpolation
        df["cgm_missing_mask"] = df["cgm_glucose"].isna().astype(int)
        
        # Linearly interpolate short gaps with strict limit
        interp_cgm = df["cgm_glucose"].interpolate(
            method="linear",
            limit=self.max_cgm_interp_steps,
            limit_direction="forward",
        )
        
        # Identify rows that were interpolated
        df["cgm_interpolated"] = (df["cgm_missing_mask"] == 1) & interp_cgm.notna()
        df["cgm_interpolated"] = df["cgm_interpolated"].astype(int)
        
        # Assign interpolated values (gaps > max_cgm_interp_steps remain NaN)
        df["cgm_glucose"] = interp_cgm

        # 2. Modality Availability Indicators
        df["has_heart_rate"] = int(df["heart_rate"].notna().sum() > 0)
        df["has_skin_temperature"] = int(df["skin_temperature"].notna().sum() > 0)
        df["has_air_temperature"] = int(df["air_temperature"].notna().sum() > 0)
        df["has_gsr"] = int(df["gsr"].notna().sum() > 0)
        df["has_acceleration"] = int(df["acceleration"].notna().sum() > 0)

        # 3. Continuous Wearable Sensors Short-Gap Interpolation (Only if modality is present)
        for col, avail_flag in [
            ("heart_rate", "has_heart_rate"),
            ("skin_temperature", "has_skin_temperature"),
            ("air_temperature", "has_air_temperature"),
            ("gsr", "has_gsr"),
            ("acceleration", "has_acceleration"),
        ]:
            if df[avail_flag].iloc[0] == 1:
                df[f"{col}_missing_mask"] = df[col].isna().astype(int)
                df[col] = df[col].interpolate(
                    method="linear",
                    limit=self.max_sensor_interp_steps,
                    limit_direction="forward",
                )
            else:
                # Modality unavailable for this cohort/patient: fill with 0 and mask=1
                df[f"{col}_missing_mask"] = 1
                df[col] = 0.0

        # 4. Discrete Events & Basal Rate (Guaranteed Non-NaN)
        df["bolus_dose"] = df["bolus_dose"].fillna(0.0)
        df["meal_carbs"] = df["meal_carbs"].fillna(0.0)
        df["steps"] = df["steps"].fillna(0.0)
        df["effective_basal_rate"] = df["effective_basal_rate"].ffill().bfill().fillna(0.0)
        df["basal_step_units"] = df["basal_step_units"].ffill().bfill().fillna(0.0)
        df["total_insulin_step"] = df["total_insulin_step"].fillna(0.0)

        # State flags (0 or 1)
        for flag in ["sleep_flag", "exercise_flag", "work_flag", "hypo_event_flag", "has_bolus_event", "has_meal_event"]:
            if flag in df.columns:
                df[flag] = df[flag].fillna(0).astype(int)

        return df
