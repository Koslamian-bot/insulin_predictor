"""
Dataset-Agnostic Causal Feature Engineering Engine.

Extracts model-ready features from synchronized multi-modal diabetes time series.
Guarantees strictly backward-looking (causal) calculations to prevent data leakage.
"""

from typing import Dict, Any, List, Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Computes causal rolling, rate-of-change, pharmacokinetic IOB, and circadian features.
    """

    def __init__(
        self,
        grid_step_minutes: int = 5,
        dia_hours: float = 3.5,
    ):
        """
        Args:
            grid_step_minutes: Time step of the grid in minutes.
            dia_hours: Duration of Insulin Action (DIA) in hours for IOB calculation (default 3.5 hrs).
        """
        self.step_mins = grid_step_minutes
        self.dia_mins = dia_hours * 60.0

    def compute_features(self, df_synced: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all derived features on the synchronized DataFrame.
        """
        if df_synced.empty:
            return df_synced.copy()

        df = df_synced.copy()

        # Ensure sorted by timestamp
        if "timestamp" in df.columns:
            df.sort_values("timestamp", inplace=True)

        # -------------------------------------------------------------
        # 1. Glucose Dynamics & Rolling Statistics (Causal)
        # -------------------------------------------------------------
        # Rates of change (backward difference)
        df["glucose_roc_5min"] = df["cgm_glucose"].diff(1)
        df["glucose_roc_15min"] = df["cgm_glucose"].diff(3)
        df["glucose_accel"] = df["glucose_roc_5min"].diff(1)

        # Rolling statistics (closed='right' to ensure backward-only window)
        # 30 min (6 steps), 60 min (12 steps), 120 min (24 steps)
        for window_mins, steps in [(30, 6), (60, 12), (120, 24)]:
            df[f"glucose_mean_{window_mins}m"] = df["cgm_glucose"].rolling(window=steps, min_periods=1).mean()
            df[f"glucose_std_{window_mins}m"] = df["cgm_glucose"].rolling(window=steps, min_periods=2).std().fillna(0.0)
            df[f"glucose_min_{window_mins}m"] = df["cgm_glucose"].rolling(window=steps, min_periods=1).min()
            df[f"glucose_max_{window_mins}m"] = df["cgm_glucose"].rolling(window=steps, min_periods=1).max()

        df["glucose_range_60m"] = df["glucose_max_60m"] - df["glucose_min_60m"]
        df["glucose_ewma_30m"] = df["cgm_glucose"].ewm(span=6, adjust=False).mean()

        # Clinical Glycemic Range Indicators
        df["is_hypo"] = (df["cgm_glucose"] < 70.0).astype(int)
        df["is_target_range"] = ((df["cgm_glucose"] >= 70.0) & (df["cgm_glucose"] <= 180.0)).astype(int)
        df["is_hyper"] = (df["cgm_glucose"] > 180.0).astype(int)

        # -------------------------------------------------------------
        # 2. Meal & Carbohydrate Dynamics (Causal Sums & Elapsed Time)
        # -------------------------------------------------------------
        for window_mins, steps in [(30, 6), (60, 12), (120, 24), (240, 48)]:
            df[f"carbs_past_{window_mins}m"] = df["meal_carbs"].rolling(window=steps, min_periods=1).sum()

        # Time elapsed since last recorded meal (minutes, capped at 720m / 12 hrs)
        time_since_meal = []
        last_meal_time = None
        for ts, carbs in zip(df["timestamp"], df["meal_carbs"]):
            if carbs > 0:
                last_meal_time = ts
            if last_meal_time is None:
                time_since_meal.append(720.0)
            else:
                elapsed = (ts - last_meal_time).total_seconds() / 60.0
                time_since_meal.append(min(elapsed, 720.0))
        df["time_since_last_meal_mins"] = time_since_meal

        # -------------------------------------------------------------
        # 3. Insulin Dynamics & Insulin-On-Board (IOB)
        # -------------------------------------------------------------
        for window_mins, steps in [(30, 6), (60, 12), (120, 24), (240, 48)]:
            df[f"bolus_past_{window_mins}m"] = df["bolus_dose"].rolling(window=steps, min_periods=1).sum()

        # Time elapsed since last bolus (minutes, capped at 720m)
        time_since_bolus = []
        last_bolus_time = None
        for ts, bolus in zip(df["timestamp"], df["bolus_dose"]):
            if bolus > 0:
                last_bolus_time = ts
            if last_bolus_time is None:
                time_since_bolus.append(720.0)
            else:
                elapsed = (ts - last_bolus_time).total_seconds() / 60.0
                time_since_bolus.append(min(elapsed, 720.0))
        df["time_since_last_bolus_mins"] = time_since_bolus

        # Pharmacokinetic IOB Calculation (Standard Bilinear/Linear Decay)
        # For bolus at t_i with dose D_i, remaining fraction at t is max(0, 1 - (t - t_i)/DIA)
        iob_series = np.zeros(len(df))
        dia_steps = int(np.ceil(self.dia_mins / self.step_mins))
        bolus_values = df["bolus_dose"].values

        for idx in range(len(df)):
            if bolus_values[idx] > 0:
                dose = bolus_values[idx]
                for k in range(dia_steps):
                    target_idx = idx + k
                    if target_idx < len(df):
                        elapsed_mins = k * self.step_mins
                        decay_fraction = max(0.0, 1.0 - (elapsed_mins / self.dia_mins))
                        iob_series[target_idx] += dose * decay_fraction

        df["insulin_on_board"] = iob_series

        # -------------------------------------------------------------
        # 4. Circadian & Temporal Cyclics
        # -------------------------------------------------------------
        if "timestamp" in df.columns:
            ts_series = pd.to_datetime(df["timestamp"])
            hours = ts_series.dt.hour + ts_series.dt.minute / 60.0
            weekdays = ts_series.dt.weekday

            df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
            df["day_of_week_sin"] = np.sin(2 * np.pi * weekdays / 7.0)
            df["day_of_week_cos"] = np.cos(2 * np.pi * weekdays / 7.0)
            df["is_weekend"] = (weekdays >= 5).astype(int)

        # -------------------------------------------------------------
        # 5. Wearables Rolling Statistics (Conditional on Modality Presence)
        # -------------------------------------------------------------
        if "heart_rate" in df.columns:
            df["hr_mean_30m"] = df["heart_rate"].rolling(window=6, min_periods=1).mean().fillna(0.0)
            df["hr_std_30m"] = df["heart_rate"].rolling(window=6, min_periods=2).std().fillna(0.0)

        if "gsr" in df.columns:
            df["gsr_mean_30m"] = df["gsr"].rolling(window=6, min_periods=1).mean().fillna(0.0)
            df["gsr_roc_15m"] = df["gsr"].diff(3).fillna(0.0)

        if "skin_temperature" in df.columns:
            df["skin_temp_mean_30m"] = df["skin_temperature"].rolling(window=6, min_periods=1).mean().fillna(0.0)

        if "steps" in df.columns:
            df["steps_past_60m"] = df["steps"].rolling(window=12, min_periods=1).sum().fillna(0.0)

        if "acceleration" in df.columns:
            df["accel_mean_30m"] = df["acceleration"].rolling(window=6, min_periods=1).mean().fillna(0.0)

        return df

    @staticmethod
    def get_feature_manifest() -> Dict[str, List[str]]:
        """Return categorized manifest of all raw and derived features."""
        return {
            "metadata_fields": ["patient_id", "dataset_source", "cohort", "split_label", "timestamp"],
            "raw_measurements": [
                "cgm_glucose", "fingerstick_glucose", "bolus_dose", "meal_carbs",
                "effective_basal_rate", "basal_step_units", "total_insulin_step",
                "heart_rate", "skin_temperature", "air_temperature", "gsr", "steps", "acceleration",
                "sleep_flag", "exercise_flag", "work_flag", "hypo_event_flag",
            ],
            "missingness_indicators": [
                "cgm_missing_mask", "cgm_interpolated", "has_heart_rate",
                "has_skin_temperature", "has_air_temperature", "has_gsr", "has_acceleration",
            ],
            "derived_glucose_features": [
                "glucose_roc_5min", "glucose_roc_15min", "glucose_accel",
                "glucose_mean_30m", "glucose_std_30m", "glucose_min_30m", "glucose_max_30m",
                "glucose_mean_60m", "glucose_std_60m", "glucose_min_60m", "glucose_max_60m",
                "glucose_mean_120m", "glucose_std_120m", "glucose_range_60m", "glucose_ewma_30m",
                "is_hypo", "is_target_range", "is_hyper",
            ],
            "derived_nutrition_features": [
                "carbs_past_30m", "carbs_past_60m", "carbs_past_120m", "carbs_past_240m",
                "time_since_last_meal_mins",
            ],
            "derived_insulin_features": [
                "bolus_past_30m", "bolus_past_60m", "bolus_past_120m", "bolus_past_240m",
                "time_since_last_bolus_mins", "insulin_on_board",
            ],
            "derived_temporal_features": [
                "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "is_weekend",
            ],
            "derived_wearable_features": [
                "hr_mean_30m", "hr_std_30m", "gsr_mean_30m", "gsr_roc_15m",
                "skin_temp_mean_30m", "steps_past_60m", "accel_mean_30m",
            ],
        }
