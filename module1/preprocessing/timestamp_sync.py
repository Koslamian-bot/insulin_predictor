"""
Dataset-Agnostic Multi-Modal Timestamp Synchronization Engine (Vectorized & High-Performance).

Aligns asynchronous continuous and discrete multi-modal events to a uniform
5-minute temporal backbone (derived from the CGM timeline) while strictly
conserving discrete quantities (bolus insulin units, carbohydrate grams)
and active basal step rates.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np

from module1.schema.canonical_schema import (
    CanonicalPatientRecord,
    CanonicalEvent,
    ModalityType,
)

logger = logging.getLogger(__name__)


class TimestampSynchronizer:
    """
    Synchronizes heterogeneous event streams into a model-ready regular time-series grid.
    """

    def __init__(self, grid_step_minutes: int = 5, tolerance_minutes: float = 2.5):
        self.grid_step = timedelta(minutes=grid_step_minutes)
        self.tolerance = timedelta(minutes=tolerance_minutes)
        self.grid_step_minutes = grid_step_minutes

    def synchronize(self, record: CanonicalPatientRecord) -> pd.DataFrame:
        """
        Synchronize all events in the patient record into a uniform grid DataFrame.
        """
        if not record.events:
            return pd.DataFrame()

        # 1. Determine timeline bounds from CGM or all events
        cgm_events = [e for e in record.events if e.modality == ModalityType.CGM_GLUCOSE]
        if cgm_events:
            min_ts = min(e.timestamp for e in cgm_events)
            max_ts = max(e.timestamp for e in cgm_events)
        else:
            min_ts = min(e.timestamp for e in record.events)
            max_ts = max(e.timestamp for e in record.events)

        # Round min_ts down and max_ts up to nearest 5 minutes
        start_ts = min_ts.replace(
            minute=(min_ts.minute // self.grid_step_minutes) * self.grid_step_minutes,
            second=0,
            microsecond=0,
        )
        end_ts = max_ts.replace(
            minute=(max_ts.minute // self.grid_step_minutes) * self.grid_step_minutes,
            second=0,
            microsecond=0,
        )
        if end_ts < max_ts:
            end_ts += self.grid_step

        # Build regular grid timestamps
        grid_timestamps = pd.date_range(start=start_ts, end=end_ts, freq=f"{self.grid_step_minutes}min")
        if len(grid_timestamps) == 0:
            return pd.DataFrame()

        df_grid = pd.DataFrame(index=grid_timestamps)
        df_grid["patient_id"] = record.metadata.patient_id
        df_grid["dataset_source"] = record.metadata.dataset_source
        df_grid["cohort"] = record.metadata.cohort or "standard"
        df_grid["split_label"] = record.metadata.split_label or "all"

        # Separate events by modality
        events_by_mod: Dict[ModalityType, List[CanonicalEvent]] = {}
        for e in record.events:
            events_by_mod.setdefault(e.modality, []).append(e)

        # 2. Continuous Modalities (CGM, Finger-stick, HR, Skin Temp, Air Temp, GSR, Accel)
        continuous_mods = [
            (ModalityType.CGM_GLUCOSE, "cgm_glucose"),
            (ModalityType.FINGERSTICK_GLUCOSE, "fingerstick_glucose"),
            (ModalityType.HEART_RATE, "heart_rate"),
            (ModalityType.SKIN_TEMPERATURE, "skin_temperature"),
            (ModalityType.AIR_TEMPERATURE, "air_temperature"),
            (ModalityType.GSR, "gsr"),
            (ModalityType.ACCELERATION, "acceleration"),
        ]

        for mod, col_name in continuous_mods:
            if mod in events_by_mod:
                evs = events_by_mod[mod]
                val_accum: Dict[datetime, List[float]] = {}
                for e in evs:
                    g_ts = self._round_to_grid(e.timestamp)
                    val_accum.setdefault(g_ts, []).append(e.value)
                mean_dict = {k: float(np.mean(v)) for k, v in val_accum.items()}
                s = pd.Series(mean_dict)
                df_grid[col_name] = s.reindex(df_grid.index)
            else:
                df_grid[col_name] = np.nan

        # 3. Discrete Sum Modalities (Bolus Dose, Meal Carbs, Steps)
        # Bolus
        bolus_sums: Dict[datetime, float] = {}
        bolus_types: Dict[datetime, str] = {}
        if ModalityType.BOLUS_DOSE in events_by_mod:
            for e in events_by_mod[ModalityType.BOLUS_DOSE]:
                target = self._ceil_to_grid(e.timestamp)
                bolus_sums[target] = bolus_sums.get(target, 0.0) + e.value
                b_type = e.attributes.get("type", "normal")
                if b_type:
                    bolus_types[target] = b_type

        df_grid["bolus_dose"] = pd.Series(bolus_sums).reindex(df_grid.index).fillna(0.0)
        df_grid["has_bolus_event"] = (df_grid["bolus_dose"] > 0).astype(int)
        df_grid["bolus_type"] = pd.Series(bolus_types).reindex(df_grid.index)

        # Meals
        meal_sums: Dict[datetime, float] = {}
        meal_types: Dict[datetime, str] = {}
        if ModalityType.MEAL_CARBS in events_by_mod:
            for e in events_by_mod[ModalityType.MEAL_CARBS]:
                target = self._ceil_to_grid(e.timestamp)
                meal_sums[target] = meal_sums.get(target, 0.0) + e.value
                m_type = e.attributes.get("type", "")
                if m_type:
                    meal_types[target] = m_type

        df_grid["meal_carbs"] = pd.Series(meal_sums).reindex(df_grid.index).fillna(0.0)
        df_grid["has_meal_event"] = (df_grid["meal_carbs"] > 0).astype(int)
        df_grid["meal_type"] = pd.Series(meal_types).reindex(df_grid.index)

        # Steps
        steps_sums: Dict[datetime, float] = {}
        if ModalityType.STEPS in events_by_mod:
            for e in events_by_mod[ModalityType.STEPS]:
                target = self._ceil_to_grid(e.timestamp)
                steps_sums[target] = steps_sums.get(target, 0.0) + e.value
        df_grid["steps"] = pd.Series(steps_sums).reindex(df_grid.index).fillna(0.0)

        # 4. Basal Rate Timeline (Scheduled Basal + Temp Basal Overrides)
        basal_events = sorted(
            [e for e in record.events if e.modality == ModalityType.BASAL_RATE],
            key=lambda e: e.timestamp,
        )
        if basal_events:
            b_ts = [e.timestamp for e in basal_events]
            b_vals = [e.value for e in basal_events]
            s_basal = pd.Series(b_vals, index=pd.DatetimeIndex(b_ts))
            # Combine index with grid index, forward fill, then reindex to grid
            combined_idx = df_grid.index.union(s_basal.index).sort_values()
            s_prop = s_basal.reindex(combined_idx).ffill().bfill()
            df_grid["basal_rate"] = s_prop.reindex(df_grid.index)
        else:
            df_grid["basal_rate"] = 0.0

        # Temp basal overrides
        df_grid["temp_basal_rate"] = np.nan
        temp_basal_events = events_by_mod.get(ModalityType.TEMP_BASAL_RATE, [])
        for tbe in temp_basal_events:
            t_start = tbe.timestamp
            t_end = tbe.timestamp_end or (t_start + timedelta(hours=1))
            mask = (df_grid.index >= t_start) & (df_grid.index <= t_end)
            df_grid.loc[mask, "temp_basal_rate"] = tbe.value

        df_grid["effective_basal_rate"] = df_grid["temp_basal_rate"].fillna(df_grid["basal_rate"])
        df_grid["basal_step_units"] = df_grid["effective_basal_rate"] * (self.grid_step_minutes / 60.0)
        df_grid["total_insulin_step"] = df_grid["bolus_dose"] + df_grid["basal_step_units"]

        # 5. State Intervals (Sleep, Exercise, Work, Hypo Events)
        df_grid["sleep_flag"] = 0
        df_grid["exercise_flag"] = 0
        df_grid["work_flag"] = 0
        df_grid["hypo_event_flag"] = 0

        for e in events_by_mod.get(ModalityType.SLEEP, []):
            t_start = e.timestamp
            t_end = e.timestamp_end or (t_start + timedelta(hours=1))
            df_grid.loc[(df_grid.index >= t_start) & (df_grid.index <= t_end), "sleep_flag"] = 1

        for e in events_by_mod.get(ModalityType.EXERCISE, []):
            dur = e.value_secondary or 30.0
            t_start = e.timestamp
            t_end = t_start + timedelta(minutes=dur)
            df_grid.loc[(df_grid.index >= t_start) & (df_grid.index <= t_end), "exercise_flag"] = 1

        for e in events_by_mod.get(ModalityType.WORK, []):
            t_start = e.timestamp
            t_end = e.timestamp_end or (t_start + timedelta(hours=1))
            df_grid.loc[(df_grid.index >= t_start) & (df_grid.index <= t_end), "work_flag"] = 1

        for e in events_by_mod.get(ModalityType.HYPO_EVENT, []):
            target_grid = self._round_to_grid(e.timestamp)
            if target_grid in df_grid.index:
                df_grid.loc[target_grid, "hypo_event_flag"] = 1

        df_grid.reset_index(names=["timestamp"], inplace=True)
        return df_grid

    def _round_to_grid(self, dt: datetime) -> datetime:
        """Round datetime to nearest grid step."""
        mins = dt.minute + dt.second / 60.0
        rounded_mins = int(round(mins / self.grid_step_minutes)) * self.grid_step_minutes
        base = dt.replace(minute=0, second=0, microsecond=0)
        return base + timedelta(minutes=rounded_mins)

    def _ceil_to_grid(self, dt: datetime) -> datetime:
        """Round datetime up to enclosing grid step."""
        mins = dt.minute + dt.second / 60.0
        ceil_mins = int(np.ceil(mins / self.grid_step_minutes)) * self.grid_step_minutes
        base = dt.replace(minute=0, second=0, microsecond=0)
        return base + timedelta(minutes=ceil_mins)
