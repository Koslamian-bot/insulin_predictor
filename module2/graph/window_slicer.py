"""
Time-Series Sliding Window Generator.

Slices continuous 5-minute canonical patient time-series data into bounded
temporal windows [t - W, t] while strictly enforcing patient and split boundaries.
"""

from dataclasses import dataclass
from typing import Generator, List, Optional, Dict, Any
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TimeWindowSlice:
    """
    Data slice representing a single temporal window [t - W, t].
    """
    window_id: str
    patient_id: str
    start_time: pd.Timestamp
    prediction_time: pd.Timestamp
    data: pd.DataFrame
    target_dose: float
    split_label: str


class WindowSlicer:
    """
    Slices canonical DataFrames into patient-specific time-window slices.
    """

    def __init__(
        self,
        window_minutes: int = 120,
        step_minutes: int = 5,
        stride_minutes: int = 5,
        max_cgm_missing_steps: int = 6,  # 6 steps * 5m = 30 min max gap in window
        include_all_bolus: bool = False,
    ):
        self.window_steps = window_minutes // step_minutes
        self.stride_steps = max(1, stride_minutes // step_minutes)
        self.step_minutes = step_minutes
        self.window_minutes = window_minutes
        self.max_cgm_missing_steps = max_cgm_missing_steps
        self.include_all_bolus = include_all_bolus

    def slice_dataframe(
        self, df: pd.DataFrame, drop_invalid_cgm: bool = True
    ) -> Generator[TimeWindowSlice, None, None]:
        """
        Generate TimeWindowSlice objects from a multi-patient canonical DataFrame.
        Enforces strict patient boundaries.
        """
        if df.empty:
            return

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        patient_groups = df.groupby("patient_id", sort=False)

        for patient_id, df_p in patient_groups:
            # Sort chronologically
            df_p = df_p.sort_values("timestamp").reset_index(drop=True)
            n_rows = len(df_p)

            if n_rows < self.window_steps:
                logger.warning(
                    "Patient %s has only %d steps (less than window size %d). Skipping.",
                    patient_id,
                    n_rows,
                    self.window_steps,
                )
                continue

            # Slide window across patient timeline
            for end_idx in range(self.window_steps - 1, n_rows):
                is_stride_point = ((end_idx - (self.window_steps - 1)) % self.stride_steps == 0)
                target_dose = float(df_p.iloc[end_idx].get("target_current_bolus_dose", 0.0))
                is_active_bolus = target_dose > 0.0

                # Sample if at stride point OR if this is an active bolus event
                if not is_stride_point and not (is_active_bolus and self.include_all_bolus):
                    continue

                start_idx = end_idx - self.window_steps + 1
                window_df = df_p.iloc[start_idx : end_idx + 1].copy()

                # Check CGM missingness criteria
                if drop_invalid_cgm and "cgm_missing_mask" in window_df.columns:
                    mask_arr = window_df["cgm_missing_mask"].values
                    consec_missing = 0
                    cur_c = 0
                    for v in mask_arr:
                        if v == 1:
                            cur_c += 1
                            if cur_c > consec_missing:
                                consec_missing = cur_c
                        else:
                            cur_c = 0
                    if consec_missing > self.max_cgm_missing_steps:
                        continue  # Skip window with severe sensor dropout

                start_ts = window_df.iloc[0]["timestamp"]
                pred_ts = window_df.iloc[-1]["timestamp"]
                split_lbl = str(window_df.iloc[-1].get("split_label", "unknown"))

                # Extract prediction target at time t (last row)
                target_dose = float(window_df.iloc[-1].get("target_current_bolus_dose", 0.0))

                start_str = start_ts.strftime("%Y%m%d_%H%M")
                pred_str = pred_ts.strftime("%Y%m%d_%H%M")
                window_id = f"p{patient_id}_{start_str}_{pred_str}"

                yield TimeWindowSlice(
                    window_id=window_id,
                    patient_id=str(patient_id),
                    start_time=start_ts,
                    prediction_time=pred_ts,
                    data=window_df,
                    target_dose=target_dose,
                    split_label=split_lbl,
                )
