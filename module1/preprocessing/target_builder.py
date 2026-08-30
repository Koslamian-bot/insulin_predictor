"""
Dataset-Agnostic Insulin Dosage Target Builder.

Generates ground-truth prediction targets derived strictly from genuine
clinical insulin administration records. Supports event-based dosage regression,
forward-horizon total bolus prediction, and next-step net insulin targets.
"""

from typing import List, Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TargetBuilder:
    """
    Constructs ground-truth targets for downstream insulin prediction models.
    """

    def __init__(self, grid_step_minutes: int = 5, forward_horizons_mins: List[int] = None):
        self.step_mins = grid_step_minutes
        self.horizons = forward_horizons_mins or [30, 60]

    def build_targets(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Add ground-truth prediction target columns to the feature DataFrame.
        """
        if df_features.empty:
            return df_features.copy()

        df = df_features.copy()

        # -------------------------------------------------------------
        # Target 1: Event-Level Bolus Dosage Target (Continuous Float)
        # Value is the exact bolus dose (Units) delivered at this time step
        # -------------------------------------------------------------
        df["target_current_bolus_dose"] = df["bolus_dose"].copy()

        # Target 1b: Bolus Delivery Event Binary Flag (Classification)
        df["target_has_bolus"] = (df["bolus_dose"] > 0).astype(int)

        # -------------------------------------------------------------
        # Target 2: Next Step Total Insulin Requirement (t+1)
        # Total insulin = bolus + basal step units
        # -------------------------------------------------------------
        df["target_next_step_total_insulin"] = df["total_insulin_step"].shift(-1)

        # -------------------------------------------------------------
        # Target 3: Forward Horizon Bolus Insulin Delivery (Units in [t, t+H])
        # -------------------------------------------------------------
        for h_mins in self.horizons:
            steps = int(h_mins / self.step_mins)
            # Forward rolling sum: shift backward so window covers (t, t + steps]
            # using reverse rolling sum
            future_bolus = df["bolus_dose"].iloc[::-1].rolling(window=steps, min_periods=1).sum().iloc[::-1]
            df[f"target_future_bolus_{h_mins}m"] = future_bolus

        return df
