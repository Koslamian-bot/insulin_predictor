"""
Leakage-Safe Dataset Splitting and Feature Scaling Engine.

Guarantees zero temporal or patient-level data leakage:
- Chronological intra-patient splitting (Train -> Val -> Test)
- Patient-wise holdout splitting
- Scalers fit strictly on Training splits and applied to Validation/Test splits.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Splits time-series datasets cleanly without temporal or patient leakage.
    """

    @staticmethod
    def chronological_split(
        df_patient: pd.DataFrame,
        val_fraction: float = 0.15,
        test_fraction: float = 0.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Split a single patient's time series strictly by time.
        
        Args:
            df_patient: Chronologically sorted patient DataFrame.
            val_fraction: Fraction of records allocated to validation (from the end of train).
            test_fraction: Optional fraction for test (if not using predefined test files).
            
        Returns:
            Tuple of (df_train, df_val, df_test).
        """
        if df_patient.empty:
            return df_patient.copy(), df_patient.copy(), df_patient.copy()

        df = df_patient.sort_values("timestamp").reset_index(drop=True)
        n = len(df)

        if test_fraction > 0.0:
            n_test = int(n * test_fraction)
            n_val = int((n - n_test) * val_fraction)
            n_train = n - n_val - n_test

            df_train = df.iloc[:n_train].copy()
            df_val = df.iloc[n_train:n_train + n_val].copy()
            df_test = df.iloc[n_train + n_val:].copy()
        else:
            n_val = int(n * val_fraction)
            n_train = n - n_val

            df_train = df.iloc[:n_train].copy()
            df_val = df.iloc[n_train:].copy()
            df_test = pd.DataFrame()

        return df_train, df_val, df_test

    @staticmethod
    def patient_wise_split(
        patient_records: List[pd.DataFrame],
        train_pids: List[str],
        val_pids: List[str],
        test_pids: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partition multi-patient DataFrames by Patient ID.
        """
        df_all = pd.concat(patient_records, ignore_index=True) if patient_records else pd.DataFrame()
        if df_all.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df_train = df_all[df_all["patient_id"].astype(str).isin([str(p) for p in train_pids])].copy()
        df_val = df_all[df_all["patient_id"].astype(str).isin([str(p) for p in val_pids])].copy()
        df_test = df_all[df_all["patient_id"].astype(str).isin([str(p) for p in test_pids])].copy()

        return df_train, df_val, df_test


class LeakageSafeScaler:
    """
    Feature scaler that learns normalization parameters exclusively from training data.
    """

    def __init__(self, method: str = "standard"):
        """
        Args:
            method: 'standard' (StandardScaler), 'robust' (RobustScaler), or 'minmax' (MinMaxScaler).
        """
        self.method = method.lower()
        if self.method == "standard":
            self._scaler = StandardScaler()
        elif self.method == "robust":
            self._scaler = RobustScaler()
        elif self.method == "minmax":
            self._scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method '{method}'. Choose 'standard', 'robust', or 'minmax'.")

        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.params: Dict[str, Any] = {}

    def fit(self, df_train: pd.DataFrame, feature_columns: List[str]) -> "LeakageSafeScaler":
        """
        Fit scaler parameters exclusively on training data.
        """
        self.feature_names = [col for col in feature_columns if col in df_train.columns]
        X = df_train[self.feature_names].values.astype(float)
        
        # Replace NaNs with 0 temporarily for fitting if any exist
        X_fit = np.nan_to_num(X, nan=0.0)
        self._scaler.fit(X_fit)
        self.is_fitted = True

        # Extract parameters for transparency
        if hasattr(self._scaler, "mean_"):
            self.params = {
                "method": self.method,
                "features": self.feature_names,
                "means": self._scaler.mean_.tolist(),
                "scales": self._scaler.scale_.tolist(),
            }
        elif hasattr(self._scaler, "center_"):
            self.params = {
                "method": self.method,
                "features": self.feature_names,
                "centers": self._scaler.center_.tolist(),
                "scales": self._scaler.scale_.tolist(),
            }

        logger.info("Fitted %s scaler on %d features across %d training samples", self.method, len(self.feature_names), len(df_train))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a DataFrame using fitted scaling parameters.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fit on training data before calling transform().")

        df_out = df.copy()
        if not self.feature_names:
            return df_out

        X = df[self.feature_names].values.astype(float)
        nan_mask = np.isnan(X)
        X_clean = np.nan_to_num(X, nan=0.0)
        X_scaled = self._scaler.transform(X_clean)

        # Restore NaNs so missingness indicators remain honest
        X_scaled[nan_mask] = np.nan

        for idx, col in enumerate(self.feature_names):
            df_out[col] = X_scaled[:, idx]

        return df_out

    def fit_transform(self, df_train: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """Fit on train and return transformed train DataFrame."""
        self.fit(df_train, feature_columns)
        return self.transform(df_train)

    def export_params_json(self, file_path: str) -> None:
        """Save scaler parameters to JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2)
