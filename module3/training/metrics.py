"""
Evaluation metrics for bolus insulin dosage regression.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression and clinical evaluation metrics for bolus predictions.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    n_samples = len(y_true)
    if n_samples == 0:
        return {}

    # 1. Primary Regression Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    # R2 Score (guard against zero variance in ground truth)
    var_y = np.var(y_true)
    if var_y > 1e-6:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = 0.0

    # Pearson Correlation
    if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = 0.0

    # 2. Subgroup Analysis: Active vs. Zero Bolus Events
    active_mask = y_true > 0.1  # Clinically active bolus delivery (> 0.1 Units)
    if np.any(active_mask):
        active_mae = float(mean_absolute_error(y_true[active_mask], y_pred[active_mask]))
        active_rmse = float(np.sqrt(mean_squared_error(y_true[active_mask], y_pred[active_mask])))
    else:
        active_mae = 0.0
        active_rmse = 0.0

    zero_mask = ~active_mask
    if np.any(zero_mask):
        zero_mae = float(mean_absolute_error(y_true[zero_mask], y_pred[zero_mask]))
    else:
        zero_mae = 0.0

    # 3. Detection Quality (Binary classification: Bolus Needed vs. Not Needed)
    y_true_bin = (y_true > 0.1).astype(int)
    y_pred_bin = (y_pred > 0.1).astype(int)

    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    precision = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))

    # 4. Extended Active Bolus Characterization
    mean_act_actual = float(np.mean(y_true[active_mask])) if np.any(active_mask) else 0.0
    mean_act_pred = float(np.mean(y_pred[active_mask])) if np.any(active_mask) else 0.0
    pct_positive = float(np.mean(y_pred > 0.05) * 100.0)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": pearson_r,
        "active_bolus_mae": active_mae,
        "active_bolus_rmse": active_rmse,
        "zero_bolus_mae": zero_mae,
        "bolus_detection_f1": f1,
        "bolus_detection_precision": precision,
        "bolus_detection_recall": recall,
        "mean_actual_active_dose": mean_act_actual,
        "mean_predicted_active_dose": mean_act_pred,
        "percentage_predictions_positive": pct_positive,
        "total_samples": n_samples,
        "active_samples": int(np.sum(active_mask)),
    }
