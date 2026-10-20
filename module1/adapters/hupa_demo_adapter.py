"""
HUPA Cross-Dataset Demonstration Adapter.

Provides an isolated, non-destructive bridge between HUPA-UCM raw patient records
and the canonical ontology knowledge graph representation for cross-dataset proof-of-concept.

Guarantees:
- Strictly read-only on raw HUPA files.
- Strictly isolated: does not alter existing OhioT1DM adapter or pipelines.
- Enforces strict zero target leakage at prediction instant t.
- Maps available HUPA modalities to canonical variables and OWL ontology classes.
- Explicitly flags unrecorded modalities as unavailable (no fabricated data).
"""

import os
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import torch

from module2.graph.window_slicer import TimeWindowSlice
from module2.graph.graph_builder import GraphBuilder
from module2.dataset.pyg_converter import PyGConverter
from module2.ontology.schema_types import NodeType
from module3.config import ModelConfig
from module3.models import build_model
from module3.data.graph_dataset import create_dataloader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUPA_DEFAULT_PATH = os.path.join(BASE_DIR, "HUPA-UCM Diabetes Dataset", "Preprocessed", "HUPA0001P.csv")

# Baseline normalization parameters derived from OhioT1DM training distribution
# Ensures the GCN receives feature magnitudes compatible with trained weights
OHIO_CGM_MEAN: float = 158.0
OHIO_CGM_STD: float = 57.0
OHIO_CARB_MEAN: float = 2.5
OHIO_CARB_STD: float = 15.0
OHIO_HR_MEAN: float = 75.0
OHIO_HR_STD: float = 15.0


def get_hupa_demo_patient_info(file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspects and returns structural metadata for the selected HUPA demonstration patient.
    """
    csv_path = file_path or HUPA_DEFAULT_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"HUPA file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=";")
    boluses = df[df["bolus_volume_delivered"] > 0]
    meals = df[df["carb_input"] > 0]

    return {
        "patient_id": "HUPA0001P",
        "dataset_source": "HUPA-UCM",
        "file_path": csv_path,
        "total_rows": len(df),
        "sampling_frequency": "5 minutes",
        "time_start": str(df["time"].min()),
        "time_end": str(df["time"].max()),
        "total_bolus_events": len(boluses),
        "total_meal_events": len(meals),
        "columns_in_file": list(df.columns),
        "available_modalities": [
            "CGM Glucose (glucose)",
            "Bolus Insulin (bolus_volume_delivered)",
            "Basal Insulin Rate (basal_rate)",
            "Carbohydrate Meals (carb_input)",
            "Heart Rate (heart_rate)",
            "Step Count (steps)",
            "Energy Expenditure (calories)",
        ],
        "unavailable_modalities": [
            "Skin Temperature (not recorded in HUPA)",
            "Air Temperature (not recorded in HUPA)",
            "Galvanic Skin Response (GSR) (not recorded in HUPA)",
            "3-Axis Acceleration (not recorded in HUPA)",
            "Sleep Intervals (not recorded in HUPA)",
        ],
    }


def extract_hupa_demo_window(
    file_path: Optional[str] = None,
    prediction_time_str: str = "2018-06-14 14:40:00",
    window_minutes: int = 120,
    step_minutes: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, TimeWindowSlice]:
    """
    Extracts a 120-minute pre-prediction window from HUPA0001P leading up to prediction time t.

    Returns:
        raw_window_df: Original HUPA records across [t - 120m, t].
        canonical_df: Transformed and normalized canonical dataframe.
        slice_obj: TimeWindowSlice ready for graph construction.
    """
    csv_path = file_path or HUPA_DEFAULT_PATH
    df = pd.read_csv(csv_path, sep=";")
    df["dt"] = pd.to_datetime(df["time"])
    df = df.sort_values("dt").reset_index(drop=True)

    target_dt = pd.to_datetime(prediction_time_str)
    matches = df.index[df["dt"] == target_dt].tolist()
    if not matches:
        raise ValueError(f"Timestamp {prediction_time_str} not found in HUPA records.")

    pred_idx = matches[0]
    num_steps = window_minutes // step_minutes  # 24 steps
    start_idx = pred_idx - num_steps

    if start_idx < 0:
        raise ValueError(f"Insufficient history for 120-min window before {prediction_time_str}.")

    # Subslice covering [t - 120m, t] (25 rows total: 24 steps history + prediction instant t)
    raw_sub = df.iloc[start_idx : pred_idx + 1].copy().reset_index(drop=True)

    target_dose = float(raw_sub.iloc[-1]["bolus_volume_delivered"])
    start_time = raw_sub.iloc[0]["dt"]
    pred_time = raw_sub.iloc[-1]["dt"]

    # Construct Canonical DataFrame
    canonical_df = pd.DataFrame()
    canonical_df["timestamp"] = raw_sub["dt"]
    canonical_df["patient_id"] = "HUPA0001P"

    # 1. Glucose: z-score normalized to match Ohio training feature distribution
    canonical_df["cgm_glucose"] = (raw_sub["glucose"].astype(float) - OHIO_CGM_MEAN) / OHIO_CGM_STD
    canonical_df["cgm_missing_mask"] = 0

    # 2. Meal Carbs: normalized to match Ohio training feature distribution
    canonical_df["meal_carbs"] = (raw_sub["carb_input"].fillna(0.0).astype(float) - OHIO_CARB_MEAN) / OHIO_CARB_STD

    # 3. Basal Rate: actual U/h as expected by Ohio graph builder
    canonical_df["basal_rate"] = raw_sub["basal_rate"].fillna(0.0).astype(float)

    # 4. Bolus Dose: historical only; current bolus at time t is strictly zeroed out
    canonical_df["bolus_dose"] = raw_sub["bolus_volume_delivered"].fillna(0.0).astype(float)
    canonical_df.loc[len(canonical_df) - 1, "bolus_dose"] = 0.0  # ZERO TARGET LEAKAGE MASKING

    # 5. Wearables: Heart rate and Steps
    canonical_df["heart_rate"] = (raw_sub["heart_rate"].fillna(0.0).astype(float) - OHIO_HR_MEAN) / OHIO_HR_STD
    canonical_df["has_heart_rate"] = 1
    canonical_df["heart_rate_missing_mask"] = 0

    canonical_df["steps"] = raw_sub["steps"].fillna(0.0).astype(float) / 50.0

    # 6. Explicitly unavailable modalities marked as missing (no fabrication)
    canonical_df["has_skin_temperature"] = 0
    canonical_df["skin_temperature_missing_mask"] = 1
    canonical_df["has_air_temperature"] = 0
    canonical_df["air_temperature_missing_mask"] = 1
    canonical_df["has_gsr"] = 0
    canonical_df["gsr_missing_mask"] = 1
    canonical_df["has_acceleration"] = 0
    canonical_df["acceleration_missing_mask"] = 1

    slice_obj = TimeWindowSlice(
        window_id=f"HUPA0001P_{pred_time.strftime('%Y%m%d_%H%M')}",
        patient_id="HUPA0001P",
        start_time=start_time,
        prediction_time=pred_time,
        data=canonical_df,
        target_dose=target_dose,
        split_label="hupa_demo",
    )

    return raw_sub, canonical_df, slice_obj


def build_hupa_patient_graph(slice_obj: TimeWindowSlice) -> Tuple[Any, Dict[str, Any]]:
    """
    Constructs a PyTorch Geometric Data object from the HUPA TimeWindowSlice.
    Performs zero-leakage target audit.
    """
    builder = GraphBuilder(normalize_values=True)
    graph_dict = builder.build_graph(slice_obj)
    pyg_data = PyGConverter.to_pyg_data(graph_dict)

    # Audit zero leakage
    x_np = pyg_data.x.cpu().numpy()
    nt_np = pyg_data.node_types.cpu().numpy()
    zero_tau_bolus_nodes = [
        i for i in range(pyg_data.num_nodes)
        if nt_np[i] == int(NodeType.BOLUS_INSULIN) and np.isclose(x_np[i, 1], 0.0, atol=1e-4)
    ]

    hist_bolus_nodes = [
        {"node_id": i, "dose": float(x_np[i, 0]), "delta_tau": float(x_np[i, 1])}
        for i in range(pyg_data.num_nodes)
        if nt_np[i] == int(NodeType.BOLUS_INSULIN) and x_np[i, 1] < -1e-4
    ]

    audit_info = {
        "num_nodes": pyg_data.num_nodes,
        "num_edges": pyg_data.edge_index.shape[1],
        "is_leak_free": len(zero_tau_bolus_nodes) == 0,
        "bolus_nodes_at_zero": len(zero_tau_bolus_nodes),
        "historical_boluses": hist_bolus_nodes,
        "actual_target_dose": float(pyg_data.y.item()),
    }

    return pyg_data, audit_info


def run_hupa_inference(
    pyg_data: Any,
    model_name: str = "gcn_gru",
) -> Dict[str, Any]:
    """
    Executes inference with an existing trained model checkpoint on the HUPA graph.
    """
    models_dir = os.path.join(BASE_DIR, "data", "models")
    ckpt_path = os.path.join(models_dir, f"{model_name}_best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for model '{model_name}': {ckpt_path}")

    cfg = ModelConfig(model_type=model_name)
    model = build_model(cfg)
    model.load_checkpoint(ckpt_path)
    model.eval()

    loader = create_dataloader([pyg_data], batch_size=1, shuffle=False)
    batch = next(iter(loader))

    with torch.no_grad():
        if "gcn" in model_name:
            pred_val = float(model(batch).item())
        else:
            from module3.data.graph_dataset import extract_temporal_sequence_from_batch
            x_seq, _ = extract_temporal_sequence_from_batch(batch)
            pred_val = float(model(x_seq).item())

    actual_val = float(pyg_data.y.item())
    abs_err = abs(actual_val - pred_val)

    return {
        "model_name": model_name.upper(),
        "checkpoint_path": ckpt_path,
        "predicted_dose": round(pred_val, 2),
        "actual_dose": round(actual_val, 2),
        "absolute_error": round(abs_err, 2),
    }
