import os
import sys

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
import torch

from module2.graph.window_slicer import TimeWindowSlice

from module2.graph.graph_builder import GraphBuilder
from module2.dataset.pyg_converter import PyGConverter
from module3.config import ModelConfig
from module3.models import build_model
from module3.data.graph_dataset import create_dataloader

def test_hupa():
    hupa_csv = r"c:\Final year\Insulin prediction\HUPA-UCM Diabetes Dataset\Preprocessed\HUPA0001P.csv"
    df = pd.read_csv(hupa_csv, sep=";")
    
    # We choose prediction time at index 240: 2018-06-14 14:40:00
    # Window spans index 216 to 240 (25 rows: 24 steps lookback + step 24 is prediction instant t)
    sub = df.iloc[216:241].copy().reset_index(drop=True)
    
    target_dose = float(sub.iloc[-1]["bolus_volume_delivered"])
    start_time = pd.to_datetime(sub.iloc[0]["time"])
    pred_time = pd.to_datetime(sub.iloc[-1]["time"])
    
    print(f"Prediction time t: {pred_time}")
    print(f"Lookback start:    {start_time}")
    print(f"Actual Target bolus at t: {target_dose} Units")
    
    # Canonical mapping for these 25 rows
    # In OhioT1DM training: cgm_glucose mean ~ 158.0, std ~ 57.0; meal_carbs mean ~ 2.5, std ~ 15.0; hr mean ~ 80, std ~ 15
    OHIO_CGM_MEAN, OHIO_CGM_STD = 158.0, 57.0
    OHIO_CARB_MEAN, OHIO_CARB_STD = 2.5, 15.0
    OHIO_HR_MEAN, OHIO_HR_STD = 75.0, 15.0
    
    canonical_df = pd.DataFrame()
    canonical_df["timestamp"] = pd.to_datetime(sub["time"])
    canonical_df["cgm_glucose"] = (sub["glucose"].astype(float) - OHIO_CGM_MEAN) / OHIO_CGM_STD
    canonical_df["meal_carbs"] = (sub["carb_input"].fillna(0.0).astype(float) - OHIO_CARB_MEAN) / OHIO_CARB_STD
    canonical_df["basal_rate"] = sub["basal_rate"].fillna(0.0).astype(float)
    canonical_df["bolus_dose"] = sub["bolus_volume_delivered"].fillna(0.0).astype(float)
    canonical_df["heart_rate"] = (sub["heart_rate"].fillna(0.0).astype(float) - OHIO_HR_MEAN) / OHIO_HR_STD
    canonical_df["steps"] = sub["steps"].fillna(0.0).astype(float) / 50.0

    
    # Flags for modalities missing in HUPA
    canonical_df["has_heart_rate"] = 1
    canonical_df["heart_rate_missing_mask"] = 0
    canonical_df["has_skin_temperature"] = 0
    canonical_df["skin_temperature_missing_mask"] = 1
    canonical_df["has_air_temperature"] = 0
    canonical_df["air_temperature_missing_mask"] = 1
    canonical_df["has_gsr"] = 0
    canonical_df["gsr_missing_mask"] = 1
    canonical_df["has_acceleration"] = 0
    canonical_df["acceleration_missing_mask"] = 1
    
    # Target builder masking: bolus_dose at prediction instant (last row) must be 0 in input features
    canonical_df.loc[len(canonical_df)-1, "bolus_dose"] = 0.0
    
    slice_obj = TimeWindowSlice(
        window_id="HUPA0001P_w0240",
        patient_id="HUPA0001P",
        start_time=start_time,
        prediction_time=pred_time,
        data=canonical_df,
        target_dose=target_dose,
        split_label="hupa_demo",
    )
    
    builder = GraphBuilder()
    graph_dict = builder.build_graph(slice_obj)
    
    print(f"Graph constructed:")
    print(f"  Nodes: {graph_dict.x.shape[0]}")
    print(f"  Edges: {graph_dict.edge_index.shape[1]}")
    print(f"  y target: {graph_dict.y}")
    
    # PyG conversion
    pyg_data = PyGConverter.to_pyg_data(graph_dict)
    
    # Test zero-leakage
    # Find any bolus node at delta_tau == 0.0
    from module2.ontology.schema_types import NodeType
    x_np = pyg_data.x.numpy()
    nt_np = pyg_data.node_types.numpy()
    zero_tau_bolus = [
        i for i in range(pyg_data.num_nodes)
        if nt_np[i] == int(NodeType.BOLUS_INSULIN) and np.isclose(x_np[i, 1], 0.0, atol=1e-4)
    ]
    print(f"Bolus nodes at delta_tau == 0.0 (leakage check): {len(zero_tau_bolus)}")
    assert len(zero_tau_bolus) == 0, "Leakage detected!"
    
    # Model inference
    model_types = ["gcn_gru", "gcn_lstm", "gcn_bilstm"]
    loader = create_dataloader([pyg_data], batch_size=1, shuffle=False)
    batch = next(iter(loader))
    
    for mt in model_types:
        ckpt_path = os.path.join("data", "models", f"{mt}_best.pt")
        if os.path.exists(ckpt_path):
            cfg = ModelConfig(model_type=mt)
            m = build_model(cfg)
            m.load_checkpoint(ckpt_path)
            m.eval()
            with torch.no_grad():
                pred = float(m(batch).item())
            err = abs(target_dose - pred)
            print(f"Model: {mt.upper():<10} | Pred: {pred:.2f} U | Actual: {target_dose:.2f} U | Error: {err:.2f} U")

if __name__ == "__main__":
    test_hupa()
