"""
Cached Data Loader for Demonstration Module.

Efficiently loads and caches:
- Canonical time-series datasets (Train, Validation, Test)
- PyTorch Geometric graph datasets (.pt)
- Ontology schema (diabetes_insulin_ontology.owl)
- Preprocessing and Graph Generation reports
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import torch
from torch_geometric.data import Data
import streamlit as st

from module2.ontology.ontology_parser import OntologyParser, OntologySchema
from module2.ontology.schema_types import NodeType, EdgeType, NODE_TYPE_TO_OWL_CLASS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@st.cache_data
def load_canonical_df(split: str = "train", max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load canonical CSV for a given split."""
    path = os.path.join(BASE_DIR, "data", "processed", f"canonical_{split}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, nrows=max_rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data
def get_dataset_metadata() -> Dict[str, Any]:
    """Return dataset size and split statistics."""
    splits = ["train", "val", "test"]
    stats = {}
    for s in splits:
        path = os.path.join(BASE_DIR, "data", "processed", f"canonical_{s}.csv")
        if os.path.exists(path):
            # Fast line count
            with open(path, "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1
            stats[s] = {"row_count": row_count, "path": path}
        else:
            stats[s] = {"row_count": 0, "path": path}
    return stats


@st.cache_resource
def load_graph_split(split: str = "train") -> List[Data]:
    """Load pre-generated PyTorch Geometric graph list from disk."""
    path = os.path.join(BASE_DIR, "data", "graphs", f"{split}.pt")
    if not os.path.exists(path):
        return []
    try:
        # Load weights safely
        data_list = torch.load(path, weights_only=False)
        return data_list
    except Exception as e:
        st.error(f"Error loading graph split '{split}': {e}")
        return []


@st.cache_resource
def load_ontology_schema() -> Optional[OntologySchema]:
    """Parse and return the in-memory schema from diabetes_insulin_ontology.owl."""
    owl_path = os.path.join(BASE_DIR, "diabetes_insulin_ontology.owl")
    if not os.path.exists(owl_path):
        return None
    parser = OntologyParser(owl_path)
    return parser.parse()


@st.cache_data
def load_quality_report() -> str:
    """Read Module 1 data quality report markdown."""
    path = os.path.join(BASE_DIR, "data", "reports", "data_quality_report.md")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "No Data Quality Report found at `data/reports/data_quality_report.md`."


@st.cache_data
def load_graph_report_metrics() -> Dict[str, Any]:
    """Read Module 2 graph generation JSON report."""
    path = os.path.join(BASE_DIR, "data", "reports", "graph_generation_report.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_raw_to_canonical_example() -> Dict[str, Any]:
    """Demonstration snippet illustrating transformation from raw XML to 5-min row."""
    return {
        "raw_xml_glucose": '<glucose_level ts="07-12-2021 11:23:00" value="142"/>',
        "raw_xml_meal": '<meal ts="07-12-2021 11:24:00" carbs="65" type="Lunch"/>',
        "raw_xml_bolus": '<bolus ts_begin="07-12-2021 11:25:00" dose="5.2"/>',
        "canonical_events": [
            {"timestamp": "2021-12-07 11:23:00", "modality": "CGM_GLUCOSE", "value": 142.0, "unit": "mg/dL"},
            {"timestamp": "2021-12-07 11:24:00", "modality": "MEAL_CARBS", "value": 65.0, "unit": "grams"},
            {"timestamp": "2021-12-07 11:25:00", "modality": "BOLUS_DOSE", "value": 5.2, "unit": "Units"},
        ],
        "synchronized_5min_row": {
            "timestamp": "2021-12-07 11:25:00",
            "cgm_glucose": 142.0,
            "meal_carbs": 65.0,
            "bolus_dose": 5.2,
            "basal_rate": 0.80,
            "cgm_roc_15m": 1.25,
            "iob_bilinear_decay": 2.14,
            "target_current_bolus_dose": 5.2,
        },
    }
