"""
Interactive Knowledge Graph Visualizer for PyTorch Geometric Data Objects.

Renders real PyG graphs from data/graphs/*.pt using vis-network (HTML/JS)
with interactive node physics, drag, zoom, edge labeling, node feature inspection,
and dynamic zero-leakage verification.
"""

import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch_geometric.data import Data
import streamlit.components.v1 as components

from module2.ontology.schema_types import NodeType, EdgeType, NODE_TYPE_TO_OWL_CLASS

# Curated scientific color palette for ontology node classes
NODE_COLORS: Dict[int, Dict[str, str]] = {
    int(NodeType.PATIENT): {"bg": "#fbbf24", "border": "#d97706"},               # Gold / Amber
    int(NodeType.TIME_WINDOW): {"bg": "#38bdf8", "border": "#0284c7"},           # Light Blue
    int(NodeType.GLUCOSE_OBSERVATION): {"bg": "#60a5fa", "border": "#2563eb"},   # Blue
    int(NodeType.MEAL_OBSERVATION): {"bg": "#4ade80", "border": "#16a34a"},      # Green
    int(NodeType.BASAL_INSULIN): {"bg": "#fb923c", "border": "#ea580c"},         # Orange
    int(NodeType.TEMP_BASAL_INSULIN): {"bg": "#fde047", "border": "#ca8a04"},    # Yellow
    int(NodeType.BOLUS_INSULIN): {"bg": "#f87171", "border": "#dc2626"},         # Red (Historical Only)
    int(NodeType.HEART_RATE_OBSERVATION): {"bg": "#c084fc", "border": "#9333ea"},# Purple
    int(NodeType.SKIN_TEMP_OBSERVATION): {"bg": "#f472b6", "border": "#db2777"}, # Pink
    int(NodeType.AIR_TEMP_OBSERVATION): {"bg": "#a78bfa", "border": "#7c3aed"},  # Violet
    int(NodeType.GSR_OBSERVATION): {"bg": "#2dd4bf", "border": "#0d9488"},        # Teal
    int(NodeType.ACCELERATION_OBSERVATION): {"bg": "#a3e635", "border": "#65a30d"},# Lime
    int(NodeType.STEP_OBSERVATION): {"bg": "#818cf8", "border": "#4f46e5"},        # Indigo
}

DEFAULT_COLOR = {"bg": "#cbd5e1", "border": "#64748b"}


def parse_pyg_to_network(data: Data) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Translates PyTorch Geometric tensors (x, edge_index, edge_attr) into vis-network nodes and edges.
    """
    x_np = data.x.cpu().numpy()
    edge_index_np = data.edge_index.cpu().numpy()
    edge_attr_np = data.edge_attr.cpu().numpy() if hasattr(data, "edge_attr") and data.edge_attr is not None else np.zeros((edge_index_np.shape[1],))
    node_types_np = data.node_types.cpu().numpy() if hasattr(data, "node_types") else np.zeros((data.num_nodes,))

    nodes = []
    for i in range(data.num_nodes):
        nt_idx = int(node_types_np[i])
        nt_enum = NodeType(nt_idx)
        val = float(x_np[i, 0])
        tau = float(x_np[i, 1])
        colors = NODE_COLORS.get(nt_idx, DEFAULT_COLOR)

        # Descriptive display label
        if nt_enum == NodeType.PATIENT:
            label = f"Patient {getattr(data, 'patient_id', '559')}"
            size = 28
            shape = "diamond"
        elif nt_enum == NodeType.TIME_WINDOW:
            label = "TimeWindow\n[t-120m, t]"
            size = 24
            shape = "hexagon"
        elif nt_enum == NodeType.GLUCOSE_OBSERVATION:
            label = f"CGM\n{val:.0f} mg/dL"
            size = 18
            shape = "dot"
        elif nt_enum == NodeType.MEAL_OBSERVATION:
            label = f"Meal\n{val:.0f}g"
            size = 22
            shape = "triangle"
        elif nt_enum == NodeType.BASAL_INSULIN:
            label = f"Basal\n{val:.2f} U/h"
            size = 16
            shape = "dot"
        elif nt_enum == NodeType.BOLUS_INSULIN:
            label = f"Bolus (Hist)\n{val:.1f} U"
            size = 22
            shape = "star"
        elif nt_enum == NodeType.HEART_RATE_OBSERVATION:
            label = f"HR\n{val:.0f} bpm"
            size = 16
            shape = "dot"
        else:
            owl_name = NODE_TYPE_TO_OWL_CLASS.get(nt_enum, nt_enum.name)
            label = f"{owl_name}\n{val:.1f}"
            size = 16
            shape = "dot"

        # Tooltip content
        title = (
            f"<b>Node #{i}: {nt_enum.name}</b><br>"
            f"OWL Class: {NODE_TYPE_TO_OWL_CLASS.get(nt_enum, nt_enum.name)}<br>"
            f"Measurement: {val:.2f}<br>"
            f"Relative Time (&Delta;&tau;): {tau:.3f}<br>"
            f"{'Prediction Instant (t)' if np.isclose(tau, 0.0) else 'Historical Window'}"
        )

        nodes.append({
            "id": i,
            "label": label,
            "title": title,
            "shape": shape,
            "size": size,
            "color": {
                "background": colors["bg"],
                "border": colors["border"],
                "highlight": {"background": "#ffffff", "border": "#000000"},
            },
            "font": {"size": 11, "color": "#1e293b", "face": "Inter, sans-serif"},
            "borderWidth": 2,
        })

    edges = []
    num_edges = edge_index_np.shape[1]
    for e in range(num_edges):
        src = int(edge_index_np[0, e])
        dst = int(edge_index_np[1, e])
        et_val = int(edge_attr_np[e])
        et_enum = EdgeType(et_val)

        # Style based on edge type
        if et_enum == EdgeType.HAS_TIME_WINDOW:
            color = "#d97706"
            dashes = False
            arrows = "to"
            title = "hasTimeWindow (Patient &rarr; TimeWindow)"
        elif et_enum == EdgeType.HAS_OBSERVATION:
            color = "#94a3b8"
            dashes = True
            arrows = "to"
            title = "hasObservation (TimeWindow &rarr; Observation)"
        elif et_enum == EdgeType.SEQUENTIAL_TEMPORAL:
            color = "#2563eb"
            dashes = False
            arrows = "to"
            title = "sequentialTemporal (t_i &rarr; t_{i+1})"
        elif et_enum == EdgeType.CONTEMPORANEOUS:
            color = "#16a34a"
            dashes = False
            arrows = "to;from"
            title = "contemporaneous (Co-occurring at same step k)"
        else:
            color = "#cbd5e1"
            dashes = False
            arrows = "to"
            title = "edge"

        edges.append({
            "from": src,
            "to": dst,
            "color": {"color": color, "highlight": "#000000", "opacity": 0.75},
            "arrows": arrows,
            "dashes": dashes,
            "title": title,
            "width": 1.5,
        })

    return nodes, edges


def render_vis_network_html(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], height: int = 560) -> str:
    """
    Generates self-contained interactive vis-network HTML canvas.
    """
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
      <style type="text/css">
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; font-family: sans-serif; }}
        #mynetwork {{
          width: 100%;
          height: {height}px;
          border: 1px solid #e2e8f0;
          background: #f8fafc;
          border-radius: 8px;
        }}
        #legend {{
          position: absolute;
          top: 10px;
          right: 10px;
          background: rgba(255, 255, 255, 0.92);
          border: 1px solid #cbd5e1;
          border-radius: 6px;
          padding: 8px 12px;
          font-size: 11px;
          line-height: 1.6;
          box-shadow: 0 2px 4px rgba(0,0,0,0.05);
          pointer-events: none;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 6px; }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
      </style>
    </head>
    <body>
      <div id="legend">
        <div style="font-weight:bold; margin-bottom: 4px;">Node Legend</div>
        <div class="legend-item"><span class="dot" style="background:#fbbf24; border:1px solid #d97706;"></span> Patient Hub</div>
        <div class="legend-item"><span class="dot" style="background:#38bdf8; border:1px solid #0284c7;"></span> TimeWindow</div>
        <div class="legend-item"><span class="dot" style="background:#60a5fa; border:1px solid #2563eb;"></span> Glucose (CGM)</div>
        <div class="legend-item"><span class="dot" style="background:#4ade80; border:1px solid #16a34a;"></span> Meal Carbs</div>
        <div class="legend-item"><span class="dot" style="background:#fb923c; border:1px solid #ea580c;"></span> Basal Rate</div>
        <div class="legend-item"><span class="dot" style="background:#f87171; border:1px solid #dc2626;"></span> Historical Bolus</div>
        <div class="legend-item"><span class="dot" style="background:#c084fc; border:1px solid #9333ea;"></span> Wearable (HR/Temp)</div>
      </div>
      <div id="mynetwork"></div>

      <script type="text/javascript">
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});

        var container = document.getElementById('mynetwork');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          physics: {{
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {{
              gravitationalConstant: -38,
              centralGravity: 0.008,
              springLength: 60,
              springConstant: 0.08,
              damping: 0.85
            }},
            maxVelocity: 40,
            minVelocity: 0.2,
            stabilization: {{ iterations: 120, updateInterval: 25 }}
          }},
          interaction: {{
            hover: true,
            tooltipDelay: 100,
            navigationButtons: true,
            keyboard: false,
            zoomView: true
          }}
        }};

        var network = new vis.Network(container, data, options);
      </script>
    </body>
    </html>
    """
    return html


def audit_target_isolation(data: Data) -> Dict[str, Any]:
    """
    Performs dynamic, non-hardcoded zero-leakage target audit on a PyG graph:
    1. Scans nodes where delta_tau == 0.0 (prediction instant t).
    2. Asserts no node has type BOLUS_INSULIN.
    3. Finds historical boluses (tau < 0.0).
    4. Extracts target y value.
    """
    x_np = data.x.cpu().numpy()
    node_types_np = data.node_types.cpu().numpy() if hasattr(data, "node_types") else np.zeros((data.num_nodes,))
    target_dose = float(data.y.item()) if hasattr(data, "y") else 0.0

    # Find nodes at prediction instant t (tau ~= 0.0)
    zero_tau_indices = [
        i for i in range(data.num_nodes)
        if np.isclose(x_np[i, 1], 0.0, atol=1e-4) and int(node_types_np[i]) not in [int(NodeType.PATIENT), int(NodeType.TIME_WINDOW)]
    ]

    bolus_nodes_at_zero = [
        i for i in zero_tau_indices
        if int(node_types_np[i]) == int(NodeType.BOLUS_INSULIN)
    ]

    # Find historical boluses
    hist_bolus_indices = [
        i for i in range(data.num_nodes)
        if int(node_types_np[i]) == int(NodeType.BOLUS_INSULIN) and x_np[i, 1] < -1e-4
    ]
    historical_boluses = [
        {"node_id": i, "dose": float(x_np[i, 0]), "delta_tau": float(x_np[i, 1])}
        for i in hist_bolus_indices
    ]

    is_leak_free = len(bolus_nodes_at_zero) == 0

    return {
        "is_leak_free": is_leak_free,
        "target_dose": target_dose,
        "bolus_nodes_at_zero_count": len(bolus_nodes_at_zero),
        "bolus_nodes_at_zero_indices": bolus_nodes_at_zero,
        "historical_boluses": historical_boluses,
        "total_nodes": data.num_nodes,
        "total_edges": data.edge_index.shape[1],
        "x_shape": list(data.x.shape),
        "edge_index_shape": list(data.edge_index.shape),
    }
