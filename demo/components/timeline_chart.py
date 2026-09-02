"""
Interactive Multi-Modal Patient Timeline Visualization using Plotly.
"""

from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_patient_timeline(
    df: pd.DataFrame,
    patient_id: str,
    highlight_window_start: Optional[pd.Timestamp] = None,
    highlight_window_end: Optional[pd.Timestamp] = None,
    show_wearables: bool = False,
) -> go.Figure:
    """
    Constructs an interactive multi-modal timeline:
    - Primary Y: CGM Glucose (mg/dL) with target range shading (70-180 mg/dL)
    - Meal Carbs (grams)
    - Bolus Deliveries (Units)
    - Basal Rate (U/hr)
    """
    df_p = df[df["patient_id"].astype(str) == str(patient_id)].sort_values("timestamp")
    if df_p.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data found for selected patient.", showarrow=False)
        return fig

    # Create subplots: Row 1 = Glucose, Insulin, Meals; Row 2 = Wearables (if toggled)
    rows = 2 if show_wearables and "heart_rate" in df_p.columns else 1
    row_heights = [0.7, 0.3] if rows == 2 else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=row_heights,
        specs=[[{"secondary_y": True}]] if rows == 1 else [[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    # 1. Target Glucose Range Shading (70 to 180 mg/dL)
    if "cgm_glucose" in df_p.columns:
        fig.add_hrect(
            y0=70,
            y1=180,
            fillcolor="rgba(16, 185, 129, 0.10)",
            layer="below",
            line_width=0,
            annotation_text="Target Range (70-180 mg/dL)",
            annotation_position="top left",
            row=1,
            col=1,
        )

        # CGM Glucose Line
        fig.add_trace(
            go.Scatter(
                x=df_p["timestamp"],
                y=df_p["cgm_glucose"],
                mode="lines",
                name="CGM Glucose (mg/dL)",
                line=dict(color="#0284c7", width=2.5),
                hovertemplate="Time: %{x}<br>Glucose: %{y:.1f} mg/dL<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # 2. Basal Insulin Rate (Secondary Y)
    if "basal_rate" in df_p.columns:
        fig.add_trace(
            go.Scatter(
                x=df_p["timestamp"],
                y=df_p["basal_rate"],
                mode="lines",
                name="Basal Rate (U/hr)",
                line=dict(color="#f59e0b", width=1.5, dash="dash"),
                hovertemplate="Basal Rate: %{y:.2f} U/hr<extra></extra>",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

    # 3. Meals (Carbohydrates)
    if "meal_carbs" in df_p.columns:
        meals = df_p[df_p["meal_carbs"] > 0]
        if not meals.empty:
            fig.add_trace(
                go.Scatter(
                    x=meals["timestamp"],
                    y=meals["cgm_glucose"].fillna(120),
                    mode="markers+text",
                    name="Meal Event (g)",
                    marker=dict(symbol="triangle-up", size=14, color="#10b981", line=dict(width=1, color="#047857")),
                    text=[f"{val:.0f}g" for val in meals["meal_carbs"]],
                    textposition="top center",
                    textfont=dict(color="#047857", size=11, family="sans-serif"),
                    hovertemplate="Meal: %{text}<br>Time: %{x}<extra></extra>",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

    # 4. Bolus Insulin Injections
    if "bolus_dose" in df_p.columns:
        boluses = df_p[df_p["bolus_dose"] > 0]
        if not boluses.empty:
            fig.add_trace(
                go.Scatter(
                    x=boluses["timestamp"],
                    y=boluses["cgm_glucose"].fillna(120),
                    mode="markers+text",
                    name="Bolus Delivery (U)",
                    marker=dict(symbol="circle", size=13, color="#ef4444", line=dict(width=2, color="#991b1b")),
                    text=[f"{val:.1f}U" for val in boluses["bolus_dose"]],
                    textposition="bottom center",
                    textfont=dict(color="#b91c1c", size=11, family="sans-serif"),
                    hovertemplate="Bolus Dose: %{text}<br>Time: %{x}<extra></extra>",
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

    # 5. Wearables (Heart Rate) on Row 2
    if rows == 2 and "heart_rate" in df_p.columns:
        valid_hr = df_p[df_p["heart_rate"].notna()]
        if not valid_hr.empty:
            fig.add_trace(
                go.Scatter(
                    x=valid_hr["timestamp"],
                    y=valid_hr["heart_rate"],
                    mode="lines",
                    name="Heart Rate (BPM)",
                    line=dict(color="#8b5cf6", width=1.5),
                    hovertemplate="Heart Rate: %{y:.0f} BPM<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.update_yaxes(title_text="Heart Rate (BPM)", row=2, col=1)

    # 6. Highlight Selected 120-minute Window
    if highlight_window_start and highlight_window_end:
        fig.add_vrect(
            x0=highlight_window_start,
            x1=highlight_window_end,
            fillcolor="rgba(59, 130, 246, 0.18)",
            layer="below",
            line=dict(color="#2563eb", width=2, dash="dash"),
            annotation_text="120-min Window [t-120m, t]",
            annotation_position="top right",
            row=1,
            col=1,
        )
        # Prediction instant vertical line
        fig.add_vline(
            x=highlight_window_end,
            line=dict(color="#dc2626", width=2.5),
            annotation_text="Prediction Time t",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

    # Layout styling
    fig.update_layout(
        height=520 if rows == 2 else 420,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )

    fig.update_yaxes(title_text="Glucose (mg/dL)", range=[30, 380], row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Basal Rate (U/hr)", range=[0, 3.5], showgrid=False, row=1, col=1, secondary_y=True)
    fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6")
    fig.update_yaxes(showgrid=True, gridcolor="#f3f4f6")

    return fig
