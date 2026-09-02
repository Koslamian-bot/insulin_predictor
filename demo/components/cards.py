"""
Styled UI Cards and Metric Badges for the Demo Dashboard.
"""

import streamlit as st


def render_kpi_card(title: str, value: str, subtitle: str = "", border_color: str = "#2563eb"):
    """Renders a clean, modern KPI card."""
    html = f"""
    <div style="
        background: #ffffff;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 5px solid {border_color};
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    ">
        <div style="color: #6b7280; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
            {title}
        </div>
        <div style="color: #111827; font-size: 1.8rem; font-weight: 700; margin: 4px 0;">
            {value}
        </div>
        <div style="color: #9ca3af; font-size: 0.8rem;">
            {subtitle}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_leakage_badge(is_leak_free: bool, target_dose: float, bolus_nodes_at_zero: int):
    """Renders an explicit, prominent Target Isolation / Leakage badge."""
    if is_leak_free:
        bg = "#ecfdf5"
        border = "#10b981"
        text_color = "#065f46"
        status_text = "TARGET ISOLATION AUDIT: PASS"
        detail_text = f"ZERO Bolus nodes present at prediction time t (delta_tau = 0.0). Actual Bolus dose ({target_dose:.2f} U) is isolated strictly to prediction target y."
        icon = "🛡️"
    else:
        bg = "#fef2f2"
        border = "#ef4444"
        text_color = "#991b1b"
        status_text = "TARGET ISOLATION AUDIT: CRITICAL FAIL (LEAKAGE DETECTED)"
        detail_text = f"FATAL: Found {bolus_nodes_at_zero} Bolus nodes at prediction time t! Ground truth is leaking into input features X!"
        icon = "⚠️"

    html = f"""
    <div style="
        background-color: {bg};
        border: 2px solid {border};
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
    ">
        <div style="color: {text_color}; font-size: 1.1rem; font-weight: 700; display: flex; align-items: center; gap: 8px;">
            <span>{icon}</span> {status_text}
        </div>
        <div style="color: {text_color}; font-size: 0.95rem; margin-top: 6px; line-height: 1.4;">
            {detail_text}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
