"""
Dashboard Chart Components
===========================
All Plotly figures used by the Streamlit dashboard.
Each function is self-contained and returns a go.Figure ready for
st.plotly_chart().  Dark-theme consistent with the dashboard palette.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared palette & theme ─────────────────────────────────────────────────────

_C = {
    "fraud":   "#EF4444",
    "legit":   "#10B981",
    "blue":    "#4F46E5",
    "warning": "#F59E0B",
    "orange":  "#F97316",
    "bg":      "#0F172A",
    "card":    "#1E293B",
    "border":  "#334155",
    "text":    "#E2E8F0",
    "muted":   "#94A3B8",
}

_BASE_LAYOUT = dict(
    plot_bgcolor  = _C["card"],
    paper_bgcolor = _C["card"],
    font          = dict(color=_C["text"], family="Inter, sans-serif", size=12),
    margin        = dict(l=48, r=20, t=44, b=40),
    legend        = dict(bgcolor=_C["card"], bordercolor=_C["border"], borderwidth=1),
    xaxis         = dict(gridcolor=_C["border"], zeroline=False),
    yaxis         = dict(gridcolor=_C["border"], zeroline=False),
)


def _layout(**overrides) -> dict:
    """Merge base layout with caller-specific overrides."""
    base = dict(_BASE_LAYOUT)
    base.update(overrides)
    return base


# ── Risk gauge ────────────────────────────────────────────────────────────────

def risk_gauge(risk_score: float, risk_level: str) -> go.Figure:
    """
    Animated gauge chart showing the hybrid risk score [0, 100].
    Color changes with risk level: green → amber → orange → red.
    """
    colour_map = {
        "LOW":      _C["legit"],
        "MEDIUM":   _C["warning"],
        "HIGH":     _C["orange"],
        "CRITICAL": _C["fraud"],
    }
    colour = colour_map.get(risk_level, _C["blue"])
    pct    = round(risk_score * 100, 1)

    fig = go.Figure(go.Indicator(
        mode   = "gauge+number",
        value  = pct,
        domain = {"x": [0, 1], "y": [0, 1]},
        title  = {
            "text": (
                f"Risk Score<br>"
                f"<span style='font-size:0.85em;color:{colour};font-weight:700'>"
                f"{risk_level}</span>"
            ),
            "font": {"size": 16, "color": _C["text"]},
        },
        number = {"suffix": "%", "font": {"size": 36, "color": colour}},
        gauge  = {
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": _C["muted"],
                "tickfont":  {"color": _C["muted"], "size": 10},
            },
            "bar":         {"color": colour, "thickness": 0.22},
            "bgcolor":     _C["card"],
            "borderwidth": 2,
            "bordercolor": colour,
            "steps": [
                {"range": [0,  35],  "color": "#064E3B"},   # dark green
                {"range": [35, 65],  "color": "#3B2A00"},   # dark amber
                {"range": [65, 85],  "color": "#431407"},   # dark orange
                {"range": [85, 100], "color": "#450A0A"},   # dark red
            ],
            "threshold": {
                "line":      {"color": "white", "width": 3},
                "thickness": 0.8,
                "value":     50,
            },
        },
    ))
    fig.update_layout(**_layout(height=280, margin=dict(l=20, r=20, t=50, b=20)))
    return fig


# ── Transaction timeline ──────────────────────────────────────────────────────

def live_transaction_timeline(transactions: List[dict]) -> go.Figure:
    """
    Scatter chart of recent transactions — legitimate (green circles)
    vs fraud (red X marks, oversized for immediate visibility).
    """
    fig = go.Figure()

    if not transactions:
        fig.update_layout(**_layout(title="Live Transaction Feed — waiting for data…", height=320))
        return fig

    df  = pd.DataFrame(transactions)
    idx = list(range(len(df)))

    legit = df[~df["is_fraud"]]
    fraud = df[df["is_fraud"]]

    if not legit.empty:
        l_idx = [i for i, f in enumerate(df["is_fraud"]) if not f]
        fig.add_trace(go.Scatter(
            x    = l_idx,
            y    = legit["amount"].tolist(),
            mode = "markers",
            name = "✅ Legitimate",
            marker = dict(color=_C["legit"], size=7, opacity=0.8, symbol="circle"),
            hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "Amount: $%{y:,.2f}<br>"
                "Risk: %{customdata[1]}<extra></extra>"
            ),
            customdata = list(zip(
                legit.get("id", legit.get("transaction_id", ["N/A"]*len(legit))),
                legit.get("risk_score", [0]*len(legit)).round(4).tolist()
                    if hasattr(legit.get("risk_score", None), "round") else [0]*len(legit),
            )),
        ))

    if not fraud.empty:
        f_idx = [i for i, f in enumerate(df["is_fraud"]) if f]
        fig.add_trace(go.Scatter(
            x    = f_idx,
            y    = fraud["amount"].tolist(),
            mode = "markers",
            name = "🚨 Fraud",
            marker = dict(
                color  = _C["fraud"],
                size   = 16,
                symbol = "x",
                line   = dict(width=2.5, color="white"),
            ),
            hovertemplate = (
                "<b>🚨 FRAUD: %{customdata[0]}</b><br>"
                "Amount: $%{y:,.2f}<extra></extra>"
            ),
            customdata = [[t.get("id", t.get("transaction_id", "N/A"))] for t in
                          transactions if t.get("is_fraud")],
        ))

    fig.update_layout(**_layout(
        title      = "Live Transaction Feed",
        xaxis_title = "Transaction #",
        yaxis_title = "Amount ($USD)",
        height      = 330,
        showlegend  = True,
    ))
    return fig


# ── Rolling fraud rate ────────────────────────────────────────────────────────

def fraud_trend_chart(transactions: List[dict], window: int = 20) -> go.Figure:
    """Rolling fraud rate trend with reference line at expected base rate."""
    fig = go.Figure()

    if len(transactions) < 2:
        fig.update_layout(**_layout(title="Collecting data…", height=280))
        return fig

    df  = pd.DataFrame(transactions)
    w   = min(window, len(df))
    df["rolling_rate"] = df["is_fraud"].rolling(w, min_periods=1).mean() * 100

    fig.add_trace(go.Scatter(
        x          = list(range(len(df))),
        y          = df["rolling_rate"],
        mode       = "lines",
        fill       = "tozeroy",
        name       = f"Fraud Rate ({w}-txn rolling avg)",
        line       = dict(color=_C["fraud"], width=2),
        fillcolor  = "rgba(239,68,68,0.12)",
    ))
    fig.add_hline(
        y=0.3,
        line_dash="dot",
        line_color=_C["warning"],
        annotation_text="Base rate 0.3%",
        annotation_position="top right",
        annotation_font_color=_C["warning"],
    )
    fig.update_layout(**_layout(
        title       = "Rolling Fraud Rate",
        xaxis_title = "Transaction #",
        yaxis_title = "Fraud Rate (%)",
        height      = 280,
    ))
    return fig


# ── Probability distribution ──────────────────────────────────────────────────

def probability_distribution_chart(
    probabilities: List[float],
    is_fraud_list: List[bool],
) -> go.Figure:
    """
    Overlaid histogram: fraud vs legitimate probability distributions.
    A vertical dashed line marks the 0.5 decision boundary.
    """
    fig = go.Figure()

    legit_probs = [p for p, f in zip(probabilities, is_fraud_list) if not f]
    fraud_probs = [p for p, f in zip(probabilities, is_fraud_list) if f]

    bins = dict(start=0, end=1.0, size=0.04)

    fig.add_trace(go.Histogram(
        x=legit_probs, name="Legitimate",
        xbins=bins, marker_color=_C["legit"], opacity=0.70,
    ))
    fig.add_trace(go.Histogram(
        x=fraud_probs, name="🚨 Fraud",
        xbins=bins, marker_color=_C["fraud"], opacity=0.85,
    ))
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="white",
        annotation_text="Decision boundary",
        annotation_position="top",
        annotation_font_color=_C["muted"],
    )
    fig.update_layout(**_layout(
        title       = "Fraud Probability Distribution",
        barmode     = "overlay",
        xaxis_title = "Fraud Probability",
        yaxis_title = "Count",
        height      = 320,
        showlegend  = True,
    ))
    return fig


# ── SHAP waterfall ────────────────────────────────────────────────────────────

def shap_waterfall_chart(shap_data: dict) -> go.Figure:
    """
    Horizontal bar chart of SHAP feature impacts.
    Red bars push toward fraud; green bars push toward legitimate.
    """
    fig = go.Figure()

    features = (shap_data or {}).get("top_features", [])
    if not features:
        fig.update_layout(**_layout(title="No SHAP data available", height=350))
        return fig

    # Sort ascending by |impact| so largest bar is at the top
    features = sorted(features, key=lambda x: abs(x["impact"]))[-12:]

    names    = [f["feature"] for f in features]
    impacts  = [f["impact"]  for f in features]
    vals     = [f["value"]   for f in features]
    colours  = [_C["fraud"] if v > 0 else _C["legit"] for v in impacts]
    labels   = [f"{'+' if v > 0 else ''}{v:.4f}" for v in impacts]
    y_labels = [f"{n}  (={round(v, 3)})" for n, v in zip(names, vals)]

    fig.add_trace(go.Bar(
        x           = impacts,
        y           = y_labels,
        orientation = "h",
        marker_color= colours,
        text        = labels,
        textposition= "outside",
        cliponaxis  = False,
    ))
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(**_layout(
        title       = "SHAP Feature Impact  (→ Fraud  |  ← Legitimate)",
        xaxis_title = "SHAP Value",
        height      = max(300, 30 * len(features) + 80),
        showlegend  = False,
    ))
    return fig


# ── Session summary donut ─────────────────────────────────────────────────────

def session_summary_chart(total: int, fraud: int) -> go.Figure:
    """Donut chart of fraud vs legitimate split for the current session."""
    legit = max(total - fraud, 0)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "indicator"}]],
        column_widths=[0.55, 0.45],
    )
    fig.add_trace(go.Pie(
        labels    = ["Legitimate", "Fraudulent"],
        values    = [legit, fraud],
        hole      = 0.58,
        marker    = dict(colors=[_C["legit"], _C["fraud"]], line=dict(color=_C["card"], width=2)),
        textinfo  = "percent+label",
        textfont  = dict(size=11),
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ), row=1, col=1)

    rate = (fraud / total * 100) if total > 0 else 0.0
    fig.add_trace(go.Indicator(
        mode   = "number+delta",
        value  = round(rate, 3),
        number = {"suffix": "%", "font": {"size": 30, "color": _C["fraud"]}},
        title  = {"text": "Fraud Rate", "font": {"color": _C["text"], "size": 13}},
        delta  = {"reference": 0.3, "valueformat": ".3f",
                  "increasing": {"color": _C["fraud"]},
                  "decreasing": {"color": _C["legit"]}},
    ), row=1, col=2)

    fig.update_layout(**_layout(
        title      = "Session Summary",
        height     = 270,
        showlegend = True,
    ))
    return fig


# ── Amount vs risk scatter ────────────────────────────────────────────────────

def amount_vs_risk_scatter(transactions: List[dict]) -> go.Figure:
    """Scatter of transaction amount vs risk score, coloured by fraud flag."""
    fig = go.Figure()
    if not transactions:
        fig.update_layout(**_layout(title="No data yet", height=300))
        return fig

    df = pd.DataFrame(transactions)
    for is_fraud, colour, name, sym in [
        (False, _C["legit"],   "Legitimate", "circle"),
        (True,  _C["fraud"],   "🚨 Fraud",   "x"),
    ]:
        sub = df[df["is_fraud"] == is_fraud]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x      = sub["amount"].tolist(),
            y      = sub["risk_score"].tolist() if "risk_score" in sub else sub["fraud_probability"].tolist(),
            mode   = "markers",
            name   = name,
            marker = dict(color=colour, size=8 if not is_fraud else 12,
                          symbol=sym, opacity=0.75,
                          line=dict(width=1, color="white") if is_fraud else {}),
        ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="white",
                  annotation_text="Fraud threshold", annotation_position="top right",
                  annotation_font_color=_C["muted"])
    fig.update_layout(**_layout(
        title       = "Amount vs Risk Score",
        xaxis_title = "Transaction Amount ($)",
        yaxis_title = "Risk Score",
        height      = 300,
        showlegend  = True,
        xaxis       = dict(type="log", gridcolor=_C["border"]),
    ))
    return fig
