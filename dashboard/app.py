"""
AI Fraud Detection System — Streamlit Dashboard
================================================
A production-grade, dark-themed real-time monitoring interface.

Tabs:
  📡 Live Feed      — scrolling transaction timeline + fraud feed table
  📈 Analytics      — probability distribution, session summary, scatter
  🔍 Transaction    — risk gauge, SHAP waterfall, rule triggers per transaction
  🚨 Alerts         — session fraud alert log
  🧪 Manual Test    — submit a custom transaction for scoring

Run with:
    streamlit run dashboard/app.py
    python run_dashboard.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dashboard.components.simulator import DashboardSimulator
from dashboard.components.charts import (
    risk_gauge,
    live_transaction_timeline,
    fraud_trend_chart,
    probability_distribution_chart,
    shap_waterfall_chart,
    session_summary_chart,
    amount_vs_risk_scatter,
)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --bg:      #0F172A;
    --card:    #1E293B;
    --border:  #334155;
    --blue:    #4F46E5;
    --green:   #10B981;
    --red:     #EF4444;
    --amber:   #F59E0B;
    --text:    #E2E8F0;
    --muted:   #94A3B8;
}

/* ── Global ── */
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: var(--bg); color: var(--text); }
.stApp > header { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--card);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 { color: var(--text); }

/* ── Metric widgets ── */
[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"]  { color: var(--text);  font-weight: 700; }
[data-testid="stMetricLabel"]  { color: var(--muted); font-size: 0.82em; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] { color: var(--muted); border-radius: 6px 6px 0 0; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--blue);
    border-bottom: 2px solid var(--blue);
    font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--blue), #7C3AED);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.45rem 1.2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45);
}
.stButton > button[kind="secondary"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
[data-testid="stDataFrameResizable"] th {
    background: #0F172A !important;
    color: var(--muted) !important;
    font-weight: 500;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 0.6rem 0; }

/* ── Alert boxes ── */
.fraud-alert {
    background: rgba(239, 68, 68, 0.08);
    border-left: 4px solid var(--red);
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 5px 0;
    color: #FDA4AF;
    font-family: 'Courier New', monospace;
    font-size: 0.8em;
}

/* ── Status badges ── */
.badge-fraud { color: var(--red);   font-weight: 700; }
.badge-legit { color: var(--green); font-weight: 600; }

/* ── Branding ── */
#MainMenu, footer, .stDeployButton { visibility: hidden; }

/* ── Info/success/error boxes ── */
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "sim":      None,
        "running":  False,
        "sel_txn":  None,
        "api_url":  "http://localhost:8000",
        "tps":      1.0,
        "fraud_rt": 0.05,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_sim() -> DashboardSimulator:
    if st.session_state.sim is None:
        st.session_state.sim = DashboardSimulator(
            api_url    = st.session_state.api_url,
            tps        = st.session_state.tps,
            fraud_rate = st.session_state.fraud_rt,
        )
    return st.session_state.sim


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> int:
    """Render sidebar controls; returns selected batch size."""
    with st.sidebar:
        # Branding
        st.markdown("""
        <div style="text-align:center;padding:8px 0 20px">
            <div style="font-size:2.8em;line-height:1">🛡️</div>
            <h2 style="color:#E2E8F0;margin:6px 0 0;font-size:1.15em;font-weight:700">
                Fraud Shield AI
            </h2>
            <p style="color:#4F46E5;margin:4px 0 0;font-size:0.78em;font-weight:600;
                      letter-spacing:0.08em">
                v2.0 · PRODUCTION
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # Configuration
        st.subheader("⚙️ Configuration")
        api_url = st.text_input("API URL", value=st.session_state.api_url, key="api_url_input")
        if api_url != st.session_state.api_url:
            st.session_state.api_url = api_url
            st.session_state.sim = None   # force recreation

        tps = st.slider("Transactions / second", 0.1, 10.0,
                        st.session_state.tps, 0.1, format="%.1f")
        st.session_state.tps = tps

        fraud_rt = st.slider(
            "Simulated fraud rate", 0.01, 0.40,
            st.session_state.fraud_rt, 0.01,
            format="%.0f%%",
            help="Elevated from real 0.17% for demonstration purposes",
        )
        st.session_state.fraud_rt = fraud_rt

        batch_size = st.slider("Batch size (per refresh)", 1, 20, 5)
        st.divider()

        # Controls
        c1, c2 = st.columns(2)
        with c1:
            btn_label = "⏸ Pause" if st.session_state.running else "▶ Start"
            if st.button(btn_label, width='stretch'):
                st.session_state.running = not st.session_state.running
                # Recreate simulator when (re)starting with current settings
                if st.session_state.running:
                    st.session_state.sim = DashboardSimulator(
                        api_url    = st.session_state.api_url,
                        tps        = tps,
                        fraud_rate = fraud_rt,
                    )
        with c2:
            if st.button("🔄 Reset", width='stretch'):
                st.session_state.running = False
                st.session_state.sim     = None
                st.session_state.sel_txn = None
                st.rerun()

        st.divider()

        # Session stats
        sim   = _get_sim()
        stats = sim.get_stats()
        st.subheader("📊 Session Stats")

        st.metric("Total Processed",  f"{stats['total']:,}")
        st.metric("Fraud Detected",   f"{stats['fraud']:,}",
                  delta=f"{stats['fraud_rate']:.1f}%", delta_color="inverse")
        st.metric("Total Volume",     f"${stats['total_amount']:,.0f}")
        st.metric("API Mode",
                  "🟢 Live" if sim._api_alive else ("🟡 Starting" if sim._api_alive is None else "🟠 Offline"))

        st.divider()
        st.caption("XGBoost · SHAP · FastAPI · Streamlit\nBuilt for fintech portfolio")

    return batch_size


# ── Header ────────────────────────────────────────────────────────────────────

def _render_header():
    left, right = st.columns([3, 1])
    with left:
        st.markdown("""
        <h1 style="color:#E2E8F0;font-size:1.75em;margin-bottom:4px;font-weight:700">
            🛡️ AI Fraud Detection System
        </h1>
        <p style="color:#94A3B8;margin:0;font-size:0.92em">
            Real-time monitoring · XGBoost + SHAP · Hybrid ML + Rule-Based Scoring
        </p>
        """, unsafe_allow_html=True)
    with right:
        status  = "🟢 Running" if st.session_state.running else "🔴 Stopped"
        now_str = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"""
        <div style="text-align:right;margin-top:10px">
            <div style="font-weight:600;color:#{'10B981' if st.session_state.running else 'EF4444'}">
                {status}
            </div>
            <div style="color:#94A3B8;font-size:0.82em">{now_str}</div>
        </div>
        """, unsafe_allow_html=True)


# ── KPI row ───────────────────────────────────────────────────────────────────

def _render_kpis(stats: dict):
    cols = st.columns(5)
    kpis = [
        ("Total Transactions", f"{stats['total']:,}",          None,               "normal"),
        ("Fraud Detected",     f"{stats['fraud']:,}",          f"{stats['fraud_rate']:.2f}%", "inverse"),
        ("Legitimate",         f"{stats['legit']:,}",          None,               "normal"),
        ("Avg Fraud Prob",     f"{stats['avg_prob']:.4f}",     None,               "normal"),
        ("Avg Risk Score",     f"{stats['avg_risk']:.4f}",     None,               "normal"),
    ]
    for col, (label, val, delta, dc) in zip(cols, kpis):
        with col:
            if delta:
                st.metric(label, val, delta=delta, delta_color=dc)
            else:
                st.metric(label, val)


# ── Transaction feed table ────────────────────────────────────────────────────

def _render_feed_table(history: list):
    if not history:
        st.info("No transactions yet — click **▶ Start** in the sidebar to begin streaming.")
        return

    recent = list(reversed(history[-60:]))
    rows   = []
    for t in recent:
        tid = t.get("id", t.get("transaction_id", "N/A"))
        rows.append({
            "Time":           datetime.fromtimestamp(t.get("timestamp", time.time())).strftime("%H:%M:%S"),
            "Transaction ID": str(tid)[:22],
            "Amount":         f"${t.get('amount', 0):,.2f}",
            "Fraud Prob":     f"{t.get('fraud_probability', 0):.4f}",
            "Risk Score":     f"{t.get('risk_score', 0):.4f}",
            "Level":          t.get("risk_level", "—"),
            "Status":         "🚨 Fraud" if t.get("is_fraud") else "✅ Legit",
        })

    df = pd.DataFrame(rows)

    def _style_status(v):
        if "Fraud" in str(v):
            return "color:#EF4444;font-weight:700;background:rgba(239,68,68,0.08)"
        return "color:#10B981"

    def _style_level(v):
        cm = {"CRITICAL": "#EF4444", "HIGH": "#F97316", "MEDIUM": "#F59E0B", "LOW": "#10B981"}
        return f"color:{cm.get(str(v), '#94A3B8')};font-weight:600"

    styled = (
        df.style
        .map(_style_status, subset=["Status"])
        .map(_style_level,  subset=["Level"])
    )
    st.dataframe(styled, width='stretch', height=420, hide_index=True)


# ── Transaction detail (tab 3) ────────────────────────────────────────────────

def _render_transaction_detail(txn: Optional[dict]):
    if txn is None:
        st.info("Select a transaction from the dropdown to inspect it.")
        return

    is_fraud   = txn.get("is_fraud", False)
    risk_score = txn.get("risk_score", txn.get("fraud_probability", 0.0))
    risk_level = txn.get("risk_level", "LOW")

    if is_fraud:
        st.error(f"🚨 **Fraudulent Transaction** — {risk_level} Risk")
    else:
        st.success(f"✅ **Legitimate Transaction** — {risk_level} Risk")

    left, right = st.columns([1, 2])

    with left:
        st.plotly_chart(risk_gauge(risk_score, risk_level), width='stretch')

        st.markdown("**Transaction Details**")
        fields = {
            "ID":          txn.get("id", txn.get("transaction_id", "N/A")),
            "Amount":      f"${txn.get('amount', 0):,.2f}",
            "Model":       txn.get("model_used", "N/A"),
            "Latency":     f"{txn.get('processing_time_ms', 0):.1f} ms",
            "Confidence":  f"{txn.get('confidence', 0):.1%}",
        }
        for k, v in fields.items():
            st.markdown(f"**{k}:** {v}")

    with right:
        shap_data = txn.get("shap_explanation")
        if shap_data:
            st.plotly_chart(shap_waterfall_chart(shap_data), width='stretch')
        else:
            st.info(
                "SHAP explanation unavailable.\n\n"
                "This can happen in offline mode or if the model hasn't loaded yet."
            )

        reasons = txn.get("reasons", [])
        if reasons:
            st.markdown("**🔍 Rule Triggers:**")
            for r in reasons:
                st.markdown(f"<div class='fraud-alert'>⚠️ {r}</div>", unsafe_allow_html=True)
        else:
            st.markdown("*No rule triggers — detected by ML model only.*")


# ── Alerts tab ────────────────────────────────────────────────────────────────

def _render_alerts(alert_log: list):
    if not alert_log:
        st.info("No fraud alerts in this session.")
        return

    st.markdown(f"**{len(alert_log)} Alert(s) Fired This Session**")
    for line in reversed(alert_log[-25:]):
        st.markdown(f"<div class='fraud-alert'>{line}</div>", unsafe_allow_html=True)


# ── Manual test tab ───────────────────────────────────────────────────────────

def _render_manual_test(sim: DashboardSimulator):
    st.subheader("🧪 Manual Transaction Scorer")
    st.caption(
        "Submit a custom transaction directly for fraud scoring. "
        "Use the presets to simulate known fraud patterns."
    )

    c1, c2 = st.columns(2)
    with c1:
        amount   = st.number_input("Amount ($)", 0.01, 99_999.0, 250.0, 10.0)
        time_val = st.number_input("Time (seconds from window start)", 0, 172_800, 43_200, 3_600)
    with c2:
        preset = st.selectbox("Transaction Preset", [
            "Normal Purchase (~$250)",
            "High-Value Legitimate (~$2,500)",
            "Suspicious Pattern (fraud-like PCA)",
            "Critical Amount ($8,000)",
            "Card Testing ($0.50 micro-transaction)",
            "Nighttime + High Amount",
        ])

    # Preset PCA overrides
    _presets: dict[str, dict] = {
        "Normal Purchase (~$250)":            {"amount": 250, "V1": 1.19, "V3": 0.17},
        "High-Value Legitimate (~$2,500)":    {"amount": 2_500, "V1": 0.5, "V3": -0.2},
        "Suspicious Pattern (fraud-like PCA)": {
            "amount": 320, "V1": -7.2, "V3": -6.8, "V4": 5.1,
            "V14": -9.5, "V17": -4.2,
        },
        "Critical Amount ($8,000)":           {"amount": 8_000},
        "Card Testing ($0.50 micro-transaction)": {
            "amount": 0.50, "V1": -3.1, "V3": -4.0,
        },
        "Nighttime + High Amount":            {
            "amount": 1_500, "time": 3_600 * 2,  # 2 am
        },
    }

    if st.button("🔍 Analyse Transaction", width='stretch'):
        from streaming.transaction_stream import TransactionGenerator
        gen = TransactionGenerator(fraud_rate=0.0)
        txn = gen.generate_one()

        pset = _presets.get(preset, {})
        txn.amount = pset.get("amount", amount)
        txn.time   = pset.get("time",   time_val)
        for k, v in pset.items():
            if k.startswith("V"):
                setattr(txn, k, v)

        with st.spinner("Scoring transaction …"):
            result = sim.predict_single(txn)

        if result:
            st.session_state.sel_txn = result
            if result.get("is_fraud"):
                st.error(
                    f"🚨 Fraud Detected — "
                    f"Risk: **{result['risk_score']:.4f}** ({result['risk_level']})"
                )
            else:
                st.success(
                    f"✅ Legitimate — "
                    f"Risk: **{result['risk_score']:.4f}** ({result['risk_level']})"
                )
            st.info("Switch to the **🔍 Transaction** tab to see the full SHAP explanation.")
        else:
            st.warning("Could not score transaction. Is the API running?")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init_state()
    batch_size = _render_sidebar()
    _render_header()
    st.divider()

    sim = _get_sim()

    # Run batch if streaming is active
    if st.session_state.running:
        sim.stream.tps                    = st.session_state.tps
        sim.stream.generator.fraud_rate   = st.session_state.fraud_rt
        sim.run_batch(n=batch_size)

    stats = sim.get_stats()
    _render_kpis(stats)
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "📡 Live Feed",
        "📈 Analytics",
        "🔍 Transaction",
        "🚨 Alerts",
        "🧪 Manual Test",
    ])

    # ── Tab 1: Live Feed ──────────────────────────────────────────────────────
    with t1:
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.plotly_chart(
                live_transaction_timeline(sim.history),
                width='stretch',
            )
        with c_right:
            st.plotly_chart(
                fraud_trend_chart(sim.history),
                width='stretch',
            )
        st.subheader("Recent Transactions")
        _render_feed_table(sim.history)

    # ── Tab 2: Analytics ──────────────────────────────────────────────────────
    with t2:
        probs  = [t.get("fraud_probability", 0) for t in sim.history]
        frauds = [t.get("is_fraud", False)       for t in sim.history]

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                probability_distribution_chart(probs, frauds),
                width='stretch',
            )
        with c2:
            st.plotly_chart(
                session_summary_chart(stats["total"], stats["fraud"]),
                width='stretch',
            )

        st.plotly_chart(
            amount_vs_risk_scatter(sim.history),
            width='stretch',
        )

        # Fraud-only table
        fraud_txns = [t for t in sim.history if t.get("is_fraud")]
        if fraud_txns:
            st.subheader(f"🚨 Flagged Transactions — {len(fraud_txns)} detected")
            fraud_df = pd.DataFrame([{
                "ID":         t.get("id", "N/A"),
                "Amount":     f"${t.get('amount', 0):,.2f}",
                "Risk Score": f"{t.get('risk_score', 0):.4f}",
                "Level":      t.get("risk_level", "N/A"),
                "Triggers":   "; ".join(t.get("reasons", [])) or "ML model",
            } for t in reversed(fraud_txns[-30:])])
            st.dataframe(fraud_df, width='stretch', hide_index=True)

    # ── Tab 3: Transaction Detail ─────────────────────────────────────────────
    with t3:
        txn_to_show = st.session_state.sel_txn

        if sim.history:
            options = [
                f"{t.get('id', t.get('transaction_id', 'N/A'))[:18]}  "
                f"{'🚨' if t.get('is_fraud') else '✅'}  "
                f"${t.get('amount', 0):,.0f}  |  {t.get('risk_level', '?')}"
                for t in reversed(sim.history[-40:])
            ]
            idx = st.selectbox(
                "Select transaction to inspect",
                range(len(options)),
                format_func=lambda i: options[i],
            )
            txn_to_show = list(reversed(sim.history[-40:]))[idx]

        _render_transaction_detail(txn_to_show)

    # ── Tab 4: Alerts ─────────────────────────────────────────────────────────
    with t4:
        _render_alerts(sim.alert_log)

    # ── Tab 5: Manual Test ────────────────────────────────────────────────────
    with t5:
        _render_manual_test(sim)

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if st.session_state.running:
        interval = max(0.5, 1.0 / st.session_state.tps)
        time.sleep(min(interval, 2.5))
        st.rerun()


if __name__ == "__main__":
    main()
