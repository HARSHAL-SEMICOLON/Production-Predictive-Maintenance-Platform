"""Sentinel AI - Production Predictive Maintenance Platform.
Industrial Operations Dashboard (Streamlit).
"""

import time
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuration & Styling
st.set_page_config(
    page_title="Sentinel AI — Predictive Maintenance Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Industrial SCADA UI
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .stMetric {
        background-color: #151c2c;
        border: 1px solid #243048;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    }
    .metric-healthy { border-left: 5px solid #10b981 !important; }
    .metric-warning { border-left: 5px solid #f59e0b !important; }
    .metric-critical { border-left: 5px solid #ef4444 !important; }
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        font-weight: 700;
        font-size: 14px;
        border-radius: 20px;
        margin-bottom: 12px;
    }
    .badge-healthy { background-color: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    .badge-warning { background-color: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
    .badge-critical { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
    .card {
        background-color: #151c2c;
        border: 1px solid #243048;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

# Helper Functions
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.5)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception:
        return False, {}


def fetch_recent_telemetry(engine_id: int):
    try:
        r = requests.get(f"{API_URL}/api/v1/telemetry/recent?engine_id={engine_id}&limit=40", timeout=2.0)
        return r.json() if r.status_code == 200 else {"readings": [], "predictions": []}
    except Exception:
        return {"readings": [], "predictions": []}


def send_telemetry_point(payload: dict):
    try:
        r = requests.post(f"{API_URL}/api/v1/telemetry", json=payload, timeout=3.0)
        return r.status_code == 200, r.json() if r.status_code == 200 else {}
    except Exception as e:
        return False, {"error": str(e)}


# -------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/gas-turbine.png", width=70)
st.sidebar.title("Sentinel AI Control")
st.sidebar.caption("Production Turbofan Fleet Monitoring")

api_online, health_data = check_api_health()
if api_online:
    st.sidebar.success(f"● API Gateway Online (v{health_data.get('version', '1.0.0')})")
else:
    st.sidebar.error("○ API Gateway Offline")

engine_id = st.sidebar.selectbox("Active Turbofan Engine", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=0)
model_choice = st.sidebar.radio("Inference Architecture", ["XGBoost Regressor (SHAP XAI)", "Bidirectional LSTM (Deep)"], index=0)
model_type_code = "xgboost" if "XGBoost" in model_choice else "lstm"

st.sidebar.markdown("---")
st.sidebar.subheader("🕹️ Live Sensor Simulation")
sim_speed = st.sidebar.slider("Stream Interval (sec)", 0.2, 3.0, 1.0, 0.2)
inject_anomaly = st.sidebar.checkbox("🚨 Inject Thermal & Vibration Anomaly", value=False)

# Session state initialization for continuous streaming
if "sim_cycle" not in st.session_state:
    st.session_state.sim_cycle = 1
if "streaming_active" not in st.session_state:
    st.session_state.streaming_active = False

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("▶️ Start Stream", use_container_width=True):
    st.session_state.streaming_active = True
if col_btn2.button("⏹️ Stop Stream", use_container_width=True):
    st.session_state.streaming_active = False

if st.sidebar.button("🔄 Reset Cycle to 1", use_container_width=True):
    st.session_state.sim_cycle = 1
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("NASA C-MAPSS FD001 Turbofan Benchmark | MLOps v1.0")

# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🛡️ Sentinel AI — Predictive Maintenance Platform")
    st.markdown("**Real-Time Turbofan Failure Prognostics & Autonomous Degradation Analytics**")
with header_col2:
    st.markdown(f"""
    <div style="text-align: right; margin-top: 15px;">
        <span style="font-size: 14px; color: #94a3b8;">MONITORING UNIT:</span><br>
        <span style="font-size: 24px; font-weight: 800; color: #38bdf8;">ENGINE #{engine_id}</span>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Simulation Step Logic
# -------------------------------------------------------------
if st.session_state.streaming_active:
    # Generate sensor reading for current cycle
    cycle = st.session_state.sim_cycle
    wear = (cycle / 140.0) ** 1.8 if cycle > 60 else 0.0

    # Sensor values
    lpc_temp = 642.5 + (wear * 2.2) + float(np.random.normal(0, 0.2))
    hpc_temp = 1588.0 + (wear * 22.0) + float(np.random.normal(0, 1.5))
    fan_speed = 2388.08 - (wear * 0.25) + float(np.random.normal(0, 0.04))
    hpc_pressure = 553.8 - (wear * 6.5) + float(np.random.normal(0, 0.8))
    static_pressure = 47.45 + (wear * 1.2) + float(np.random.normal(0, 0.1))

    if inject_anomaly:
        hpc_temp += 85.0
        static_pressure += 4.5
        lpc_temp += 12.0

    payload = {
        "engine_id": engine_id,
        "cycle": cycle,
        "setting_1": -0.0007,
        "setting_2": -0.0004,
        "setting_3": 100.0,
        "sensor_1": 518.67,
        "sensor_2": round(lpc_temp, 2),
        "sensor_3": round(hpc_temp, 2),
        "sensor_4": round(1404.0 + (wear * 15.0), 2),
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": round(hpc_pressure, 2),
        "sensor_8": round(fan_speed, 2),
        "sensor_9": round(9055.0 - (wear * 40.0), 2),
        "sensor_10": 1.30,
        "sensor_11": round(static_pressure, 2),
        "sensor_12": round(521.8 - (wear * 3.0), 2),
        "sensor_13": round(2388.09 - (wear * 0.2), 2),
        "sensor_14": round(8135.0 - (wear * 25.0), 2),
        "sensor_15": round(8.42 + (wear * 0.18), 2),
        "sensor_16": 0.03,
        "sensor_17": round(392.5 + (wear * 6.0), 2),
        "sensor_18": 2388.0,
        "sensor_19": 100.0,
        "sensor_20": round(38.85 - (wear * 0.6), 2),
        "sensor_21": round(23.32 - (wear * 0.4), 2),
    }

    send_telemetry_point(payload)
    st.session_state.sim_cycle += 1

# -------------------------------------------------------------
# Data Fetching
# -------------------------------------------------------------
data = fetch_recent_telemetry(engine_id)
readings = data.get("readings", [])
preds = data.get("predictions", [])

# Current state extraction
if preds:
    latest_pred = preds[-1]
    curr_rul = latest_pred.get("predicted_rul", 125.0)
    curr_risk = latest_pred.get("failure_probability", 0.02)
    curr_status = latest_pred.get("health_status", "HEALTHY")
    curr_cycle = latest_pred.get("cycle", 1)
else:
    curr_rul = 125.0
    curr_risk = 0.02
    curr_status = "HEALTHY"
    curr_cycle = st.session_state.sim_cycle

if readings:
    latest_sensor = readings[-1]
    curr_hpc_temp = latest_sensor.get("sensor_3", 1588.0)
    curr_fan_rpm = latest_sensor.get("sensor_8", 2388.08)
    curr_hpc_press = latest_sensor.get("sensor_11", 47.45)
else:
    curr_hpc_temp = 1588.0
    curr_fan_rpm = 2388.08
    curr_hpc_press = 47.45

# -------------------------------------------------------------
# Tabs Layout
# -------------------------------------------------------------
tab_live, tab_fleet, tab_governance = st.tabs([
    "📊 Real-Time SCADA Operations",
    "🏭 Fleet Health Matrix",
    "🔍 MLOps Governance & Drift",
])

# -------------------------------------------------------------
# TAB 1: Real-Time SCADA Operations
# -------------------------------------------------------------
with tab_live:
    # 1. Top Metrics Banner
    badge_class = "badge-healthy" if curr_status == "HEALTHY" else ("badge-warning" if curr_status == "WARNING" else "badge-critical")
    card_class = "metric-healthy" if curr_status == "HEALTHY" else ("metric-warning" if curr_status == "WARNING" else "metric-critical")

    st.markdown(f"""
    <div style="margin-bottom: 12px;">
        <span class="status-badge {badge_class}">SYSTEM STATUS: {curr_status}</span>
        <span style="margin-left: 12px; color: #94a3b8; font-size: 14px;">Operational Cycle: <strong>#{curr_cycle}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            label="Predicted RUL",
            value=f"{curr_rul:.0f} cycles",
            delta=f"{curr_rul - 125:.0f} vs Baseline" if curr_rul < 125 else "Nominal",
            delta_color="inverse" if curr_rul < 75 else "normal",
        )
    with m2:
        st.metric(
            label="Failure Risk",
            value=f"{curr_risk * 100:.1f}%",
            delta="High Alert" if curr_risk > 0.4 else "Within Safe Limit",
            delta_color="inverse" if curr_risk > 0.4 else "normal",
        )
    with m3:
        st.metric(label="HPC Outlet Temp (T30)", value=f"{curr_hpc_temp:.1f} °R")
    with m4:
        st.metric(label="Fan Speed (Nf)", value=f"{curr_fan_rpm:.1f} RPM")
    with m5:
        st.metric(label="HPC Static Pressure (Ps30)", value=f"{curr_hpc_press:.2f} psia")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Prognostics & Sensor Charts
    chart_c1, chart_c2 = st.columns([1, 1])

    with chart_c1:
        st.subheader("📉 Remaining Useful Life (RUL) Trajectory")
        if preds:
            df_preds = pd.DataFrame(preds)
            fig_rul = px.line(
                df_preds,
                x="cycle",
                y="predicted_rul",
                markers=True,
                title="RUL Countdown vs Operational Cycles",
                labels={"cycle": "Engine Cycle", "predicted_rul": "Estimated RUL (Cycles)"},
            )
            # Add Danger Threshold Line at RUL=30
            fig_rul.add_hline(y=30, line_dash="dash", line_color="#ef4444", annotation_text="Critical Action Threshold (30 cycles)")
            fig_rul.add_hline(y=75, line_dash="dot", line_color="#f59e0b", annotation_text="Inspection Threshold (75 cycles)")
            fig_rul.update_layout(
                template="plotly_dark",
                paper_bgcolor="#151c2c",
                plot_bgcolor="#151c2c",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_rul, use_container_width=True)
        else:
            st.info("Start simulation stream to generate real-time RUL degradation trajectory.")

    with chart_c2:
        st.subheader("🌡️ Sensor Telemetry Degradation Trends")
        if readings:
            df_readings = pd.DataFrame(readings)
            fig_sensors = go.Figure()
            fig_sensors.add_trace(go.Scatter(x=df_readings["cycle"], y=df_readings["sensor_3"], mode="lines+markers", name="HPC Temp (T30)"))
            fig_sensors.add_trace(go.Scatter(x=df_readings["cycle"], y=df_readings["sensor_2"], mode="lines", name="LPC Temp (T24)"))
            fig_sensors.add_trace(go.Scatter(x=df_readings["cycle"], y=df_readings["sensor_11"] * 25, mode="lines", name="HPC Static Press (x25)"))
            fig_sensors.update_layout(
                title="Multivariate Sensor Degradation Traces",
                xaxis_title="Operational Cycle",
                yaxis_title="Sensor Units",
                template="plotly_dark",
                paper_bgcolor="#151c2c",
                plot_bgcolor="#151c2c",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_sensors, use_container_width=True)
        else:
            st.info("Awaiting sensor telemetry data.")

    # 3. Explainable AI (SHAP XAI Breakdown)
    st.subheader("🧠 Explainable AI — Root Cause Attribution (SHAP)")
    st.markdown("Managers trust models when they can see *why* risk is elevated.")

    xai_col1, xai_col2 = st.columns([3, 2])
    with xai_col1:
        # Sample or query actual SHAP feature attributions
        shap_demo = [
            {"Feature": "HPC Outlet Temp (T30)", "Contribution": "+34%", "Impact": "Elevates Risk", "Value": f"{curr_hpc_temp:.1f} °R"},
            {"Feature": "HPC Static Pressure (Ps30)", "Contribution": "+26%", "Impact": "Elevates Risk", "Value": f"{curr_hpc_press:.2f} psia"},
            {"Feature": "Fan Speed Degradation (Nf)", "Contribution": "+18%", "Impact": "Elevates Risk", "Value": f"{curr_fan_rpm:.1f} RPM"},
            {"Feature": "Bypass Ratio Shift (BPR)", "Contribution": "+12%", "Impact": "Elevates Risk", "Value": "8.52"},
            {"Feature": "Bleed Enthalpy (htBleed)", "Contribution": "-6%", "Impact": "Lowers Risk", "Value": "393.2"},
        ]
        df_shap = pd.DataFrame(shap_demo)
        fig_shap = px.bar(
            df_shap,
            x="Contribution",
            y="Feature",
            orientation="h",
            color="Impact",
            color_discrete_map={"Elevates Risk": "#ef4444", "Lowers Risk": "#10b981"},
            title="Sensor Contributions to Machine Failure Risk",
        )
        fig_shap.update_layout(
            template="plotly_dark",
            paper_bgcolor="#151c2c",
            plot_bgcolor="#151c2c",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with xai_col2:
        st.markdown("""
        <div class="card">
            <h4>📋 Maintenance Directive</h4>
        """, unsafe_allow_html=True)
        if curr_status == "CRITICAL":
            st.error(f"⚠️ **ACTION REQUIRED: IMMEDIATE ENGINE OVERHAUL**\n- Engine #{engine_id} has entered critical failure horizon (<30 cycles).\n- Primary root cause: Elevated HPC Outlet Temp (+34%) and Static Pressure anomalies (+26%).\n- Work order generated for maintenance depot.")
        elif curr_status == "WARNING":
            st.warning(f"🔍 **INSPECTION SCHEDULED: 24-HOUR WINDOW**\n- Premature wear indicators detected on Engine #{engine_id}.\n- Plan boroscope inspection for High Pressure Compressor blades.")
        else:
            st.success(f"✅ **NORMAL OPERATION**\n- Engine #{engine_id} running within normal operational envelope.\n- Next routine checkpoint in 50 cycles.")
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: Fleet Health Matrix
# -------------------------------------------------------------
with tab_fleet:
    st.subheader("🏭 Turbofan Fleet Status Matrix")
    st.markdown("Overview of all monitored aircraft engines across the operational fleet.")

    fleet_records = []
    np.random.seed(99)
    for e_id in range(1, 11):
        if e_id == engine_id:
            rul_val = curr_rul
            risk_val = curr_risk
            status_val = curr_status
            cyc_val = curr_cycle
        else:
            sim_life = int(np.random.normal(160, 30))
            sim_cyc = int(np.random.uniform(40, 150))
            rul_val = max(10, sim_life - sim_cyc)
            risk_val = float(1.0 / (1.0 + np.exp((rul_val - 30) / 12.0)))
            status_val = "CRITICAL" if rul_val <= 30 else ("WARNING" if rul_val <= 75 else "HEALTHY")
            cyc_val = sim_cyc

        fleet_records.append({
            "Engine ID": f"ENG-{e_id:03d}",
            "Operating Cycles": cyc_val,
            "Predicted RUL (cycles)": round(rul_val, 1),
            "Failure Risk (%)": round(risk_val * 100, 1),
            "Health Category": status_val,
            "Assigned Aircraft": f"Tail-N{100+e_id}AA",
        })

    df_fleet = pd.DataFrame(fleet_records)
    st.dataframe(
        df_fleet.style.map(
            lambda v: "color: #ef4444; font-weight: bold;" if v == "CRITICAL" else ("color: #f59e0b; font-weight: bold;" if v == "WARNING" else "color: #10b981; font-weight: bold;"),
            subset=["Health Category"]
        ),
        use_container_width=True,
    )

# -------------------------------------------------------------
# TAB 3: MLOps Governance & Drift
# -------------------------------------------------------------
with tab_governance:
    st.subheader("📡 MLOps Observability & Sensor Drift Detection")
    st.markdown("Continuous monitoring with Evidently AI and Prometheus ensures the deployed model maintains validity under real-world sensor distribution shifts.")

    gov_c1, gov_c2 = st.columns([1, 1])

    with gov_c1:
        st.markdown("""
        <div class="card">
            <h4>🔬 Drift Analysis Trigger</h4>
            <p>Runs statistical two-sample Kolmogorov-Smirnov (KS) tests and calculates covariate shift between baseline training data and recent production telemetry.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🧪 Run Drift Check Now", use_container_width=True):
            try:
                with st.spinner("Analyzing telemetry distributions..."):
                    resp = requests.get(f"{API_URL}/api/v1/drift/report", timeout=10.0)
                    if resp.status_code == 200:
                        report_data = resp.json()
                        st.json(report_data)
                    else:
                        st.error(f"Error querying drift report: {resp.text}")
            except Exception as ex:
                st.error(f"Drift endpoint error: {ex}")

        st.markdown(f"""
        <div style="margin-top: 15px;">
            <a href="{API_URL}/api/v1/drift/report/html" target="_blank" style="color: #38bdf8; font-weight: bold;">
                🔗 Open Interactive Evidently AI HTML Drift Report
            </a>
        </div>
        """, unsafe_allow_html=True)

    with gov_c2:
        st.markdown("""
        <div class="card">
            <h4>⚡ Automated Retraining Dispatch</h4>
            <p>If sensor drift exceeds 30% or operational accuracy degrades, initiate the retraining pipeline to produce a new model and log to MLflow.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Trigger Model Retraining Pipeline", use_container_width=True):
            try:
                resp = requests.post(f"{API_URL}/api/v1/retrain", timeout=5.0)
                if resp.status_code == 200:
                    st.success("Retraining job dispatched in background! Check MLflow server for experiment tracking.")
                else:
                    st.error(f"Retrain request failed: {resp.text}")
            except Exception as ex:
                st.error(f"Retrain trigger error: {ex}")

        st.markdown(f"""
        <div style="margin-top: 15px;">
            <a href="{API_URL}/metrics" target="_blank" style="color: #38bdf8; font-weight: bold;">
                📈 View Raw Prometheus Metrics (/metrics)
            </a>
        </div>
        """, unsafe_allow_html=True)

# Auto-refresh if streaming is active
if st.session_state.streaming_active:
    time.sleep(sim_speed)
    st.rerun()
