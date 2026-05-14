import streamlit as st
import numpy as np
import cvxpy as cp
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

# --- UI Setup ---
st.set_page_config(layout="wide", page_title="TDIM Command Center")
st.title("🌊 TDIM Live Command Center")
st.markdown("Automated Ingestion: KMD Document Parsing & UNESCO-WRA IoT Telemetry")

# --- STRICT PHYSICAL CONSTANTS (Not Tunable) ---
TAU = 2  # 2-Day physical routing delay to Lower Tana
MAX_STORAGE = 1560  # Physical max capacity of Masinga (MCM)
SAFE_RIVER_FLOW = 45  # ~500 cumecs physical flood threshold at Garissa
MAX_DIVERSION = 40  # Physical limit of TARDA canals

# --- Sidebar Controls ---
with st.sidebar:
    st.header("📄 Data Ingestion")
    uploaded_file = st.file_uploader("Upload KMD Forecast Report (PDF)", type=['pdf'])

    st.markdown("---")
    st.header("⚙️ Operational Tuning Matrix")
    w_flood = st.slider("Priority: Prevent Downstream Floods", 100, 1000, 500)
    w_spill = st.slider("Penalty: Using Emergency Gates", 10, 300, 100)
    w_diversion = st.slider("Penalty: Using Irrigation Canals", 1, 100, 20)
    w_storage = st.slider("Priority: Maintain High Storage (Power)", 1, 100, 10)
    min_env_flow = st.number_input("Minimum Env. Flow (MCM/d)", value=10)

# --- 1. AI DOCUMENT PARSING & DATA INGESTION ---
st.markdown("#### 📡 Automated Data Feeds")

N = 30  # Default to Monthly Forecast
start_date = datetime(2026, 5, 1)

# A. KMD Document Extraction Logic
if uploaded_file is not None:
    # Simulate NLP extraction of the May 2026 KMD Report
    with st.spinner("NLP Engine parsing KMD document..."):
        time.sleep(1.5)  # Simulate processing time

    st.success(f"✔️ **KMD Document Parsed:** '{uploaded_file.name}'")
    st.info(
        '**AI Extraction Summary:** "Enhanced rainfall expected in the first week (MJO phase). Extending into the second week. Depressed rainfall in the second half of the month."')

    # Generate the 30-day mathematical curve based on the text extraction
    # High first 7 days, tapering 8-14, flatlining 15-30
    base_curve = np.concatenate([
        np.linspace(120, 160, 7),  # Week 1: Enhanced
        np.linspace(150, 60, 7),  # Week 2: Decreasing
        np.linspace(50, 15, 16)  # Week 3 & 4: Depressed
    ])
    inflow_forecast = np.clip(base_curve + np.random.normal(0, 5, N), 0, None)
else:
    st.warning("⚠️ No KMD Forecast Uploaded. Using historical baseline averages.")
    inflow_forecast = np.full(N, 40) + np.random.normal(0, 5, N)


# B. UNESCO IoT Telemetry
def fetch_unesco_telemetry(N_days):
    live_nodes = {"Garissa_01": 12.5, "Bura_02": 8.0, "Garsen_03": 14.2}
    total_lower_basin_runoff = sum(live_nodes.values())
    projected_runoff = [total_lower_basin_runoff * (0.95 ** i) for i in range(N_days)]
    return {"current_storage": 1480, "live_nodes": live_nodes, "local_runoff": np.array(projected_runoff)}


unesco_data = fetch_unesco_telemetry(N)

c1, c2 = st.columns(2)
c1.success(
    f"✔️ **UNESCO IoT:** 3 downstream nodes active ({sum(unesco_data['live_nodes'].values()):.1f} MCM/d local runoff).")
c2.info(f"⚙️ **KenGen Telemetry:** Masinga Load at {(unesco_data['current_storage'] / MAX_STORAGE) * 100:.1f}%.")
st.markdown("---")

# --- 2. RUN PREDICTIVE MODEL (MPC) ---
storage = cp.Variable(N + 1)
spillway = cp.Variable(N)
flood_gates = cp.Variable(N)
diversion = cp.Variable(N)
downstream = cp.Variable(N)

cost = 0
constraints = [storage[0] == unesco_data["current_storage"]]
current_release_buffer = [20] * TAU

for k in range(N):
    total_release = spillway[k] + flood_gates[k]
    constraints += [storage[k + 1] == storage[k] + inflow_forecast[k] - total_release]

    constraints += [storage[k + 1] >= 0, storage[k + 1] <= MAX_STORAGE]
    constraints += [spillway[k] >= min_env_flow, spillway[k] <= 60]
    constraints += [flood_gates[k] >= 0]
    constraints += [diversion[k] >= 0, diversion[k] <= MAX_DIVERSION]

    if k < TAU:
        constraints += [
            downstream[k] == unesco_data["local_runoff"][k] + current_release_buffer[-(TAU - k)] - diversion[k]]
    else:
        constraints += [
            downstream[k] == unesco_data["local_runoff"][k] + spillway[k - TAU] + flood_gates[k - TAU] - diversion[k]]

    cost += w_flood * cp.pos(downstream[k] - SAFE_RIVER_FLOW) ** 2
    cost += w_spill * flood_gates[k] ** 2
    cost += w_storage * (MAX_STORAGE * 0.95 - storage[k + 1]) ** 2
    cost += w_diversion * diversion[k] ** 2

problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()

# Ensure we have fallback values if math is pushed beyond physical limits
if problem.status not in ["optimal", "optimal_inaccurate"]:
    st.error(
        "🚨 CRITICAL: Storm volume exceeds absolute physical capacity of all infrastructure. Displaying fallback projection.")
    spillway_vals = np.full(N, 60)
    flood_gates_vals = np.full(N, 50)
    diversion_vals = np.full(N, MAX_DIVERSION)
    downstream_vals = np.full(N, 150)  # Massive flood
else:
    spillway_vals = spillway.value
    flood_gates_vals = flood_gates.value
    diversion_vals = diversion.value
    downstream_vals = downstream.value

# --- 3. DASHBOARD UI ---
dates = [start_date + timedelta(days=i) for i in range(N)]
df = pd.DataFrame({
    'Date': dates, 'Forecasted Inflow': inflow_forecast,
    'Suggested Spillway': spillway_vals, 'Suggested Flood Gates': flood_gates_vals,
    'Suggested Diversion': diversion_vals, 'Predicted Downstream Flow': downstream_vals
}).set_index('Date')

st.markdown("### 📅 30-Day Predictive Outlook")

# Plotly Charts
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df.index, y=df['Forecasted Inflow'], fill='tozeroy', mode='lines', name='KMD Forecast',
                          line=dict(color='blue')))
# Annotate the specific periods mentioned in the KMD report
fig1.add_vrect(x0=dates[0], x1=dates[7], fillcolor="blue", opacity=0.1, layer="below", line_width=0,
               annotation_text="Enhanced Rain")
fig1.add_vrect(x0=dates[14], x1=dates[29], fillcolor="orange", opacity=0.1, layer="below", line_width=0,
               annotation_text="Depressed Rain")
fig1.update_layout(title="1. Masinga Inflow Forecast (KMD Monthly)", yaxis_title="Flow (MCM/d)", hovermode="x unified",
                   margin=dict(l=0, r=0, t=40, b=0))

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(x=df.index, y=df['Predicted Downstream Flow'], fill='tozeroy', mode='lines', name='Predicted River Flow',
               line=dict(color='red')))
fig2.add_trace(go.Scatter(x=df.index, y=[SAFE_RIVER_FLOW] * N, mode='lines', name='Flood Limit',
                          line=dict(color='darkred', dash='dash')))
fig2.update_layout(title="2. Downstream Impact Projection", yaxis_title="Flow (MCM/d)", hovermode="x unified",
                   margin=dict(l=0, r=0, t=40, b=0))

col_chart1, col_chart2 = st.columns(2)
col_chart1.plotly_chart(fig1, use_container_width=True)
col_chart2.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --- 4. ACTION DISPATCH CENTER ---
st.markdown("### ⚡ AI Recommended Action Plan (Today)")
col_act1, col_act2, col_act3, col_act4 = st.columns(4)
col_act1.metric("Spillway Target", f"{df['Suggested Spillway'].iloc[0]:.1f} MCM/d", "KenGen Baseline")
fg_val = df['Suggested Flood Gates'].iloc[0]
col_act2.metric("Emergency Gates", f"{fg_val:.1f} MCM/d", "CRITICAL" if fg_val > 1 else "Closed",
                delta_color="inverse" if fg_val > 1 else "normal")
col_act3.metric("TARDA Diversion", f"{df['Suggested Diversion'].iloc[0]:.1f} MCM/d", "Irrigation Routing")
col_act4.metric("Garissa Projection", f"{df['Predicted Downstream Flow'].iloc[0]:.1f} MCM/d",
                f"{(SAFE_RIVER_FLOW - df['Predicted Downstream Flow'].iloc[0]):.1f} from limit")

# IoT Command Log
st.markdown("#### 📡 Automated API Dispatch Outbox")
trigger_log = st.container(border=True)
if max(df['Predicted Downstream Flow']) > SAFE_RIVER_FLOW:
    trigger_log.error(
        f"[ EWS TRIGGER ] Transmitting Cell-Broadcast to UNESCO Lower Tana Nodes. Projected breach on {df['Predicted Downstream Flow'].idxmax().strftime('%b %d')}.")
else:
    trigger_log.success(f"[ EWS STANDBY ] Downstream flow stabilized by MPC routing.")

if uploaded_file is not None:
    trigger_log.code(f'''
    // DISPATCH TO: TARDA Diversion Controller API
    POST https://api.tarda.go.ke/v1/actuators/diversion
    {{ "target_mcm_d": {df['Suggested Diversion'].iloc[0]:.1f}, "auth_token": "bearer_token" }}

    // DISPATCH TO: KenGen Control API
    POST https://api.kengen.co.ke/v1/masinga/gates
    {{ "spillway_mcm_d": {df['Suggested Spillway'].iloc[0]:.1f}, "flood_gate_mcm_d": {df['Suggested Flood Gates'].iloc[0]:.1f} }}
    ''', language='json')
else:
    trigger_log.warning("Awaiting KMD document upload to generate dispatch commands...")
