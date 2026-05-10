import streamlit as st
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

# --- UI Setup & Design Thinking Layout ---
st.set_page_config(layout="wide", page_title="Tana River EWS & Dam Control")
st.title("🌊 Lower Tana Early Warning & Dam Management System")
st.markdown("Integrated Model Predictive Control with automated downstream routing and EWS triggers.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🌤️ Weather Forecast")
    weather_condition = st.selectbox(
        "_Data From KMD & Sensors_",
        ["Very Sunny (Drought)", "Sunny (Normal)", "Rainy (Heavy Showers)", "Very Rainy (Storm Surge)"],
        index=3  # Default to the storm scenario to show the EWS in action
    )

    st.markdown("---")
    st.header("⚙️ System Parameters")
    N = st.slider("Prediction Horizon (days)", 5, 30, 14)
    tau = st.number_input("Routing Delay to Lower Tana (days)", min_value=1, max_value=5, value=2)

    st.markdown("---")
    st.header("📊 Capacity Limits")
    max_storage = st.number_input("Max Dam Storage (MCM)", value=1500)
    initial_storage = st.number_input("Current Storage (MCM)", value=1400)  # Closer to capacity
    safe_river_flow = st.number_input("Safe Downstream Flow (MCM/day)", value=60)

    st.markdown("---")
    st.header("🎛️ Controller Tuning")
    w_flood = st.slider("Penalty: Downstream Flooding", 1, 100, 90)
    w_storage = st.slider("Penalty: Deviating from Optimal Storage", 1, 100, 10)

# --- Hydrological Model (Forecast & Runoff) ---
np.random.seed(42)
days = np.arange(N)

# Dynamic Weather Logic
if weather_condition == "Very Sunny (Drought)":
    base_flow = 15
    peak_amplitude = 0  # Flatline, no storm peak
elif weather_condition == "Sunny (Normal)":
    base_flow = 30
    peak_amplitude = 10  # Slight natural variation
elif weather_condition == "Rainy (Heavy Showers)":
    base_flow = 45
    peak_amplitude = 50  # Noticeable swell in the river
else:  # Very Rainy (Storm Surge)
    base_flow = 60
    peak_amplitude = 90  # Massive flood peak

# Simulating inflow based on selected weather
inflow_forecast = np.sin(days / N * np.pi) * peak_amplitude + base_flow + np.random.normal(0, 5, N)
inflow_forecast = np.clip(inflow_forecast, 0, None)  # Prevent impossible negative flows

# Simulating local runoff happening downstream independent of the dam
local_runoff = np.random.normal(10, 3, N)
local_runoff = np.clip(local_runoff, 0, None)

# --- Improved MPC Optimization (With Routing Delays) ---
storage = cp.Variable(N + 1)
release = cp.Variable(N)
downstream_flow = cp.Variable(N)

cost = 0
constraints = [storage[0] == initial_storage]

# Assume a steady release for the days prior to the current horizon (for delay calculation)
previous_release = 20

for k in range(N):
    # 1. Dam Mass Balance
    constraints += [storage[k + 1] == storage[k] + inflow_forecast[k] - release[k]]
    constraints += [storage[k + 1] >= 0, storage[k + 1] <= max_storage]
    constraints += [release[k] >= 10]  # Minimum environmental flow

    # 2. Downstream Routing (Transport Delay)
    if k < tau:
        # Flow depends on water released BEFORE the current prediction horizon
        constraints += [downstream_flow[k] == local_runoff[k] + previous_release]
    else:
        # Flow depends on water released tau days ago
        constraints += [downstream_flow[k] == local_runoff[k] + release[k - tau]]

    # 3. Objective Function
    # Strictly penalize exceeding the safe downstream capacity
    cost += w_flood * cp.pos(downstream_flow[k] - safe_river_flow) ** 2
    # Keep storage high for power generation, but leave a small buffer
    target_storage = max_storage * 0.95
    cost += w_storage * (target_storage - storage[k + 1]) ** 2

problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()

# --- Early Warning System (EWS) Logic ---
if problem.status in ['optimal', 'optimal_inaccurate']:
    max_predicted_flow = np.max(downstream_flow.value)

    st.markdown("### 🚨 Early Warning System Status")

    # Determine EWS State and set map colors simultaneously
    if max_predicted_flow <= safe_river_flow * 0.8:
        status_text = "SAFE"
        zone_color = "green"
        st.success("**STATUS: SAFE.** Projected flow remains well within channel capacity. No action required.")
    elif max_predicted_flow <= safe_river_flow:
        status_text = "ALERT"
        zone_color = "orange"
        st.warning(
            "**STATUS: ALERT.** Projected flow is approaching capacity limits. \n* **Action:** Notify WRA field teams and standby local radio stations in Garissa.")
    else:
        status_text = "CRITICAL (EVACUATE)"
        zone_color = "red"
        st.error(
            f"**STATUS: CRITICAL (EVACUATE).** Projected peak flow ({max_predicted_flow:.1f} MCM/d) exceeds safe capacity ({safe_river_flow} MCM/d)!\n* **Action:** Trigger Cell Broadcast SMS and activate community sirens in Garsen and Lower Tana immediately.")

    # --- Top-Level Dashboard Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Dam Storage", f"{initial_storage} MCM", f"{initial_storage - max_storage} to overflow",
                delta_color="inverse")
    col2.metric("Peak Forecasted Inflow", f"{np.max(inflow_forecast):.1f} MCM/d")
    col3.metric("Peak Downstream Flow", f"{max_predicted_flow:.1f} MCM/d",
                f"{max_predicted_flow - safe_river_flow:.1f} over limit", delta_color="inverse")
    col4.metric("MPC Status", "OPTIMAL", "Controller Active")

    st.markdown("---")

    # --- Visualizations ---
    fig_col, map_col = st.columns([2, 1])

    with fig_col:
        st.markdown("#### Hydrograph")
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(days, inflow_forecast, label="Predicted Dam Inflow", linestyle='--', color='blue', alpha=0.6)
        ax1.plot(days, release.value, label="MPC Dam Release", color='green', linewidth=2)
        ax1.plot(days, downstream_flow.value, label="Delayed Downstream Flow", color='red', linewidth=2)
        ax1.axhline(safe_river_flow, color='darkred', linestyle=':', label="Safe River Capacity")

        ax1.set_ylabel("Flow Rate (MCM/day)")
        ax1.set_xlabel("Days into Future")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig)

    with map_col:
        st.markdown("#### Topographic Impact Zone Map")

        # 1. Topographic Map: Shows the actual river valley elevation
        m = folium.Map(location=[-1.3, 39.9], zoom_start=7, tiles="OpenTopoMap")

        # 2. Key Infrastructure: Define the actual vulnerable towns
        towns = {
            "Garissa": [-0.4532, 39.6461],
            "Bura": [-1.09, 39.96],
            "Hola": [-1.5, 40.03],
            "Garsen": [-2.2696, 40.1143]
        }

        # Add interactive pins for each town
        for town, coords in towns.items():
            folium.Marker(
                location=coords,
                tooltip=f"Check status for {town}",
                popup=f"<b>{town}</b><br>Overall Threat: {status_text}<br>Peak Projected Flow: {max_predicted_flow:.1f} MCM/d",
                icon=folium.Icon(color=zone_color, icon="info-sign")
            ).add_to(m)

        # 3. Dynamic River Corridor: The river swells based on the math!
        river_path = [towns["Garissa"], towns["Bura"], towns["Hola"], towns["Garsen"]]

        # Calculate visual thickness based on how far over the safe limit we are
        # If flow is heavily over safe limit, weight gets thick.
        dynamic_thickness = max(5, (max_predicted_flow / safe_river_flow) * 15)

        folium.PolyLine(
            locations=river_path,
            color=zone_color,
            weight=dynamic_thickness,
            opacity=0.6,
            tooltip="Tana River Main Channel"
        ).add_to(m)

        st_folium(m, height=400, width=400)

else:
    st.error("Optimization failed! The current state may be hydraulically unfeasible. Check constraints.")
