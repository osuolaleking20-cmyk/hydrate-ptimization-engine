import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Thermodynamic Optimization Engine", layout="wide")

st.title("Thermodynamic Optimization Engine")
st.markdown("Predictive minimum methanol dosage for steady-state flow assurance.")

# 2. Load the trained Engine
@st.cache_resource
def load_model():
    return joblib.load('thermodynamic_engine.pkl')

model = load_model()

# 3. Sidebar Inputs (What the client sees)
st.sidebar.header("Pipeline Boundary Conditions")
target_p = st.sidebar.number_input("Target Plateau Pressure (psi)", value=115.0, step=1.0)
init_p = st.sidebar.number_input("Initial Pressure (psi)", value=150.0, step=1.0)
init_t = st.sidebar.number_input("Initial Temperature (°C)", value=30.0, step=0.5)
corr = st.sidebar.number_input("Corrosion Inhibitor (wt%)", value=0.05, step=0.01)
scale = st.sidebar.number_input("Scale Inhibitor (wt%)", value=0.0025, step=0.001)

# 4. Boundary Extrapolation Warning (Your engineering safeguard)
if init_p > 160 or init_p < 100 or init_t > 35 or init_t < 20:
    st.sidebar.warning("**Extrapolation Warning**\n\nInputs exceed physical laboratory boundaries. Output accuracy may be diminished.")

# 5. Optimization Engine (Runs when client clicks the button)
if st.sidebar.button("Run Optimization"):
    best_methanol = None
    
    # Grid Search: Test Methanol from 0 to 30 wt% in 0.5 increments
    for m in np.arange(0.0, 30.5, 0.5):
        # Check the steady-state prediction at t = 120 minutes
        test_df = pd.DataFrame({
            'time_min': [120],
            'Methanol_wt': [m],
            'Corrosion_wt': [corr],
            'Scale_wt': [scale],
            'Initial_Pressure_psi': [init_p],
            'Initial_Temperature_C': [init_t]
        })
        
        pred = model.predict(test_df)[0]
        pred_steady_p = pred[0]
        
        if pred_steady_p >= target_p:
            best_methanol = m
            break
            
    if best_methanol is None:
        st.error("Target pressure cannot be achieved within the physical limits of the laboratory data (Max Methanol tested: 30 wt%).")
    else:
        # Simulate the full time series for the optimum dosage found
        times = np.arange(0, 122, 2)
        sim_df = pd.DataFrame({
            'time_min': times,
            'Methanol_wt': best_methanol,
            'Corrosion_wt': corr,
            'Scale_wt': scale,
            'Initial_Pressure_psi': init_p,
            'Initial_Temperature_C': init_t
        })
        
        preds = model.predict(sim_df)
        sim_df['pred_P'] = preds[:, 0]
        sim_df['pred_T'] = preds[:, 1]
        
        # Identify the exact time it hits the plateau (within 1 psi of final P)
        final_p = sim_df['pred_P'].iloc[-1]
        plateau_mask = np.abs(sim_df['pred_P'] - final_p) <= 1.0
        plateau_time = sim_df.loc[plateau_mask, 'time_min'].iloc[0]
        final_t = sim_df.loc[sim_df['time_min'] == plateau_time, 'pred_T'].values[0]
        
        # 6. Display Metrics (The 3 metrics you promised)
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal Methanol Dosage", f"{best_methanol:.1f} wt%")
        col2.metric("Time to Plateau", f"{plateau_time} minutes")
        col3.metric("Final Temperature", f"{final_t:.1f} °C")
        
        # 7. Visualization (Dual-axis plotting)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Pressure Line
        color1 = 'tab:blue'
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Predicted Pressure (psi)', color=color1)
        ax1.plot(sim_df['time_min'], sim_df['pred_P'], color=color1, linewidth=2, label='Pressure Decay')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.axhline(target_p, color='black', linestyle='--', label=f'Target ({target_p} psi)')
        ax1.axvline(plateau_time, color='gray', linestyle=':', label='Plateau Reached')
        
        # Temperature Line
        ax2 = ax1.twinx()  
        color2 = 'tab:red'
        ax2.set_ylabel('Predicted Temperature (°C)', color=color2)  
        ax2.plot(sim_df['time_min'], sim_df['pred_T'], color=color2, linewidth=2, linestyle='-.', label='Temperature Cooling')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combine legends for clean UI
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        st.pyplot(fig)
        st.success("Optimization Complete! The pipeline reaches steady-state phase equilibrium successfully.")
