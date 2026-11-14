import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Thermodynamic Optimization Engine", layout="wide")
st.title("⚙️ Thermodynamic Optimization Engine")
st.markdown("Predictive minimum methanol dosage for steady-state flow assurance.")

@st.cache_resource
def load_model():
    return joblib.load('thermodynamic_engine.pkl')

model = load_model()
st.sidebar.header("Pipeline Boundary Conditions")

# Custom UI Function: Synced Slider and Number Input side-by-side
def synced_input(label, min_val, max_val, default_val, step, key, format_str=None):
    st.sidebar.markdown(f"**{label}**")
    
    if key not in st.session_state:
        st.session_state[key] = default_val

    def sync_slider(): st.session_state[key] = st.session_state[f"{key}_num"]
    def sync_num(): st.session_state[key] = st.session_state[f"{key}_slider"]

    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        st.slider("slider", min_val, max_val, st.session_state[key], step, key=f"{key}_slider", on_change=sync_num, label_visibility="collapsed", format=format_str)
    with col2:
        st.number_input("num", min_val, max_val, st.session_state[key], step, key=f"{key}_num", on_change=sync_slider, label_visibility="collapsed", format=format_str)
        
    return st.session_state[key]

# Instantiating the synced inputs with bracketed laboratory limits
target_p = synced_input("Target Plateau Pressure (psi) [Max: 141]", 50.0, 141.0, 114.0, 1.0, "target_p")
init_p = synced_input("Initial Pressure (psi) [Lab: 150]", 100.0, 160.0, 150.0, 1.0, "init_p")
init_t = synced_input("Initial Temperature (°C) [Lab: 20-35]", 20.0, 35.0, 30.0, 0.5, "init_t")
corr = synced_input("Corrosion Inhibitor (wt%) [Lab: 0.00 - 0.10]", 0.00, 0.10, 0.05, 0.01, "corr")
scale = synced_input("Scale Inhibitor (wt%) [Lab: 0.0000 - 0.0045]", 0.0000, 0.0045, 0.0025, 0.0005, "scale", format_str="%.4f")

# Optimization Engine
if st.sidebar.button("Run Optimization", type="primary"):
    best_methanol = None
    
    for m in np.arange(0.0, 30.5, 0.5):
        test_df = pd.DataFrame({
            'time_min': [120], 'Methanol_wt': [m], 'Corrosion_wt': [corr],
            'Scale_wt': [scale], 'Initial_Pressure_psi': [init_p], 'Initial_Temperature_C': [init_t]
        })
        
        if model.predict(test_df)[0][0] >= target_p:
            best_methanol = m
            break
            
    if best_methanol is None:
        st.error("❌ Target pressure cannot be achieved within the physical limits of the laboratory data.")
    else:
        times = np.arange(0, 122, 2)
        sim_df = pd.DataFrame({
            'time_min': times, 'Methanol_wt': best_methanol, 'Corrosion_wt': corr,
            'Scale_wt': [scale]*len(times), 'Initial_Pressure_psi': init_p, 'Initial_Temperature_C': init_t
        })
        
        preds = model.predict(sim_df)
        sim_df['pred_P'], sim_df['pred_T'] = preds[:, 0], preds[:, 1]
        
        final_p = sim_df['pred_P'].iloc[-1]
        plateau_time = sim_df.loc[np.abs(sim_df['pred_P'] - final_p) <= 1.0, 'time_min'].iloc[0]
        final_t = sim_df.loc[sim_df['time_min'] == plateau_time, 'pred_T'].values[0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal Methanol Dosage", f"{best_methanol:.1f} wt%")
        col2.metric("Time to Plateau", f"{plateau_time} minutes")
        col3.metric("Final Temperature", f"{final_t:.1f} °C")
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        color1 = 'tab:blue'
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Predicted Pressure (psi)', color=color1)
        ax1.plot(sim_df['time_min'], sim_df['pred_P'], color=color1, linewidth=2, label='Pressure Decay')
        ax1.axhline(target_p, color='black', linestyle='--', label=f'Target ({target_p} psi)')
        ax1.axvline(plateau_time, color='gray', linestyle=':', label='Plateau Reached')
        
        ax2 = ax1.twinx()  
        color2 = 'tab:red'
        ax2.set_ylabel('Predicted Temperature (°C)', color=color2)  
        ax2.plot(sim_df['time_min'], sim_df['pred_T'], color=color2, linewidth=2, linestyle='-.', label='Temperature Cooling')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        st.pyplot(fig)
        st.success("Optimization Complete! The pipeline reaches steady-state phase equilibrium successfully.")
