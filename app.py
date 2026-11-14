import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Thermodynamic Optimization Engine", layout="wide")
st.title("⚙️ Thermodynamic Optimization Engine")
st.markdown("Predictive minimum methanol dosage for steady-state flow assurance.")

@st.cache_resource
def load_model():
    return joblib.load('thermodynamic_engine.pkl')

model = load_model()
st.sidebar.header("Pipeline Boundary Conditions")

# 1. Custom UI Function: Synced Slider and Number Input side-by-side
def synced_input(label, min_val, max_val, default_val, step_val, key_base, format_str=None):
    st.sidebar.markdown(f"**{label}**")
    
    slider_key = f"{key_base}_slider"
    num_key = f"{key_base}_num"

    # Initialize the session state for both widgets
    if slider_key not in st.session_state:
        st.session_state[slider_key] = default_val
    if num_key not in st.session_state:
        st.session_state[num_key] = default_val

    # Callback functions to enforce two-way binding
    def sync_to_num():
        st.session_state[num_key] = st.session_state[slider_key]
        
    def sync_to_slider():
        st.session_state[slider_key] = st.session_state[num_key]

    # Render the widgets side-by-side
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        st.slider("slider", min_value=min_val, max_value=max_val, step=step_val, 
                  key=slider_key, on_change=sync_to_num, label_visibility="collapsed", format=format_str)
    with col2:
        st.number_input("num", min_value=min_val, max_value=max_val, step=step_val, 
                        key=num_key, on_change=sync_to_slider, label_visibility="collapsed", format=format_str)
        
    return st.session_state[slider_key]

# 2. Instantiating the synced inputs with bracketed laboratory limits
target_p = synced_input("Target Plateau Pressure (psi) [Max: 141]", 50.0, 141.0, 114.0, 1.0, "target_p")
init_p = synced_input("Initial Pressure (psi) [Lab: 150]", 100.0, 160.0, 150.0, 1.0, "init_p")
init_t = synced_input("Initial Temperature (°C) [Lab: 20-35]", 20.0, 35.0, 30.0, 0.5, "init_t")
corr = synced_input("Corrosion Inhibitor (wt%) [Lab: 0.00 - 0.10]", 0.00, 0.10, 0.05, 0.01, "corr")
scale = synced_input("Scale Inhibitor (wt%) [Lab: 0.0000 - 0.0045]", 0.0000, 0.0045, 0.0025, 0.0005, "scale", format_str="%.4f")

# 3. Optimization Engine
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
            
    # 4. Split-Screen Layout (Inputs Left | Outputs Right)
    st.divider()
    col_in, col_out = st.columns(2)
    
    with col_in:
        st.subheader("📥 Input Parameters")
        st.info(
            f"**Target Plateau Pressure:** {target_p} psi\n\n"
            f"**Initial Pressure:** {init_p} psi\n\n"
            f"**Initial Temperature:** {init_t} °C\n\n"
            f"**Corrosion Inhibitor:** {corr} wt%\n\n"
            f"**Scale Inhibitor:** {scale} wt%"
        )
        
    with col_out:
        st.subheader("📤 Optimization Output")
        if best_methanol is None:
            st.error("❌ Target pressure cannot be achieved within the physical limits of the laboratory data.")
        else:
            times = np.arange(0, 122, 2)
            sim_df = pd.DataFrame({
                'time_min': times, 'Methanol_wt': best_methanol, 'Corrosion_wt': corr,
                'Scale_wt': scale, 'Initial_Pressure_psi': init_p, 'Initial_Temperature_C': init_t
            })
            
            preds = model.predict(sim_df)
            sim_df['pred_P'], sim_df['pred_T'] = preds[:, 0], preds[:, 1]
            
            final_p = sim_df['pred_P'].iloc[-1]
            plateau_time = sim_df.loc[np.abs(sim_df['pred_P'] - final_p) <= 1.0, 'time_min'].iloc[0]
            final_t = sim_df.loc[sim_df['time_min'] == plateau_time, 'pred_T'].values[0]
            
            st.success(
                f"✅ **Optimal Methanol Dosage:** {best_methanol:.1f} wt%\n\n"
                f"⏱️ **Time to Plateau:** {plateau_time} minutes\n\n"
                f"🌡️ **Final Temperature:** {final_t:.1f} °C"
            )
