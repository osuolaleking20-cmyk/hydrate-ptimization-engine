from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "thermodynamic_engine.pkl"
FEATURES = [
    "time_min", "Methanol_wt", "Corrosion_wt",
    "Scale_wt", "Initial_Pressure_psi", "Initial_Temperature_C",
]
FINAL_TIME_MIN = 120          # end of the laboratory test window
METHANOL_GRID = np.arange(0.0, 30.5, 0.5)   # 0.0 -> 30.0 wt% in 0.5 steps
TIME_GRID = np.arange(0, FINAL_TIME_MIN + 2, 2)
PLATEAU_TOL_PSI = 1.0

st.set_page_config(page_title="Thermodynamic Optimization Engine", layout="wide")
st.title("Thermodynamic Optimization Engine")
st.markdown("Predictive optimum methanol dosage for steady-state flow assurance in the presence of CI and SI.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file not found: `{MODEL_PATH.name}`. "
             "Place it in the same folder as this script and reload.")
    st.stop()


def predict(df: pd.DataFrame) -> np.ndarray:
    """Predict [pressure, temperature]; column order is enforced to match training."""
    return np.asarray(model.predict(df[FEATURES]))


# ----------------------------------------------------------------------------
# Synced slider + number input
# ----------------------------------------------------------------------------
def synced_input(label, min_val, max_val, default_val, step_val, key_base, format_str=None):
    st.sidebar.markdown(f"**{label}**")

    slider_key = f"{key_base}_slider"
    num_key = f"{key_base}_num"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = default_val
    if num_key not in st.session_state:
        st.session_state[num_key] = default_val

    def sync_to_num():
        st.session_state[num_key] = st.session_state[slider_key]

    def sync_to_slider():
        st.session_state[slider_key] = st.session_state[num_key]

    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        st.slider(f"{label} (slider)", min_value=min_val, max_value=max_val, step=step_val,
                  key=slider_key, on_change=sync_to_num,
                  label_visibility="collapsed", format=format_str)
    with col2:
        st.number_input(f"{label} (value)", min_value=min_val, max_value=max_val, step=step_val,
                        key=num_key, on_change=sync_to_slider,
                        label_visibility="collapsed", format=format_str)

    return st.session_state[slider_key]


st.sidebar.header("Pipeline Boundary Conditions")

target_p = synced_input("Target Plateau Pressure (psi) [Max: 141]", 50.0, 141.0, 114.0, 1.0, "target_p")
init_p = synced_input("Initial Pressure (psi) [Lab: 150]", 100.0, 160.0, 150.0, 1.0, "init_p")
init_t = synced_input("Initial Temperature (°C) [Lab: 20-35]", 20.0, 35.0, 30.0, 0.5, "init_t")
corr = synced_input("Corrosion Inhibitor (wt%) [Lab: 0.00 - 0.10]", 0.00, 0.10, 0.05, 0.01, "corr")
scale = synced_input("Scale Inhibitor (wt%) [Lab: 0.0000 - 0.0045]", 0.0000, 0.0045, 0.0025, 0.0005,
                     "scale", format_str="%.4f")


# ----------------------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------------------
def build_frame(times, methanol):
    """Build a model-input frame; scalars are broadcast against array arguments."""
    n = max(np.size(times), np.size(methanol))
    return pd.DataFrame({
        "time_min": np.broadcast_to(times, n),
        "Methanol_wt": np.broadcast_to(methanol, n),
        "Corrosion_wt": corr,
        "Scale_wt": scale,
        "Initial_Pressure_psi": init_p,
        "Initial_Temperature_C": init_t,
    })


def find_minimum_methanol():
    """Single batched prediction over the whole methanol grid (instead of one call per step)."""
    final_pressure = predict(build_frame(FINAL_TIME_MIN, METHANOL_GRID))[:, 0]
    hits = np.flatnonzero(final_pressure >= target_p)
    return None if hits.size == 0 else float(METHANOL_GRID[hits[0]])


def plateau_metrics(sim_df):
    """Plateau = first time after which pressure stays within tolerance of the final value."""
    within = (np.abs(sim_df["pred_P"] - sim_df["pred_P"].iloc[-1]) <= PLATEAU_TOL_PSI).to_numpy()
    stays_within = np.logical_and.accumulate(within[::-1])[::-1]
    idx = int(np.argmax(stays_within))
    return sim_df["time_min"].iloc[idx], sim_df["pred_T"].iloc[idx]


if st.sidebar.button("Run Optimization", type="primary"):
    best_methanol = find_minimum_methanol()
    result = {
        "inputs": dict(target_p=target_p, init_p=init_p, init_t=init_t, corr=corr, scale=scale),
        "best_methanol": best_methanol,
    }
    if best_methanol is not None:
        sim_df = build_frame(TIME_GRID, best_methanol)
        preds = predict(sim_df)
        sim_df["pred_P"], sim_df["pred_T"] = preds[:, 0], preds[:, 1]
        result["sim_df"] = sim_df
        result["plateau_time"], result["final_t"] = plateau_metrics(sim_df)
    # Persist so results survive later widget interactions (Streamlit reruns the script)
    st.session_state["result"] = result

# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------
result = st.session_state.get("result")
if result:
    st.divider()
    col_in, col_out = st.columns(2)
    inp = result["inputs"]

    with col_in:
        st.subheader("Input Parameters")
        st.info(
            f"**Target Plateau Pressure:** {inp['target_p']} psi\n\n"
            f"**Initial Pressure:** {inp['init_p']} psi\n\n"
            f"**Initial Temperature:** {inp['init_t']} °C\n\n"
            f"**Corrosion Inhibitor:** {inp['corr']} wt%\n\n"
            f"**Scale Inhibitor:** {inp['scale']:.4f} wt%"
        )
        if (inp != dict(target_p=target_p, init_p=init_p, init_t=init_t, corr=corr, scale=scale)):
            st.caption("⚠️ Sidebar values changed since this run — click **Run Optimization** to update.")

    with col_out:
        st.subheader("Optimization Output")
        if result["best_methanol"] is None:
            st.error(
                f"❌ Target pressure cannot be achieved with up to {METHANOL_GRID[-1]:.0f} wt% "
                "methanol within the laboratory data range. Try a lower target pressure."
            )
        else:
            st.success(
                f" **Optimal Methanol Dosage:** {result['best_methanol']:.1f} wt%\n\n"
                f"⏱️ **Time to Plateau:** {result['plateau_time']} minutes\n\n"
                f"🌡️ **Final Temperature:** {result['final_t']:.1f} °C"
            )

    if result["best_methanol"] is not None:
        sim = result["sim_df"].set_index("time_min")
        c1, c2 = st.columns(2)
        c1.markdown("**Predicted pressure (psi) vs time (min)**")
        c1.line_chart(sim["pred_P"])
        c2.markdown("**Predicted temperature (°C) vs time (min)**")
        c2.line_chart(sim["pred_T"])
