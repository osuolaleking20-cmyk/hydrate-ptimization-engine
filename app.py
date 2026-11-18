from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "thermodynamic_engine (1).pkl"
FEATURES = [
    "time_min", "Methanol_wt", "Corrosion_wt",
    "Scale_wt", "Initial_Pressure_psi", "Initial_Temperature_C",
]

# Laboratory operating condition: every experiment started at 150 psi and 30 deg C,
# so the engine is calibrated at this condition only.
LAB_INIT_P = 150.0   # psi
LAB_INIT_T = 30.0    # deg C

# Reference for inhibition efficiency: the uninhibited laboratory run (no methanol, no
# corrosion or scale inhibitor) fell from 150 psi to 60 psi after 120 minutes.
UNINHIBITED_FINAL_P = 60.0   # psi

FINAL_TIME_MIN = 120
TIME_GRID = np.arange(0, FINAL_TIME_MIN + 2, 2)
PLATEAU_TOL_PSI = 1.0

# Methanol is searched only over the tested range (10-30 wt%).
METHANOL_MIN, METHANOL_MAX, METHANOL_STEP = 10.0, 30.0, 0.5
METHANOL_GRID = np.arange(METHANOL_MIN, METHANOL_MAX + METHANOL_STEP / 2, METHANOL_STEP)

# ----------------------------------------------------------------------------
# Lab-fitted equations (runs with corrosion inhibitor, 10-30 wt% methanol).
#   y = b0 + b_m*methanol + b_c*corrosion + b_s*scale + b_mc*methanol*corrosion
# Scale coefficient is negative: more scale inhibitor always lowers plateau pressure
# (about -1.36 psi per +0.001 wt%). Leave-one-run-out error: ~0.9 psi (pressure).
# ----------------------------------------------------------------------------
P_COEF = dict(b0=113.85196293921909, m=0.12807377049180496, c=-87.3633879781421,
              s=-1364.3699565487275, mc=1.2704918032786594)
T_COEF = dict(b0=8.335100131269654, m=-0.0273565573770492, c=0.37568306010928154,
              s=76.35009310986969, mc=0.0051229508196725115)


def equation(coef, methanol, corr, scale):
    return (coef["b0"] + coef["m"] * methanol + coef["c"] * corr
            + coef["s"] * scale + coef["mc"] * methanol * corr)


st.set_page_config(page_title="THI Optimum Dosage Engine", layout="wide")
st.title("THI Optimum Dosage Engine")
st.markdown("Predictive optimum methanol dosage for steady-state flow assurance.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model file not found: `{MODEL_PATH.name}`. "
             "Place it in the same folder as this script and reload.")
    st.stop()


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

target_p = synced_input("Target Plateau Pressure (psi)", 90.0, 125.0, 110.0, 0.5, "target_p")
op_p = synced_input("Operating Pressure (psi) [Lab: 150]", 100.0, 160.0, LAB_INIT_P, 1.0, "op_p")
op_t = synced_input("Operating Temperature (°C) [Lab: 30]", 20.0, 35.0, LAB_INIT_T, 0.5, "op_t")
corr = synced_input("Corrosion Inhibitor (wt%) [Lab: 0.01 - 0.10]", 0.01, 0.10, 0.05, 0.01, "corr")
scale = synced_input("Scale Inhibitor (wt%) [Lab: 0.0000 - 0.0045]", 0.0000, 0.0045, 0.0025, 0.0005,
                     "scale", format_str="%.4f")


# ----------------------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------------------
def find_optimum_methanol(target, corr_v, scale_v):
    """Lowest tested-range methanol dose whose plateau pressure meets the target."""
    plateau = equation(P_COEF, METHANOL_GRID, corr_v, scale_v)
    hits = np.flatnonzero(plateau >= target)
    best = None if hits.size == 0 else float(METHANOL_GRID[hits[0]])
    return best, float(plateau.min()), float(plateau.max())


def inhibition_efficiency(initial_p, final_p, uninhibited_final_p=UNINHIBITED_FINAL_P):
    """IE (%) = (1 - dP_inhibited / dP_uninhibited) * 100, with dP = initial - final pressure.
    Returns (IE limited to 0-100 %, True if the 100 % limit was applied)."""
    dp_inhibited = initial_p - final_p
    dp_uninhibited = initial_p - uninhibited_final_p
    if dp_uninhibited <= 0:
        return None, False
    ie = (1.0 - dp_inhibited / dp_uninhibited) * 100.0
    return float(np.clip(ie, 0.0, 100.0)), bool(ie > 100.0)


def anchored_curve(raw, start, end_target):
    """Shift a model curve so it ends at `end_target`, applying the shift gradually
    in proportion to how much of the curve's own drop has happened."""
    drop = start - raw[-1]
    if abs(drop) < 1e-6:
        w = np.linspace(0.0, 1.0, len(raw))
    else:
        w = np.clip((start - raw) / drop, 0.0, 1.0)
    return raw + w * (end_target - raw[-1])


def simulate(methanol, corr_v, scale_v):
    frame = pd.DataFrame({
        "time_min": TIME_GRID, "Methanol_wt": methanol, "Corrosion_wt": corr_v,
        "Scale_wt": scale_v, "Initial_Pressure_psi": LAB_INIT_P,
        "Initial_Temperature_C": LAB_INIT_T,
    })[FEATURES]
    raw = np.asarray(model.predict(frame))
    final_p = equation(P_COEF, methanol, corr_v, scale_v)
    final_t = equation(T_COEF, methanol, corr_v, scale_v)
    pressure = anchored_curve(raw[:, 0], LAB_INIT_P, final_p)
    temperature = anchored_curve(raw[:, 1], LAB_INIT_T, final_t)
    return pd.DataFrame({"time_min": TIME_GRID, "pred_P": pressure, "pred_T": temperature})


def plateau_time(sim_df):
    """First time after which pressure stays within tolerance of the final value."""
    within = (np.abs(sim_df["pred_P"] - sim_df["pred_P"].iloc[-1]) <= PLATEAU_TOL_PSI).to_numpy()
    stays_within = np.logical_and.accumulate(within[::-1])[::-1]
    return int(sim_df["time_min"].iloc[int(np.argmax(stays_within))])


if st.sidebar.button("Run Optimization", type="primary"):
    best, p_low, p_high = find_optimum_methanol(target_p, corr, scale)
    result = {
        "inputs": dict(target_p=target_p, op_p=op_p, op_t=op_t, corr=corr, scale=scale),
        "best_methanol": best,
        "p_low": p_low,
        "p_high": p_high,
    }
    if best is not None:
        sim_df = simulate(best, corr, scale)
        result["sim_df"] = sim_df
        result["plateau_time"] = plateau_time(sim_df)
        result["plateau_p"] = float(sim_df["pred_P"].iloc[-1])
        result["final_t"] = float(sim_df["pred_T"].iloc[-1])   # temperature at 120 min
        result["ie"], result["ie_limited"] = inhibition_efficiency(op_p, result["plateau_p"])
    st.session_state["result"] = result

# ----------------------------------------------------------------------------
# Sidebar footer: measured laboratory extremes (final pressure at 120 min)
# ----------------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.markdown("**Laboratory extremes (measured at 120 min)**")
st.sidebar.markdown(
    "**Best case:** the highest pressure retained was **141 psi**, with the maximum methanol "
    "(30 wt%) and **no** corrosion or scale inhibitor.\n\n"
    "**Worst case:** the lowest pressure with inhibitors was **101 psi**, with the maximum corrosion "
    "inhibitor (0.10 wt%) and maximum scale inhibitor (0.0045 wt%) at the **lowest** methanol "
    "tested (10 wt%). At 30 wt% methanol the same inhibitor mix gave 106 psi."
)
st.sidebar.caption(
    "Without any methanol or inhibitor the lab run collapsed to 60 psi (hydrate). "
    "The 141 psi case needs zero corrosion inhibitor, which is outside this app's range "
    "(corrosion starts at 0.01 wt%); within the app's range the highest measured pressure was 117 psi."
)

# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------
result = st.session_state.get("result")
if result:
    st.divider()
    col_in, col_out = st.columns(2)
    inp = result["inputs"]

    with col_in:
        st.subheader("📥 Input Parameters")
        st.info(
            f"**Target Plateau Pressure:** {inp['target_p']} psi\n\n"
            f"**Operating Pressure:** {inp['op_p']:.0f} psi\n\n"
            f"**Operating Temperature:** {inp['op_t']:.1f} °C\n\n"
            f"**Corrosion Inhibitor:** {inp['corr']} wt%\n\n"
            f"**Scale Inhibitor:** {inp['scale']:.4f} wt%"
        )
        if inp != dict(target_p=target_p, op_p=op_p, op_t=op_t, corr=corr, scale=scale):
            st.caption("⚠️ Sidebar values changed since this run — click **Run Optimization** to update.")

    with col_out:
        st.subheader("📤 Optimization Output")
        best = result["best_methanol"]
        if best is None:
            st.error(
                f"❌ Target pressure cannot be reached with up to {METHANOL_MAX:.0f} wt% methanol "
                f"for these inhibitor levels. Predicted plateau pressure for "
                f"{METHANOL_MIN:.0f}-{METHANOL_MAX:.0f} wt% methanol is "
                f"{result['p_low']:.1f}-{result['p_high']:.1f} psi. Try a lower target."
            )
        else:
            st.success(
                f"✅ **Optimal Methanol Dosage:** {best:.1f} wt%\n\n"
                f"📈 **Plateau Pressure:** {result['plateau_p']:.1f} psi\n\n"
                f"⏱️ **Time to Plateau:** {result['plateau_time']} minutes\n\n"
                f"🌡️ **Temperature at {FINAL_TIME_MIN} min:** {result['final_t']:.1f} °C\n\n"
                f"🛡️ **Inhibition Efficiency:** {result['ie']:.1f} %"
            )
            st.caption(
                "Inhibition Efficiency (IE) % = (1 − ΔP inhibited / ΔP uninhibited) × 100, where ΔP is "
                "the initial pressure minus the final pressure. The uninhibited reference is the "
                f"laboratory run with no methanol and no inhibitors ({UNINHIBITED_FINAL_P:.0f} psi after "
                f"{FINAL_TIME_MIN} minutes)."
            )
            if result["ie_limited"]:
                st.caption("Operating pressure is at or below the plateau pressure, so no pressure loss "
                           "is predicted and the efficiency is shown as 100 %.")
            if best <= METHANOL_MIN:
                st.warning(
                    f"The target is already met at the lowest tested dose ({METHANOL_MIN:.0f} wt%). "
                    "A lower dose may also work, but it was not tested in the laboratory."
                )

    if result["best_methanol"] is not None:
        sim = result["sim_df"].set_index("time_min")
        c1, c2 = st.columns(2)
        c1.markdown("**Predicted pressure (psi) vs time (min)**")
        c1.line_chart(sim["pred_P"])
        c2.markdown("**Predicted temperature (°C) vs time (min)**")
        c2.line_chart(sim["pred_T"])
