import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Fetal Health AI — Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Clinical Dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F2C59;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 12px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-suspect {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-pathological {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fetal_health_model.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model_obj = load_model()

if model_obj is None:
    st.error("Model file not found in `models/fetal_health_model.joblib`!")
    st.info("Run `python train_models.py` to serialise the model.")
    st.stop()

# -----------------------------------------------------------------------------
# PRESET CLINICAL CASES
# -----------------------------------------------------------------------------
PRESETS = {
    "Normal": {
        "baseline value": 133.0,
        "accelerations": 0.006,
        "fetal_movement": 0.0,
        "uterine_contractions": 0.006,
        "light_decelerations": 0.0,
        "severe_decelerations": 0.0,
        "prolongued_decelerations": 0.0,
        "abnormal_short_term_variability": 28.0,
        "mean_value_of_short_term_variability": 1.7,
        "percentage_of_time_with_abnormal_long_term_variability": 0.0,
        "mean_value_of_long_term_variability": 11.5,
        "histogram_width": 68.0,
        "histogram_min": 105.0,
        "histogram_max": 173.0,
        "histogram_number_of_peaks": 3.0,
        "histogram_number_of_zeroes": 0.0,
        "histogram_mode": 138.0,
        "histogram_mean": 137.0,
        "histogram_median": 138.0,
        "histogram_variance": 5.0,
        "histogram_tendency": 0.0,
    },
    "Suspect": {
        "baseline value": 148.0,
        "accelerations": 0.001,
        "fetal_movement": 0.0,
        "uterine_contractions": 0.003,
        "light_decelerations": 0.002,
        "severe_decelerations": 0.0,
        "prolongued_decelerations": 0.0,
        "abnormal_short_term_variability": 65.0,
        "mean_value_of_short_term_variability": 0.6,
        "percentage_of_time_with_abnormal_long_term_variability": 35.0,
        "mean_value_of_long_term_variability": 4.5,
        "histogram_width": 42.0,
        "histogram_min": 125.0,
        "histogram_max": 167.0,
        "histogram_number_of_peaks": 2.0,
        "histogram_number_of_zeroes": 0.0,
        "histogram_mode": 150.0,
        "histogram_mean": 147.0,
        "histogram_median": 149.0,
        "histogram_variance": 4.0,
        "histogram_tendency": 0.0,
    },
    "Pathological": {
        "baseline value": 130.0,
        "accelerations": 0.0,
        "fetal_movement": 0.0,
        "uterine_contractions": 0.001,
        "light_decelerations": 0.002,
        "severe_decelerations": 0.0,
        "prolongued_decelerations": 0.005,
        "abnormal_short_term_variability": 85.0,
        "mean_value_of_short_term_variability": 0.5,
        "percentage_of_time_with_abnormal_long_term_variability": 70.0,
        "mean_value_of_long_term_variability": 2.0,
        "histogram_width": 130.0,
        "histogram_min": 50.0,
        "histogram_max": 180.0,
        "histogram_number_of_peaks": 4.0,
        "histogram_number_of_zeroes": 0.0,
        "histogram_mode": 130.0,
        "histogram_mean": 120.0,
        "histogram_median": 125.0,
        "histogram_variance": 60.0,
        "histogram_tendency": 0.0,
    }
}

# Initialize session state with Normal preset
if "inputs" not in st.session_state:
    st.session_state.inputs = PRESETS["Normal"].copy()

# -----------------------------------------------------------------------------
# TOP HEADER BAR WITH HELP MENU BUTTON
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3.5, 1.5])

with col_h1:
    st.markdown('<div class="main-header">🩺 Fetal Health Clinical Decision Support</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Cardiotocography (CTG) Triage • UE24CS352A ML Mini-Project</div>', unsafe_allow_html=True)

with col_h2:
    st.write("")
    # Top-corner help popover
    with st.popover("❓ Patient & Clinical Knowledge Guide", use_container_width=True):
        st.markdown("### 📋 What Every Clinician & Patient Needs to Know")
        
        st.markdown("#### 1. Fetal Health Diagnostic Classifications")
        st.markdown("""
        * 🟢 **Normal (Class 1):** Reassuring fetal status. Normal physiological response to contractions. No signs of hypoxia.
        * 🟡 **Suspect (Class 2):** Equivocal tracing. Absence of acute acidemia, but requires close continuous monitoring and conservative intervention.
        * 🔴 **Pathological (Class 3):** High likelihood of fetal hypoxia/acidemia. Mandates urgent obstetrician bedside evaluation.
        """)
        
        st.markdown("#### 2. Baseline Heart Rate Standards (FIGO Guidelines)")
        st.markdown("""
        * **Normal Baseline FHR:** **110 – 160 bpm** (beats per minute).
        * **Baseline Tachycardia (>160 bpm):** Associated with maternal fever, early fetal hypoxemia, or intrauterine infection.
        * **Baseline Bradycardia (<110 bpm):** Prolonged drop indicating cord compression, acute hypoxemia, or conduction anomalies.
        """)
        
        st.markdown("#### 3. Key Physiological Biomarkers")
        st.markdown("""
        * **Accelerations:** Transitory increases in FHR (≥15 bpm for ≥15 sec). Reassuring sign of fetal movement and intact autonomic control.
        * **Decelerations:** Transitory drops in FHR:
          * *Early/Light Decelerations:* Benign fetal head compression.
          * *Severe Decelerations:* Cord compression.
          * *Prolonged Decelerations:* Deceleration lasting ≥ 2 minutes (Highest predictor of acute hypoxemia, ANOVA F-Score: 505.85).
        * **FHR Variability (STV & LTV):** Beat-to-beat fluctuations reflecting fetal nervous system health. Diminished variability (<5 bpm) signals acidemia.
        """)

st.caption("⚠️ **Educational & Research Prototype Only:** This system is trained on the Ayres-de-Campos et al. benchmark dataset. Not intended as an independent diagnostic medical device.")

# -----------------------------------------------------------------------------
# INPUT FEATURES FORM
# -----------------------------------------------------------------------------
st.subheader("📝 Enter Cardiotocogram (CTG) Patient Measurements")

feature_names = model_obj["feature_names"]

# Compact inline preset loader
st.caption("Load a reference profile — all input values update instantly:")
_pc1, _pc2, _pc3, _pc4 = st.columns([1.6, 1.6, 1.6, 5])
with _pc1:
    if st.button("🟢 Level 1 — Normal", use_container_width=True):
        st.session_state.inputs = PRESETS["Normal"].copy()
        st.rerun()
with _pc2:
    if st.button("🟡 Level 2 — Suspect", use_container_width=True):
        st.session_state.inputs = PRESETS["Suspect"].copy()
        st.rerun()
with _pc3:
    if st.button("🔴 Level 3 — Pathological", use_container_width=True):
        st.session_state.inputs = PRESETS["Pathological"].copy()
        st.rerun()

tab1, tab2, tab3 = st.tabs([
    "💓 1. Fetal Heart Rate & Decelerations",
    "📈 2. Uterine Activity & Variability",
    "📊 3. FHR Histogram Morphometrics"
])

def num_input(feature, label, help_text):
    val = float(st.session_state.inputs.get(feature, PRESETS["Normal"][feature]))
    st.session_state.inputs[feature] = st.number_input(
        label,
        value=val,
        format="%.4f" if val < 1 and val > 0 else "%.1f" if val >= 10 else "%.3f",
        help=help_text,
        key=f"in_{feature}"
    )

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        num_input("baseline value", "Baseline Heart Rate (bpm)", "Normal: 110 - 160 bpm")
        num_input("accelerations", "Accelerations (per sec)", "Reassuring indicator of fetal reactivity")
    with c2:
        num_input("light_decelerations", "Light Decelerations (per sec)", "Transitory mild decelerations")
        num_input("severe_decelerations", "Severe Decelerations (per sec)", "Sharp variable drops indicating cord compression")
    with c3:
        num_input("prolongued_decelerations", "Prolonged Decelerations (per sec)", "Deceleration > 2 min (Key biomarker of acute hypoxia)")

with tab2:
    c4, c5, c6 = st.columns(3)
    with c4:
        num_input("uterine_contractions", "Uterine Contractions (per sec)", "Contraction frequency per second")
        num_input("fetal_movement", "Fetal Movement (per sec)", "External detection of fetal kicks/motion")
    with c5:
        num_input("abnormal_short_term_variability", "Abnormal Short-Term Variability (%)", "Percentage of time with diminished STV")
        num_input("mean_value_of_short_term_variability", "Mean Short-Term Variability", "Normal healthy value > 1.0")
    with c6:
        num_input("percentage_of_time_with_abnormal_long_term_variability", "Abnormal Long-Term Variability (%)", "Duration of reduced cyclic variability")
        num_input("mean_value_of_long_term_variability", "Mean Long-Term Variability", "Normal healthy value > 8.0")

with tab3:
    c7, c8, c9 = st.columns(3)
    with c7:
        num_input("histogram_width", "Histogram Width", "Range between minimum and maximum FHR")
        num_input("histogram_min", "Histogram Minimum (bpm)", "Lowest recorded FHR")
        num_input("histogram_max", "Histogram Maximum (bpm)", "Highest recorded FHR")
    with c8:
        num_input("histogram_number_of_peaks", "Number of Histogram Peaks", "Bimodal or polymodal distribution count")
        num_input("histogram_number_of_zeroes", "Number of Zeroes", "Count of zero values in recording")
        num_input("histogram_mode", "Histogram Mode (bpm)", "Most frequent FHR")
    with c9:
        num_input("histogram_mean", "Histogram Mean (bpm)", "Average FHR")
        num_input("histogram_median", "Histogram Median (bpm)", "Median FHR")
        num_input("histogram_variance", "Histogram Variance", "Dispersion of FHR")
        num_input("histogram_tendency", "Histogram Tendency (-1, 0, 1)", "-1: Left skew, 0: Symmetric, 1: Right skew")

# Run Inference Button
st.write("")
predict_clicked = st.button("🔍 Run Clinical Diagnostic Triage", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# PREDICTION & CLINICAL TRIAGE LOGIC
# -----------------------------------------------------------------------------
if predict_clicked:
    input_df = pd.DataFrame([st.session_state.inputs], columns=feature_names)

    # Model inference — LightGBM (primary model)
    clf = model_obj["model"]
    pred = int(clf.predict(input_df)[0])
    probas = clf.predict_proba(input_df)[0]

    class_names = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    result_name = class_names[pred]

    st.divider()
    st.subheader("🩺 Diagnostic Triage & Clinical Decision Support")

    res_col1, res_col2 = st.columns([2, 1.2])

    with res_col1:
        if pred == 1:
            st.markdown(f"""
            <div style="background-color: #ECFDF5; border: 2px solid #10B981; border-radius: 8px; padding: 18px;">
                <span class="badge-normal">🟢 CLASS 1: NORMAL</span>
                <h3 style="color: #065F46; margin-top: 8px;">Reassuring Fetal Well-Being</h3>
                <p style="color: #047857; margin-bottom: 0;">
                    <b>Clinical Action:</b> Continue routine antenatal monitoring protocol. No evidence of intrapartum hypoxemia or acidosis. Normal beat-to-beat variability and reactivity.
                </p>
            </div>
            """, unsafe_allow_html=True)
        elif pred == 2:
            st.markdown(f"""
            <div style="background-color: #FFFBEB; border: 2px solid #F59E0B; border-radius: 8px; padding: 18px;">
                <span class="badge-suspect">🟡 CLASS 2: SUSPECT</span>
                <h3 style="color: #92400E; margin-top: 8px;">Equivocal / Borderline Status</h3>
                <p style="color: #B45309; margin-bottom: 0;">
                    <b>Clinical Action:</b> Elevated surveillance recommended. Initiate maternal repositioning (left lateral), administer maternal hydration, check for maternal fever, and repeat CTG examination in 30 minutes.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #FEF2F2; border: 2px solid #EF4444; border-radius: 8px; padding: 18px;">
                <span class="badge-pathological">🔴 CLASS 3: PATHOLOGICAL</span>
                <h3 style="color: #991B1B; margin-top: 8px;">CRITICAL ALERT: Acute Fetal Distress</h3>
                <p style="color: #B91C1C; margin-bottom: 0;">
                    <b>Clinical Action:</b> IMMEDIATE OBSTETRICIAN REVIEW REQUIRED. High risk of fetal hypoxemia/acidemia. Prepare for fetal scalp blood sampling (pH testing) or emergency operative delivery protocol.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Flagged Biomarkers
        st.write("")
        st.write("**⚠️ Physiological Biomarker Flags:**")
        anomalies = []
        inp = st.session_state.inputs
        if inp["baseline value"] > 160:
            anomalies.append(f"Baseline FHR elevated ({inp['baseline value']} bpm > 160 bpm) — Tachycardia")
        elif inp["baseline value"] < 110:
            anomalies.append(f"Baseline FHR depressed ({inp['baseline value']} bpm < 110 bpm) — Bradycardia")
        if inp["prolongued_decelerations"] > 0:
            anomalies.append(f"Prolonged decelerations detected ({inp['prolongued_decelerations']} / sec) — Acute distress indicator")
        if inp["abnormal_short_term_variability"] > 50:
            anomalies.append(f"High abnormal short-term variability ({inp['abnormal_short_term_variability']}%) — Autonomic compromise")
        if inp["percentage_of_time_with_abnormal_long_term_variability"] > 40:
            anomalies.append(f"High abnormal long-term variability ({inp['percentage_of_time_with_abnormal_long_term_variability']}%)")

        if anomalies:
            for anom in anomalies:
                st.markdown(f"- 🔴 **{anom}**")
        else:
            st.markdown("- 🟢 **All primary vital parameters within normal physiological baseline ranges.**")

    with res_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Model Probability Distribution")
        
        for idx, (c_num, c_lbl) in enumerate(class_names.items()):
            p_val = probas[idx] * 100
            st.write(f"**{c_lbl} (Class {c_num}):** {p_val:.2f}%")
            st.progress(float(probas[idx]))
            
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            label="📥 Download Clinical Report (CSV)",
            data=input_df.to_csv(index=False),
            file_name="ctg_clinical_patient_report.csv",
            mime="text/csv",
            use_container_width=True
        )

st.divider()
st.caption("UE24CS352A Machine Learning Mini-Project • Developed by Naveen Prasad M & Shreya Hegde • PES University")
