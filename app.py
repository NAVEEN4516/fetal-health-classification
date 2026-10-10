import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Fetal Health Prediction",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fetal_health_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    saved = load_model()
    model = saved["model"]
    feature_names = saved["feature_names"]
    class_names = saved["class_names"]
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.info("Run `python train_app_model.py` first.")
    st.stop()


st.title("🩺 Fetal Health Prediction")
st.write(
    "Machine learning demonstration using cardiotocogram (CTG) data."
)

st.warning(
    "**Educational Prototype Only:** This tool is not a medical "
    "diagnostic system. Predictions must not be used to make clinical "
    "decisions. Consult a qualified healthcare professional."
)

st.divider()
st.subheader("Enter CTG Measurements")
st.caption(
    "The initial values are illustrative examples, not recommended "
    "clinical thresholds. Replace them with your measurements."
)

defaults = {
    "baseline value": 140.0,
    "accelerations": 0.003,
    "fetal_movement": 0.0,
    "uterine_contractions": 0.005,
    "light_decelerations": 0.0,
    "severe_decelerations": 0.0,
    "prolongued_decelerations": 0.0,
    "abnormal_short_term_variability": 50.0,
    "mean_value_of_short_term_variability": 1.0,
    "percentage_of_time_with_abnormal_long_term_variability": 0.0,
    "mean_value_of_long_term_variability": 8.0,
    "histogram_width": 60.0,
    "histogram_min": 100.0,
    "histogram_max": 160.0,
    "histogram_number_of_peaks": 3.0,
    "histogram_number_of_zeroes": 0.0,
    "histogram_mode": 140.0,
    "histogram_mean": 135.0,
    "histogram_median": 140.0,
    "histogram_variance": 10.0,
    "histogram_tendency": 0.0,
}

with st.form("prediction_form"):
    inputs = {}

    columns = st.columns(3)

    for i, feature in enumerate(feature_names):
        with columns[i % 3]:
            inputs[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(defaults[feature]),
                format="%.5f",
                help=f"Dataset feature: {feature}"
            )

    submitted = st.form_submit_button(
        "🔍 Predict Fetal Health",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame([inputs], columns=feature_names)

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    model_classes = model.named_steps["classifier"].classes_

    st.divider()
    st.subheader("Prediction Result")

    result_name = class_names[prediction]

    if prediction == 1:
        st.success(f"Predicted class: {result_name} (Class 1)")
    elif prediction == 2:
        st.warning(f"Predicted class: {result_name} (Class 2)")
    else:
        st.error(f"Predicted class: {result_name} (Class 3)")

    st.write("**Model probability scores**")

    for label, probability in zip(model_classes, probabilities):
        st.write(
            f"{class_names[int(label)]}: {probability * 100:.2f}%"
        )
        st.progress(float(probability))

    st.caption(
        "These scores reflect the model's estimates, not clinical "
        "certainty or the probability of a medical outcome."
    )

    st.download_button(
        "Download Input Measurements (CSV)",
        data=input_df.to_csv(index=False),
        file_name="ctg_input_measurements.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Student project | LightGBM | Educational use only — not for "
    "medical diagnosis or treatment decisions."
)
