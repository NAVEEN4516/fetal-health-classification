# 🩺 Fetal Health Clinical Decision Support Interface
### Interactive Web Application & Patient Education Center
**Course:** UE24CS352A — Machine Learning | Mini-Project  
**Authors:** Naveen Prasad M & Shreya Hegde  

---

## 🌟 Overview
This directory contains the full interactive web application for the Fetal Health Classification project. It transforms theoretical machine learning models into an intuitive clinical decision support tool for obstetricians, pediatricians, and patients.

### ✨ Key Features:
1. **Top-Right Corner Help Guide (`st.popover`):** Comprehensive education covering:
   - Diagnostic categories: Normal (Class 1), Suspect (Class 2), Pathological (Class 3).
   - Baseline FHR standards (FIGO 110–160 bpm) vs Tachycardia (>160 bpm) and Bradycardia (<110 bpm).
   - Physiological biomarkers: Accelerations, Deceleration types (early, late, prolonged), and Short/Long-Term Variability.
2. **1-Click Clinical Case Loaders:** Instant loading of real patient test profiles:
   - 🟢 **Normal Reassuring Case**
   - 🟡 **Borderline Suspect Case**
   - 🔴 **Acute Distress / Pathological Emergency**
3. **Dual Model Engine:** Toggle between **LightGBM (Shreya Hegde — 95.89%)** and **XGBoost (Naveen Prasad M — 95.27%)**, or run a blended ensemble consensus.
4. **Clinical Triage & Action Protocols:** Color-coded diagnostic banners with direct obstetric action guidance (e.g. routine monitoring vs emergency intervention).
5. **Biomarker Anomaly Flagging:** Automatically highlights patient metrics that breach safe clinical thresholds.
6. **Patient Report CSV Export:** Download entered measurements and prediction outputs in one click.

---

## 🚀 How to Run the App

From the repository root:
```bash
streamlit run interface_app/app.py
```
Or from within the `interface_app/` folder:
```bash
cd interface_app
streamlit run app.py
```

---

## ⚙️ Model Training & Serialization
To retrain and update both the LightGBM and XGBoost serialized inference pipelines:
```bash
python interface_app/train_models.py
```
Models will be saved in `interface_app/models/`:
- `fetal_health_model.joblib` (LightGBM Pipeline)
- `xgboost_model.joblib` (XGBoost Pipeline)
