# Multiclass Classification of Fetal Health using Cardiotocogram Data

This project implements and benchmarks 10 machine learning models to classify fetal health from Cardiotocogram (CTG) data, along with an interactive Streamlit clinical decision support prototype.

## 👥 Team Members
* **Naveen Prasad M:** XGBoost, Random Forest, Multi-layer Perceptron (MLP), SVM (RBF), Logistic Regression. Preprocessing & pipeline architecture.
* **Shreya Hegde:** LightGBM, Gradient Boosting, Decision Tree, Linear SVM, K-Nearest Neighbors (KNN). Hyperparameter tuning & Streamlit interface.

---

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NAVEEN4516/fetal-health-classification.git
   cd fetal-health-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Execution Workflow

### 1. Exploratory Data Analysis & Modeling Scripts
Run the modules in sequence:
* `python notebooks/01_EDA.py` — Generates distribution, correlation heatmap, and SelectKBest feature rankings in `results/`.
* `python notebooks/02_preprocessing.py` — Prunes duplicates, standardizes features, and exports stratified 70/30 train/test sets to `data/processed/`.
* `python notebooks/03_models_member1.py` — Trains & benchmarks Naveen's 5 models (XGBoost, Random Forest, MLP, SVM RBF, Logistic Regression).
* `python notebooks/04_models_member2.py` — Trains & benchmarks Shreya's 5 models (LightGBM, Gradient Boosting, Decision Tree, Linear SVM, KNN) and GridSearchCV tuning.

### 2. Interactive Web Application (Streamlit Prototype)
To launch the interactive diagnostic web interface:
```bash
streamlit run app.py
```
*(Optional) To retrain and re-export the app's LightGBM inference pipeline:*
```bash
python train_app_model.py
```

---

## 📊 Deliverables & Resources
* **2-Page Academic Report:** [`writeup/writeup.pdf`](writeup/writeup.pdf)
* **16:9 Presentation Slide Deck:** [`slides/Fetal_Health_Classification_Presentation.pptx`](slides/Fetal_Health_Classification_Presentation.pptx)
* **Visual Benchmark Results:** [`results/`](results/)

## 📂 Dataset
The dataset `fetal_health.csv` was sourced from Ayres-de-Campos et al. via Kaggle. It contains 2,126 recordings across 21 continuous morphological/statistical features categorized into 3 classes:
* `1`: Normal
* `2`: Suspect
* `3`: Pathological
