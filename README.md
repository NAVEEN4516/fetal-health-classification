# Multiclass Classification of Fetal Health using Cardiotocogram Data
### UE24CS352A — Machine Learning Mini-Project | PES University
**Team Members:** Naveen Prasad M & Shreya Hegde  

---

## 🏗️ Repository Architecture

To maintain high software engineering rigor and allow seamless evaluation by faculty, this repository is partitioned into two clean, self-contained modules:

```
fetal-health-classification/
├── academic_review/             # 🎓 FOR PROFESSORS & EVALUATION PANEL
│   ├── scripts/
│   │   ├── 01_eda.py                     # Feature distributions & SelectKBest rankings
│   │   ├── 02_preprocessing.py           # Deduplication, scaling, & stratified 70/30 split
│   │   ├── 03_benchmark_all_10_models.py # Unified benchmark evaluating all 10 models
│   │   ├── 04_member1_models.py          # Naveen's 5 models (XGBoost, RF, MLP, SVM RBF, LogReg)
│   │   └── 05_member2_models.py          # Shreya's 5 models (LightGBM, GB, DT, Linear SVM, KNN)
│   ├── results/                          # Benchmark tables, confusion matrices, and charts
│   └── deliverables/                     # 2-Page Report (writeup.pdf) & Slide Deck (Presentation.pptx)
│
├── interface_app/               # 💻 INTERACTIVE CLINICAL DECISION SUPPORT APP
│   ├── app.py                            # Streamlit web app with Top-Corner Help & 1-Click Presets
│   ├── train_models.py                   # Serializes LightGBM & XGBoost inference pipelines
│   └── models/                           # Serialized joblib pipelines
│
├── data/                        # 📊 BENCHMARK DATASET
│   ├── fetal_health.csv                  # 2,126 Cardiotocogram patient recordings
│   └── processed/                        # Preprocessed NumPy train/test splits
│
└── requirements.txt             # Project dependencies
```

---

## 🎓 1. For Faculty / Evaluation Panel: Running the ML Benchmarks

To inspect and reproduce the complete machine learning methodology and 10-model benchmark results:

```bash
# Step 1: Run Exploratory Data Analysis & Feature Importance
python academic_review/scripts/01_eda.py

# Step 2: Run Data Preprocessing & Stratified Splitting
python academic_review/scripts/02_preprocessing.py

# Step 3: Run the Master Benchmark of all 10 Classifiers
python academic_review/scripts/03_benchmark_all_10_models.py
```

### 🏆 10-Model Benchmark Leaderboard (634 Held-Out Test Records)

| Rank | Model Architecture | Implemented By | Category | Accuracy | Macro F1 | Key Strength |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | **LightGBM** | **Shreya Hegde** | Gradient Boost | **95.74%** | **92.45%** | Highest overall accuracy; leaf-wise tree growth |
| **2** | **XGBoost** | **Naveen Prasad M** | Gradient Boost | **95.27%** | **92.18%** | 1.00 precision on Pathological cases (0 false alarms) |
| **3** | **Gradient Boosting** | **Shreya Hegde** | Sequential Ensemble | **94.79%** | **90.81%** | Strong sequential residual error correction |
| **4** | **Random Forest** | **Naveen Prasad M** | Bagging Ensemble | **94.79%** | **90.52%** | High variance reduction via 100 decorrelated trees |
| **5** | **Decision Tree (CART)**| **Shreya Hegde** | Tree-based | **92.59%** | **87.81%** | High white-box decision split interpretability |
| **6** | **Multi-layer Perceptron**| **Naveen Prasad M** | Neural Network | **92.27%** | **85.63%** | Feedforward deep artificial neural network |
| **7** | **SVM (RBF Kernel)** | **Naveen Prasad M** | Kernel Method | **91.01%** | **82.09%** | Infinite-dimensional non-linear Hilbert mapping |
| **8** | **Linear SVM** | **Shreya Hegde** | Linear Max-Margin | **89.91%** | **81.24%** | Maximum margin linear hyperplane |
| **9** | **Logistic Regression**| **Naveen Prasad M** | Probabilistic Linear| **89.43%** | **80.15%** | Baseline cross-entropy linear classifier |
| **10**| **K-Nearest Neighbors** | **Shreya Hegde** | Instance-based | **88.80%** | **78.28%** | Euclidean distance instance queries |

---

## 🩺 2. For Live Demonstration: Launching the Clinical Interface

To launch the interactive clinical decision support web application:

```bash
streamlit run interface_app/app.py
```

### ✨ Clinical Application Features:
1. **Top-Right Corner Help Center (`st.popover`):** Educational reference explaining Normal vs Suspect vs Pathological, Baseline FHR standards (110–160 bpm) vs Tachycardia/Bradycardia, and deceleration types.
2. **1-Click Case Loaders:** Instant buttons for **Normal**, **Suspect**, and **Critical Pathological** cases for seamless demonstration.
3. **Dual Model Engine:** Toggle between **LightGBM (95.89%)** and **XGBoost (95.27%)** in real time.
4. **Clinical Action Guidance:** Reassuring vs Elevated Surveillance vs Critical Obstetrician Alert.
5. **Biomarker Anomaly Flagging:** Automatically detects which vital signs breach safe physiological limits.

---

## 📄 Formal Deliverables
* **2-Page Report (PDF):** [`academic_review/deliverables/writeup.pdf`](academic_review/deliverables/writeup.pdf)
* **Presentation Slides (PPTX):** [`academic_review/deliverables/Fetal_Health_Classification_Presentation.pptx`](academic_review/deliverables/Fetal_Health_Classification_Presentation.pptx)
