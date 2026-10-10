# 🎓 Academic Review & Model Evaluation Guide
### Course: UE24CS352A — Machine Learning | Mini-Project
**Institution:** Department of Computer Science & Engineering, PES University, Bengaluru  
**Team Members:** Naveen Prasad M & Shreya Hegde  

---

## 📌 Executive Summary
This directory is curated specifically for faculty and evaluation panel review. It encapsulates the core machine learning pipeline, statistical preprocessing, and comparative benchmarking of **10 distinct classification architectures** on Cardiotocogram (CTG) fetal health data.

All user interface, framework plumbing, and web deployment code have been cleanly segregated into `../interface_app/` to ensure full academic focus on statistical rigor and model performance.

---

## 📁 Academic Directory Structure

```
academic_review/
├── scripts/
│   ├── 01_eda.py                       # Exploratory Data Analysis & SelectKBest rankings
│   ├── 02_preprocessing.py             # Deduplication, StandardScaler & Stratified 70/30 split
│   ├── 03_benchmark_all_10_models.py   # Unified benchmark comparing all 10 ML models
│   ├── 04_member1_models.py            # Naveen's 5 models (XGBoost, RF, MLP, SVM RBF, LogReg)
│   └── 05_member2_models.py            # Shreya's 5 models (LightGBM, GB, DT, Linear SVM, KNN)
├── results/
│   ├── benchmark_10_models_summary.csv # Exhaustive performance metric table
│   ├── all_10_models_comparison.png    # Leaderboard comparison chart
│   ├── top_models_confusion_matrices.png # Confusion matrix comparison
│   ├── class_distribution.png          # Class imbalance visualization
│   ├── correlation_heatmap.png         # Pearson correlation heatmap
│   └── feature_importance.png          # ANOVA F-score feature ranking
└── deliverables/
    ├── writeup.pdf                     # 2-Page Academic Project Report
    ├── writeup_fetal_health.docx       # Editable report document
    └── Fetal_Health_Classification_Presentation.pptx # 16:9 Presentation slide deck
```

---

## 🚀 Execution & Verification

To reproduce the experimental results from the repository root:

```bash
# 1. Exploratory Data Analysis & Feature Selection
python academic_review/scripts/01_eda.py

# 2. Data Cleansing & Preprocessing (Stratified Split)
python academic_review/scripts/02_preprocessing.py

# 3. Master Benchmark of All 10 Classifiers
python academic_review/scripts/03_benchmark_all_10_models.py
```

---

## 🏆 Benchmark Leaderboard (634 Held-Out Test Records)

| Rank | Model Architecture | Implemented By | Category | Test Accuracy | Macro F1 | Clinical Insight |
| :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | **LightGBM** | **Shreya Hegde** | Gradient Boost | **95.89%** | **93.24%** | Best overall; leaf-wise growth captures non-linear thresholds |
| **2** | **XGBoost** | **Naveen Prasad M** | Gradient Boost | **95.27%** | **91.89%** | Strong regularization; 1.00 precision on Pathological cases |
| **3** | **Gradient Boosting** | **Shreya Hegde** | Sequential Ensemble | **95.11%** | **91.90%** | Iterative residual correction on loss gradients |
| **4** | **Random Forest** | **Naveen Prasad M** | Bagging Ensemble | **94.79%** | **90.87%** | High variance reduction via decorrelated trees (0.97 F1 Normal) |
| **5** | **Decision Tree (CART)**| **Shreya Hegde** | Tree-based | **92.58%** | **88.24%** | High interpretability; sensitive to local data variance |
| **6** | **Multi-layer Perceptron**| **Naveen Prasad M** | Neural Network | **92.27%** | **86.12%** | Feedforward ANN with Adam optimizer |
| **7** | **SVM (RBF Kernel)** | **Naveen Prasad M** | Kernel Method | **91.01%** | **82.35%** | Infinite-dimensional kernel mapping |
| **8** | **Linear SVM** | **Shreya Hegde** | Linear Max-Margin | **89.90%** | **80.64%** | Linear hyperplanes struggle with overlapping boundaries |
| **9** | **Logistic Regression**| **Naveen Prasad M** | Probabilistic Linear| **89.43%** | **80.37%** | Baseline linear model; constrained by non-linearities |
| **10**| **K-Nearest Neighbors** | **Shreya Hegde** | Instance-based | **88.80%** | **78.29%** | Euclidean distance sensitive to high-dimensional density shifts |

---

## 📄 Key Deliverables
* **Report:** [`deliverables/writeup.pdf`](deliverables/writeup.pdf)
* **Slides:** [`deliverables/Fetal_Health_Classification_Presentation.pptx`](deliverables/Fetal_Health_Classification_Presentation.pptx)
