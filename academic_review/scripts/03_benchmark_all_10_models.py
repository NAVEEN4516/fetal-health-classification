from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PROCESSED_DIR = REPO_ROOT / 'data' / 'processed'
RESULTS_DIR = SCRIPT_DIR.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*90)
print("  STEP 3: RIGOROUS BENCHMARK EVALUATION OF ALL 10 MACHINE LEARNING MODELS")
print("="*90)

# Load preprocessed arrays
X_train = np.load(PROCESSED_DIR / 'X_train.npy')
X_test = np.load(PROCESSED_DIR / 'X_test.npy')
y_train = np.load(PROCESSED_DIR / 'y_train.npy')
y_test = np.load(PROCESSED_DIR / 'y_test.npy')

print(f"Dataset: 1,479 training instances | 634 held-out test instances | 21 features")

# Model definitions with author attribution
models = [
    {
        "name": "LightGBM",
        "author": "Shreya Hegde",
        "category": "Gradient Boost",
        "model": lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
    },
    {
        "name": "XGBoost",
        "author": "Naveen Prasad M",
        "category": "Gradient Boost",
        "model": XGBClassifier(random_state=42, eval_metric='mlogloss')
    },
    {
        "name": "Gradient Boosting",
        "author": "Shreya Hegde",
        "category": "Sequential Ensemble",
        "model": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    },
    {
        "name": "Random Forest",
        "author": "Naveen Prasad M",
        "category": "Bagging Ensemble",
        "model": RandomForestClassifier(n_estimators=100, random_state=42)
    },
    {
        "name": "Decision Tree (CART)",
        "author": "Shreya Hegde",
        "category": "Tree-based",
        "model": DecisionTreeClassifier(max_depth=10, random_state=42)
    },
    {
        "name": "Multi-layer Perceptron",
        "author": "Naveen Prasad M",
        "category": "Neural Network",
        "model": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    },
    {
        "name": "SVM (RBF Kernel)",
        "author": "Naveen Prasad M",
        "category": "Kernel Method",
        "model": SVC(kernel='rbf', random_state=42)
    },
    {
        "name": "Linear SVM",
        "author": "Shreya Hegde",
        "category": "Linear Max-Margin",
        "model": SVC(kernel='linear', C=1.0, random_state=42)
    },
    {
        "name": "Logistic Regression",
        "author": "Naveen Prasad M",
        "category": "Probabilistic Linear",
        "model": LogisticRegression(max_iter=1000, random_state=42)
    },
    {
        "name": "K-Nearest Neighbors",
        "author": "Shreya Hegde",
        "category": "Instance-based",
        "model": KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    }
]

benchmark_records = []
predictions_dict = {}

for m in models:
    name = m["name"]
    author = m["author"]
    category = m["category"]
    clf = m["model"]
    
    t0 = time.time()
    if name == "XGBoost":
        clf.fit(X_train, y_train - 1)
        y_pred = clf.predict(X_test) + 1
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    train_time = time.time() - t0
    
    acc = accuracy_score(y_test, y_pred) * 100
    f1_macro = f1_score(y_test, y_pred, average='macro') * 100
    f1_weighted = f1_score(y_test, y_pred, average='weighted') * 100
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec_macro = recall_score(y_test, y_pred, average='macro') * 100
    
    predictions_dict[name] = y_pred
    
    benchmark_records.append({
        "Model": name,
        "Implemented By": author,
        "Category": category,
        "Accuracy (%)": round(acc, 2),
        "Macro F1 (%)": round(f1_macro, 2),
        "Weighted F1 (%)": round(f1_weighted, 2),
        "Macro Recall (%)": round(rec_macro, 2),
        "Train Time (s)": round(train_time, 3)
    })

results_df = pd.DataFrame(benchmark_records).sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)
csv_out = RESULTS_DIR / "benchmark_10_models_summary.csv"
results_df.to_csv(csv_out, index=False)

print("\n" + results_df.to_string(index=False))
print(f"\nSaved benchmark metrics to: {csv_out}")

# ----------------------------------------------------
# 1. Comparison Bar Chart
# ----------------------------------------------------
plt.figure(figsize=(11, 6), dpi=300)
colors = ['#1E3A8A' if a == 'Naveen Prasad M' else '#0D9488' for a in results_df['Implemented By']]
bars = plt.barh(results_df['Model'][::-1], results_df['Accuracy (%)'][::-1], color=colors[::-1], height=0.65)
plt.xlim(85, 98)
plt.xlabel('Test Accuracy (%)', fontsize=11, fontweight='bold', labelpad=10)
plt.title('Rigorous Benchmark of 10 Machine Learning Classifiers', fontsize=13, fontweight='bold', pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.5)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.25, bar.get_y() + bar.get_height()/2, f'{w:.2f}%', 
             va='center', ha='left', fontsize=9.5, fontweight='bold', color='#1E293B')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1E3A8A', label='Naveen Prasad M'),
    Patch(facecolor='#0D9488', label='Shreya Hegde')
]
plt.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=9.5)
plt.tight_layout()
comp_chart_path = RESULTS_DIR / 'all_10_models_comparison.png'
plt.savefig(comp_chart_path)
plt.close()
print(f"Saved: {comp_chart_path}")

# ----------------------------------------------------
# 2. Confusion Matrices of Top Models (LightGBM vs XGBoost)
# ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
labels = ['Normal', 'Suspect', 'Pathological']

cm_lgb = confusion_matrix(y_test, predictions_dict['LightGBM'])
cm_xgb = confusion_matrix(y_test, predictions_dict['XGBoost'])

sns.heatmap(cm_lgb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=labels, yticklabels=labels, cbar=False)
axes[0].set_title('LightGBM (Accuracy: 95.89%)\n[Shreya Hegde]', fontsize=11, fontweight='bold', pad=10)
axes[0].set_xlabel('Predicted Label', fontweight='bold')
axes[0].set_ylabel('True Label', fontweight='bold')

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='GnBu', ax=axes[1],
            xticklabels=labels, yticklabels=labels, cbar=False)
axes[1].set_title('XGBoost (Accuracy: 95.27%)\n[Naveen Prasad M]', fontsize=11, fontweight='bold', pad=10)
axes[1].set_xlabel('Predicted Label', fontweight='bold')
axes[1].set_ylabel('True Label', fontweight='bold')

plt.suptitle('Top Gradient Boosted Architectures: Confusion Matrix Comparison', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
cm_chart_path = RESULTS_DIR / 'top_models_confusion_matrices.png'
plt.savefig(cm_chart_path, bbox_inches='tight')
plt.close()
print(f"Saved: {cm_chart_path}")

print("="*90)
