import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Robust path handling
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = REPO_ROOT / 'data' / 'fetal_health.csv'
RESULTS_DIR = SCRIPT_DIR.parent / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("  STEP 1: EXPLORATORY DATA ANALYSIS (EDA) & FEATURE RANKING")
print("="*70)

# 1. Load Data
df = pd.read_csv(DATA_PATH)
print(f"Dataset Shape: {df.shape}")
print(f"Missing Values: {df.isnull().sum().sum()}")
duplicate_count = df.duplicated().sum()
print(f"Duplicate records: {duplicate_count}")

# 2. Class Distribution
class_counts = df['fetal_health'].value_counts().sort_index()
print("\n--- Fetal Health Class Distribution ---")
for cls, count in class_counts.items():
    label = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}.get(int(cls), str(cls))
    print(f"  Class {int(cls)} ({label}): {count} ({count/len(df)*100:.2f}%)")

plt.figure(figsize=(7, 5), dpi=300)
palette = ['#1E3A8A', '#F59E0B', '#DC2626']
ax = sns.countplot(data=df, x='fetal_health', palette=palette)
plt.title('CTG Fetal Health Class Distribution', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Fetal Health (1: Normal, 2: Suspect, 3: Pathological)', fontweight='bold')
plt.ylabel('Patient Count', fontweight='bold')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
plt.tight_layout()
dist_path = RESULTS_DIR / 'class_distribution.png'
plt.savefig(dist_path)
plt.close()
print(f"Saved: {dist_path}")

# 3. Feature Correlations
plt.figure(figsize=(14, 11), dpi=300)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
corr_path = RESULTS_DIR / 'correlation_heatmap.png'
plt.savefig(corr_path)
plt.close()
print(f"Saved: {corr_path}")

# 4. ANOVA F-Score Feature Importance
X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X, y)
featureScores = pd.DataFrame({'Feature': X.columns, 'Score': fit.scores_})
featureScores = featureScores.sort_values(by='Score', ascending=False).reset_index(drop=True)

print("\n--- Top 10 Physiological Biomarkers (SelectKBest ANOVA) ---")
for idx, row in featureScores.head(10).iterrows():
    print(f"  {idx+1:2d}. {row['Feature']:<52} Score: {row['Score']:.2f}")

plt.figure(figsize=(10, 6), dpi=300)
sns.barplot(x='Score', y='Feature', data=featureScores.head(12), color='#1E3A8A')
plt.title('Top 12 Clinical Biomarkers Ranked by ANOVA F-Score', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('ANOVA F-Score', fontweight='bold')
plt.ylabel('CTG Feature', fontweight='bold')
plt.tight_layout()
feat_path = RESULTS_DIR / 'feature_importance.png'
plt.savefig(feat_path)
plt.close()
print(f"Saved: {feat_path}")
print("="*70)
