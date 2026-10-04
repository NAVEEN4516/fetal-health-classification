import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
df = pd.read_csv('data/fetal_health.csv')
print("--- Data Shape ---")
print(df.shape)

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

# 2. Check for Nulls and Duplicates
print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Duplicates ---")
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# 3. Class Distribution
print("\n--- Class Distribution ---")
class_counts = df['fetal_health'].value_counts()
print(class_counts)

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='fetal_health')
plt.title('Distribution of Fetal Health Classes')
plt.xlabel('Fetal Health (1: Normal, 2: Suspect, 3: Pathological)')
plt.ylabel('Count')
plt.savefig('results/class_distribution.png')
plt.close()

# 4. Feature Correlations
plt.figure(figsize=(16, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('results/correlation_heatmap.png')
plt.close()

# 5. KBest Feature Importance
from sklearn.feature_selection import SelectKBest, f_classif

X = df.drop('fetal_health', axis=1)
y = df['fetal_health']

bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Feature', 'Score']
featureScores = featureScores.sort_values(by='Score', ascending=False)

print("\n--- Feature Importance (KBest) ---")
print(featureScores.head(15))

plt.figure(figsize=(10, 8))
sns.barplot(x='Score', y='Feature', data=featureScores.head(15))
plt.title('Top 15 Features by KBest Score')
plt.tight_layout()
plt.savefig('results/feature_importance.png')
plt.close()
