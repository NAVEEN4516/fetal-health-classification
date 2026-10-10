#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Member 2 - ML Models
# Fetal Health Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import GridSearchCV

print("Libraries imported successfully!")


# In[3]:


# Load preprocessed data created by Member 1

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "processed" if (REPO_ROOT / "data" / "processed").exists() else BASE_DIR / "data" / "processed"

X_train = np.load(DATA_DIR / "X_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_test = np.load(DATA_DIR / "y_test.npy")
print("Data loaded successfully!")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[4]:


# 1. Gradient Boosting Classifier

gb_model = GradientBoostingClassifier(random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
y_pred_gb = gb_model.predict(X_test)

# Accuracy
gb_accuracy = accuracy_score(y_test, y_pred_gb)

print("Gradient Boosting Accuracy:", gb_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))


# In[5]:


# Gradient Boosting - Confusion Matrix

cm_gb = confusion_matrix(y_test, y_pred_gb)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_gb,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gradient Boosting - Confusion Matrix")
plt.close()


# In[6]:


print("X_train dtype:", X_train.dtype)
print("X_test dtype:", X_test.dtype)
print("y_train dtype:", y_train.dtype)
print("y_test dtype:", y_test.dtype)

print("X_train contiguous:", X_train.flags["C_CONTIGUOUS"])
print("y_train contiguous:", y_train.flags["C_CONTIGUOUS"])

print("Classes:", np.unique(y_train))


# In[7]:


import lightgbm
import numpy as np

print("LightGBM version:", lightgbm.__version__)
print("NumPy version:", np.__version__)


# In[8]:


# Prepare data in a LightGBM-friendly format

X_train_lgb = np.ascontiguousarray(X_train, dtype=np.float32)
X_test_lgb = np.ascontiguousarray(X_test, dtype=np.float32)

y_train_lgb = np.asarray(y_train, dtype=np.int32)
y_test_lgb = np.asarray(y_test, dtype=np.int32)

print(X_train_lgb.dtype, X_train_lgb.shape)
print(y_train_lgb.dtype, y_train_lgb.shape)


# In[9]:


# Train LightGBM with safe data types

lgbm_model = LGBMClassifier(
    random_state=42,
    verbosity=-1,
    n_jobs=1
)

lgbm_model.fit(X_train_lgb, y_train_lgb)

y_pred_lgbm = lgbm_model.predict(X_test_lgb)

lgbm_accuracy = accuracy_score(y_test_lgb, y_pred_lgbm)

print("LightGBM Accuracy:", lgbm_accuracy)
print("\nClassification Report:")
print(classification_report(y_test_lgb, y_pred_lgbm))


# In[10]:


# LightGBM - Confusion Matrix

cm_lgbm = confusion_matrix(y_test_lgb, y_pred_lgbm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_lgbm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LightGBM - Confusion Matrix")
plt.close()


# In[11]:


# 3. K-Nearest Neighbors

knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn_model.predict(X_test)

# Accuracy
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("KNN Accuracy:", knn_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn))


# In[12]:


# KNN - Confusion Matrix

cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_knn,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN - Confusion Matrix")
plt.close()


# In[13]:


# 4. Linear SVM

svm_model = SVC(
    kernel='linear',
    random_state=42
)

# Train the model
svm_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)

# Accuracy
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("Linear SVM Accuracy:", svm_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))


# In[14]:


# Linear SVM - Confusion Matrix

cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Linear SVM - Confusion Matrix")
plt.close()


# In[15]:


# 5. Decision Tree

dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_model.predict(X_test)

# Accuracy
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Decision Tree Accuracy:", dt_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))


# In[16]:


# Decision Tree - Confusion Matrix

cm_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_dt,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree - Confusion Matrix")
plt.close()


# In[17]:


# Model Comparison

results = pd.DataFrame({
    'Model': [
        'Gradient Boosting',
        'LightGBM',
        'KNN',
        'Linear SVM',
        'Decision Tree'
    ],
    'Accuracy': [
        gb_accuracy,
        lgbm_accuracy,
        knn_accuracy,
        svm_accuracy,
        dt_accuracy
    ]
})

results = results.sort_values('Accuracy', ascending=False)

print(results.to_string(index=False))


# In[18]:


# Model Accuracy Comparison

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results,
    x='Accuracy',
    y='Model'
)

plt.xlim(0, 1)
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.title('Model Accuracy Comparison')

plt.close()


# In[19]:


# LightGBM Hyperparameter Tuning using GridSearchCV

lgbm_base = LGBMClassifier(
    random_state=42,
    verbosity=-1,
    n_jobs=1
)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [15, 31]
}

grid_search = GridSearchCV(
    estimator=lgbm_base,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train_lgb, y_train_lgb)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(grid_search.best_score_)


# In[20]:


# Evaluate the tuned LightGBM on the test set

best_lgbm = grid_search.best_estimator_

y_pred_lgbm_tuned = best_lgbm.predict(X_test_lgb)

tuned_lgbm_accuracy = accuracy_score(
    y_test_lgb,
    y_pred_lgbm_tuned
)

print("Tuned LightGBM Test Accuracy:", tuned_lgbm_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test_lgb,
        y_pred_lgbm_tuned
    )
)


# In[21]:


# Tuned LightGBM - Confusion Matrix

cm_lgbm_tuned = confusion_matrix(
    y_test_lgb,
    y_pred_lgbm_tuned
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_lgbm_tuned,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Tuned LightGBM - Confusion Matrix")

plt.close()


# In[22]:


# Final Model Results

final_results = pd.DataFrame({
    'Model': [
        'Gradient Boosting',
        'LightGBM',
        'KNN',
        'Linear SVM',
        'Decision Tree',
        'Tuned LightGBM'
    ],
    'Test Accuracy': [
        gb_accuracy,
        lgbm_accuracy,
        knn_accuracy,
        svm_accuracy,
        dt_accuracy,
        tuned_lgbm_accuracy
    ]
})

final_results = final_results.sort_values(
    'Test Accuracy',
    ascending=False
).reset_index(drop=True)

final_results['Test Accuracy (%)'] = (
    final_results['Test Accuracy'] * 100
).round(2)

print("\n=== FINAL RESULTS SUMMARY (MEMBER 2) ===")
print(final_results.to_string())

