import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time

# 1. Load preprocessed data
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

# Adjust y for XGBoost (expects labels 0, 1, 2 instead of 1, 2, 3)
y_train_xgb = y_train - 1
y_test_xgb = y_test - 1

print("Data loaded successfully.")

# 2. Define Models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    "MLP (Neural Network)": MLPClassifier(random_state=42, max_iter=1000),
    "SVM (RBF)": SVC(kernel='rbf', random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000)
}

# 3. Train and Evaluate
results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    start_time = time.time()
    
    # Train
    if name == "XGBoost":
        model.fit(X_train, y_train_xgb)
        y_pred_xgb = model.predict(X_test)
        y_pred = y_pred_xgb + 1
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
    end_time = time.time()
    
    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Training Time: {end_time - start_time:.2f} seconds")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological']))

# 4. Summary Table
print("\n=== FINAL RESULTS SUMMARY ===")
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
print(results_df)

results_df.to_csv('results/member1_model_results.csv', index=False)
