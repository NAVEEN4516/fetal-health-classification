import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Create data directory for numpy arrays if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/fetal_health.csv')
print(f"Original shape: {df.shape}")

# 2. Drop Duplicates
print("\nDropping duplicates...")
df = df.drop_duplicates()
print(f"Shape after dropping duplicates: {df.shape}")

# 3. Separate features and target
X = df.drop('fetal_health', axis=1)
y = df['fetal_health'].astype(int)

# 4. Standardize features
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-Test Split (70/30, stratified)
print("\nSplitting data (70/30 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"Train size: {X_train.shape}")
print(f"Test size: {X_test.shape}")

# 6. Save processed data
print("\nSaving processed data...")
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)
print("Data preprocessing complete!")
