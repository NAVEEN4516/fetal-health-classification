from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_PATH = REPO_ROOT / 'data' / 'fetal_health.csv'
PROCESSED_DIR = REPO_ROOT / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("  STEP 2: DATA PREPROCESSING & STRATIFIED PARTITIONING")
print("="*70)

# 1. Load Data
df = pd.read_csv(DATA_PATH)
print(f"Original shape: {df.shape}")

# 2. Drop Duplicates
df = df.drop_duplicates().reset_index(drop=True)
print(f"Post-deduplication shape: {df.shape} (Pruned {2126 - len(df)} duplicates)")

# 3. Separate Features and Target
X = df.drop('fetal_health', axis=1)
y = df['fetal_health'].astype(int)

# 4. Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features standardized to mean=0, variance=1 via StandardScaler.")

# 5. Stratified 70/30 Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"Train set: {X_train.shape} records")
print(f"Test set:  {X_test.shape} records")

# Class distribution verification
print("\nClass distribution in Test Set (Stratified):")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    label = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}.get(u, str(u))
    print(f"  Class {u} ({label}): {c} ({c/len(y_test)*100:.2f}%)")

# 6. Save Processed NumPy Arrays
np.save(PROCESSED_DIR / 'X_train.npy', X_train)
np.save(PROCESSED_DIR / 'X_test.npy', X_test)
np.save(PROCESSED_DIR / 'y_train.npy', y_train)
np.save(PROCESSED_DIR / 'y_test.npy', y_test)

print(f"\nSaved processed arrays into: {PROCESSED_DIR}")
print("="*70)
