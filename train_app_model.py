from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fetal_health.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fetal_health_model.joblib"

# Load and clean dataset
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().reset_index(drop=True)

# Separate input features and target
X = df.drop(columns=["fetal_health"])
y = df["fetal_health"].astype(int)

# Train the selected model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    ))
])

model.fit(X, y)

# Save model and feature information
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_names": X.columns.tolist(),
        "class_names": {
            1: "Normal",
            2: "Suspect",
            3: "Pathological"
        }
    },
    MODEL_PATH
)

print("Model trained and saved successfully!")
print(f"Training records: {len(X)}")
print(f"Number of input features: {len(X.columns)}")
print(f"Saved model: {MODEL_PATH}")
print("Classes: Normal (1), Suspect (2), Pathological (3)")