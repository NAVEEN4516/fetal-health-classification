# Multiclass Classification of Fetal Health using Cardiotocogram Data

This project implements various machine learning models to classify fetal health from Cardiotocogram (CTG) data.

## Team Members
* Member 1 (My Work): Random Forest, XGBoost, MLP, SVM (RBF), Logistic Regression. Also responsible for data setup and preprocessing.
* Member 2 (Teammate's Work): Gradient Boosting, LightGBM, KNN, Linear SVM, Decision Tree.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd fetal-health-classification
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the scripts in order:**
    *   `notebooks/01_EDA.py` - Performs Exploratory Data Analysis and generates plots in `results/`.
    *   `notebooks/02_preprocessing.py` - Preprocesses data (scales, removes duplicates) and saves train/test splits in `data/processed/`.
    *   `notebooks/03_models_member1.py` - Runs Member 1's models and saves the summary to `results/`.

## Dataset
The dataset `fetal_health.csv` was downloaded from Kaggle ("Fetal Health Classification" by andrewmvd) and placed in the `data/` folder. It contains 21 features and a target variable `fetal_health` with 3 classes:
*   1: Normal
*   2: Suspect
*   3: Pathological
