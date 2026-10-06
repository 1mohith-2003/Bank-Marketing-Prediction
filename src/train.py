"""
Training script for the Bank Marketing Prediction project.

This script:
1. Loads the bank marketing dataset
2. Splits the data into training and testing sets
3. Builds preprocessing and Gradient Boosting pipeline
4. Trains the model
5. Evaluates the model
6. Saves the trained model artifact
"""

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "bank.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "bank_marketing_model.joblib"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, sep=";")

print(f"Dataset shape: {df.shape}")


# --------------------------------------------------
# FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("y", axis=1)

y = df["y"].map({
    "no": 0,
    "yes": 1
})


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            NUMERICAL_FEATURES
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES
        )
    ]
)


# --------------------------------------------------
# GRADIENT BOOSTING MODEL
# --------------------------------------------------

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)


# --------------------------------------------------
# COMPLETE PIPELINE
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\nTraining Gradient Boosting model...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]


# --------------------------------------------------
# THRESHOLD
# --------------------------------------------------

THRESHOLD = 0.25

y_pred = (
    y_probability >= THRESHOLD
).astype(int)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"Threshold: {THRESHOLD}")


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_artifact = {
    "model": pipeline,
    "threshold": THRESHOLD,
    "model_name": "Gradient Boosting",
    "f1_score": f1,
    "roc_auc": roc_auc
}


os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

joblib.dump(
    model_artifact,
    MODEL_PATH
)


print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print(f"Model path: {MODEL_PATH}")
print(f"File exists: {os.path.exists(MODEL_PATH)}")