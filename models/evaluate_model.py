from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from models.process import load_model
from config import TRAINED_MODEL, HELD_OUT_TEST_DATA

# This file is used to evaluate the trained machine learning model.

# It loads the trained model and test dataset (held-out data),
# which was not used during training.

# The model predicts probability for each employee-project pair,
# then uses a threshold to convert probability into final prediction (0 or 1).

# After prediction, it calculates performance metrics to check how well the model is working.

# These metrics include:
#   accuracy  → overall correct predictions
#   precision → correct "suitable" predictions
#   recall    → how many actual suitable cases are detected
#   f1-score  → balance between precision and recall

# It also creates:
#   confusion_matrix → shows TP, TN, FP, FN
#   classification_report → detailed summary of model performance

# Finally, it returns all evaluation results in a structured format (EvaluationArtifacts),
# so we can analyze model performance easily.

@dataclass
class EvaluationArtifacts:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    classification_report: str
    threshold: float
    X_test: pd.DataFrame
    y_test: pd.Series


def evaluate_model() -> EvaluationArtifacts:
    model = load_model(TRAINED_MODEL)

    held_out = joblib.load(HELD_OUT_TEST_DATA)
    X_test = held_out["X_test"]
    y_test = held_out["y_test"]
    threshold = held_out["threshold"]

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return EvaluationArtifacts(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm.tolist(),
        classification_report=report,
        threshold=threshold,
        X_test=X_test,
        y_test=y_test,
    )