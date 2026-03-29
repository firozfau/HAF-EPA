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