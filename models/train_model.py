from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE


FEATURE_COLUMNS = ["skill_match_score", "experience_score", "availability_score"]

# Training the ML model (Random Forest)
# Evaluating model performance
# Returning all results in one object, and all result for reuse

@dataclass
class TrainingArtifacts:
    model: RandomForestClassifier
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    classification_report: str
    X: pd.DataFrame
    y: pd.Series


def train_random_forest_model(labeled_df: pd.DataFrame) -> TrainingArtifacts:
    X = labeled_df[FEATURE_COLUMNS]
    y = labeled_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.30).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return TrainingArtifacts(
        model=model,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm.tolist(),
        classification_report=report,
        X=X,
        y=y,
    )