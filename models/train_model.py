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

from config import RANDOM_STATE, TEST_SIZE, TRAINING_THRESHOLD


FEATURE_COLUMNS = [
    "matched_skill_count",
    "employee_skill_count",
    "project_skill_count",
    "skill_match_score",
    "employee_skill_coverage",
    "has_any_skill_match",
    "strong_skill_match",
    "experience_score",
    "availability_score",
    "primary_skill_match",
]


@dataclass
class TrainingArtifacts:
    model: RandomForestClassifier
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    classification_report: str
    threshold: float
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _balance_training_data(labeled_df: pd.DataFrame) -> pd.DataFrame:
    positive_df = labeled_df[labeled_df["label"] == 1].copy()
    negative_df = labeled_df[labeled_df["label"] == 0].copy()

    if positive_df.empty or negative_df.empty:
        return labeled_df.copy()

    negative_sample_size = min(len(negative_df), len(positive_df) * 4)

    negative_sample_df = negative_df.sample(
        n=negative_sample_size,
        random_state=RANDOM_STATE,
    )

    balanced_df = pd.concat([positive_df, negative_sample_df], ignore_index=True)
    balanced_df = balanced_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return balanced_df


def train_random_forest_model(labeled_df: pd.DataFrame) -> TrainingArtifacts:
    training_df = _balance_training_data(labeled_df)

    X = training_df[FEATURE_COLUMNS].copy()
    y = training_df["label"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=6,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = TRAINING_THRESHOLD
    y_pred = (y_prob >= threshold).astype(int)

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
        threshold=threshold,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )