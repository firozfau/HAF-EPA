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


# Our goal is to train this HAF-EPA model so it can provide the best-fit employee list for a project.

# Based on the dataset, we create a model. After training, it learns how to determine
# the best-fit employee for upcoming projects.

# In this case, we use Supervised Learning.

# We extract features and create employee-project pairs,
# where each pair is labeled as suitable or not suitable during training.

# Finally, we apply performance metrics to evaluate how well the model is learning,
# such as accuracy, precision, recall, and F1-score.


# In this training, I chose to use the Random Forest model because:
#   1. It works well with structured/tabular data.
#   2. It can handle non-linear relationships (e.g., skill + experience combination).
#   3. It uses multiple trees, which helps reduce overfitting.


# Here:
#    0. Define which features are used in the model.
#    1. Handle data imbalance using sampling.
#    2. Split dataset into training (80%) and testing (20%).
#    3. Train the Random Forest model.
#    4. Use a probability threshold to get final class predictions.
#    5. Evaluate performance using metrics.



# For Machine Learning Model we use:
#    1. sklearn = scikit-learn machine learning library
#    2. ensemble = module for combining multiple models
#    3. RandomForestClassifier = classification model


# Model evaluation: training alone is not enough, we must measure performance
#    accuracy_score ==> How many predictions were correct overall.
#    precision_score ==> Out of predicted "suitable", how many are actually suitable.
#    recall_score ==> Out of actual suitable, how many were correctly predicted.
#    f1_score ==> Balance between precision and recall.
#    confusion_matrix ==> Shows TP, TN, FP, FN values.
#    classification_report ==> Summary of precision, recall, F1-score for each class.


# Note:
# ==> If divide-by-zero happens during precision/recall calculation,
#     zero_division=0 returns 0 instead of error.


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