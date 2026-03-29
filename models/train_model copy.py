from __future__ import annotations

from dataclasses import dataclass

# our goal is , tech to this HAF-EPA model will be able to best fit employee list provide
# Based on the dataset we create model after traing it learn how determine best fit employe upcomming project 
# in that case use use Supervised Learning
# extract feature and make employee-project pair , that employe are suitable or not sutable of this project during traing 
# Finaly apply metrics to get , how much model is learning , that raition such as accuracy, precision, recall, f1, it called performance metrics.

# in that training I decided to use Random Froest Model due to
#   1. use structured/tabular data 
#   2. Non-linear relation handle , such as skill, experienc are maching  [score = a*skill + b*experience]
#   3. Since it is use multiple tree , so overfitting low 

""" 
Here 
0. Which feature model use i used in my model those are defined 
1. I used 2 dataset for handelig data imbalance issue
2. Split daset Traing purpose use 80% and 20% for testing 
3. Random Forest model train
4. use threshold for probability to get final class prediction
5. performance metrics

"""


#For Machine Learing Model we use 3 things 
"""
1. sklearn = scikit-learn machine learning library
2. ensemble = multiple model combine  models section
3. RandomForestClassifier = classification model
"""


#model evaluation: <-- Just training a model is not enough; you also need to measure whether it learned well.
"""
accuracy_score ==> How many predictions were correct out of the total predictions.

precision_score ==> Out of those predicted as “suitable,” how many were actually suitable.

recall_score ==> Out of those who were actually suitable, how many the model correctly identified.

f1_score ==> A balanced score combining precision and recall.

confusion_matrix ==> Provides a table of TP (True Positive), TN (True Negative), FP (False Positive), FN (False Negative).

classification_report ==> A summary showing precision, recall, F1-score, and support for each class.

"""
#Here,
# ==> If a divide-by-zero situation occurs while calculating precision, it will return 0 instead of raising an error.
# ==> 


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

from config import RANDOM_STATE, TEST_SIZE,NUMBER_OF_ESTIMATORS,MAX_DEPTH,MINIMUM_SAMPLES_SPLIT,MINIMUM_SAMPLES_LEAF,CLASS_WEIGHT,N_JOBS,TRAING_THRESHOLD


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
    X: pd.DataFrame
    y: pd.Series


def _balance_training_data(labeled_df: pd.DataFrame) -> pd.DataFrame:
    positive_df = labeled_df[labeled_df["label"] == 1].copy()
    negative_df = labeled_df[labeled_df["label"] == 0].copy()

    if positive_df.empty:
        return labeled_df.copy()

    negative_sample_size = min(len(negative_df), len(positive_df) * 4)

    negative_sample_df = negative_df.sample(
        n=negative_sample_size,
        random_state=RANDOM_STATE
    )

    balanced_df = pd.concat([positive_df, negative_sample_df], ignore_index=True)
    balanced_df = balanced_df.sample(
        frac=1,
        random_state=RANDOM_STATE
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
        n_estimators=NUMBER_OF_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MINIMUM_SAMPLES_SPLIT,
        min_samples_leaf=MINIMUM_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
        n_jobs=N_JOBS,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold =TRAING_THRESHOLD
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
        X=X,
        y=y,
    )