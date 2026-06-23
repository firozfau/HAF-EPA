from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.train_model import FEATURE_COLUMNS


def evaluate_model(model, test_df: pd.DataFrame):
    missing = [c for c in FEATURE_COLUMNS + ["label"] if c not in test_df.columns]
    if missing:
        raise ValueError(f"Missing columns for evaluation: {missing}")

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df["label"].copy()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    result = {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=4),
    }

    return result