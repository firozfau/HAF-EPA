from __future__ import annotations

import pandas as pd
from config import OUTPUT_DIR


def save_feature_importance(feature_importance_df: pd.DataFrame, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    feature_importance_df.to_csv(path, index=False)
    return path


def save_balanced_training_data(balanced_train_df: pd.DataFrame, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    balanced_train_df.to_csv(path, index=False)
    return path


def save_test_predictions_csv(recommendation_df: pd.DataFrame, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    recommendation_df.to_csv(path, index=False)
    return path


def save_recommendation_excel(recommendation_df: pd.DataFrame, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    recommendation_df.to_excel(path, index=False)
    return path


def save_evaluation_report(
    filename: str,
    accuracy: float,
    confusion_mat,
    class_report: str,
    train_shape,
    test_shape,
    balanced_train_shape,
    train_label_counts,
    test_label_counts,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== HAF-EPA MODEL EVALUATION ===\n\n")
        f.write(f"Train shape (before balancing): {train_shape}\n")
        f.write(f"Balanced train shape: {balanced_train_shape}\n")
        f.write(f"Test shape: {test_shape}\n\n")
        f.write("Train label distribution (before balancing):\n")
        f.write(f"{train_label_counts}\n\n")
        f.write("Test label distribution:\n")
        f.write(f"{test_label_counts}\n\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{confusion_mat}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
        f.write("\n")

    return path