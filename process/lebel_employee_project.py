from __future__ import annotations
import pandas as pd


def add_labels(
    features_df: pd.DataFrame,
    employee_project_history_df: pd.DataFrame,
    performance_threshold: float = 7.0,
) -> pd.DataFrame:
    """
    Create binary labels for employee-project suitability.

    Label logic:
    - label = 1 যদি employee ওই project-এ আগে কাজ করে থাকে
      AND তার performance_score >= performance_threshold
    - otherwise label = 0

    This is more aligned with professor feedback than
    simply checking whether the employee worked on the project.
    """
    df = features_df.copy()
    history = employee_project_history_df.copy()

    required_columns = ["employee_id", "project_id", "performance_score"]
    missing_columns = [col for col in required_columns if col not in history.columns]
    if missing_columns:
        raise ValueError(
            f"employee_project_history_df is missing required columns: {missing_columns}"
        )

    history = history[required_columns].drop_duplicates()

    history["label"] = (
        pd.to_numeric(history["performance_score"], errors="coerce")
        .fillna(0) >= performance_threshold
    ).astype(int)

    # If multiple history rows exist for the same employee-project pair,
    # keep the strongest positive signal
    history = (
        history.groupby(["employee_id", "project_id"], as_index=False)["label"]
        .max()
    )

    df = df.merge(history, on=["employee_id", "project_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(int)

    return df