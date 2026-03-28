from __future__ import annotations

import pandas as pd

#We generate labels using historical task assignments.
#  If an employee has worked on a project before, we assign label 1, otherwise 0. 
# This allows the model to learn real-world matching patterns.

def add_labels(features_df: pd.DataFrame, tasks_df: pd.DataFrame) -> pd.DataFrame:
    labeled_df = features_df.copy()

    # Build real historical employee-project pairs from tasks
    historical_pairs = set(
        zip(tasks_df["employee_id"], tasks_df["project_id"])
    )

    # Assign label: 1 if employee actually worked on project
    labeled_df["label"] = labeled_df.apply(
        lambda row: 1
        if (row["employee_id"], row["project_id"]) in historical_pairs
        else 0,
        axis=1,
    )

    return labeled_df