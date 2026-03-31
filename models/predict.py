from __future__ import annotations

import pandas as pd

# This function is used to determine how suitable each employee is for a project.
# It uses FEATURE_COLUMNS, which are the important input features for the model.

# These features include skill matching, counts, experience, and availability,
# which help the model calculate suitability.

# The model predicts a probability score (predicted_score),
# which shows how likely an employee is suitable for a project.

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


def predict_result(input_df: pd.DataFrame, model) -> pd.DataFrame:
    predicted_df = input_df.copy()

    X = predicted_df[FEATURE_COLUMNS].copy()

    predicted_df["predicted_score"] = model.predict_proba(X)[:, 1]

    return predicted_df