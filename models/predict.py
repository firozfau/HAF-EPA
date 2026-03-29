from __future__ import annotations

import pandas as pd

# This model determine , For this project, how suitable is each employee?
# We use FEATURE_COLUMNS, which represent the number of attributes that help determine a suitable employee.
# we can say, it is the actual input variables (features) used by the model.

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