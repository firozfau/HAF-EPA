from __future__ import annotations

import pandas as pd

from config import MAX_EXPERIENCE_YEARS
#**Feature Engineering**
#  is the process of transforming raw data into useful features so that models or analyses can perform better.


def calculate_skill_match(employee_skills: object, project_skills: object) -> float:
    employee_set = set(employee_skills) if isinstance(employee_skills, list) else set()
    project_set = set(project_skills) if isinstance(project_skills, list) else set()

    if not project_set:
        return 0.0

    matched_skills = employee_set.intersection(project_set)
    return len(matched_skills) / len(project_set)


def add_features(pairs_df: pd.DataFrame) -> pd.DataFrame:
    features_df = pairs_df.copy()

    features_df["skill_match_score"] = features_df.apply(
        lambda row: calculate_skill_match(
            row["employee_skills"],
            row["project_skills"],
        ),
        axis=1,
    )

    features_df["experience_score"] = (
        features_df["experience"].clip(lower=0, upper=MAX_EXPERIENCE_YEARS)
        / MAX_EXPERIENCE_YEARS
    )

    features_df["availability_score"] = (
        features_df["availability"].clip(lower=0, upper=100)
        / 100
    )

    return features_df