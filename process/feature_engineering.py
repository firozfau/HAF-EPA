from __future__ import annotations

import pandas as pd


# Feature engineering is an important part of machine learning for employee-project suitability.
# It converts raw employee-project pair data into meaningful features like:
# skill match score, skill counts, and normalized experience and availability,
# so the model can learn better and improve matching accuracy.

def _safe_list(value):
    if isinstance(value, list):
        return value
    return []


def add_features(pairs_df: pd.DataFrame) -> pd.DataFrame:
   
    featured_df = pairs_df.copy()

    # make sure skill columns are lists
    featured_df["employee_skills"] = featured_df["employee_skills"].apply(_safe_list)
    featured_df["project_skills"] = featured_df["project_skills"].apply(_safe_list)

    # basic counts
    featured_df["employee_skill_count"] = featured_df["employee_skills"].apply(len)
    featured_df["project_skill_count"] = featured_df["project_skills"].apply(len)

    # matched skill count
    featured_df["matched_skill_count"] = featured_df.apply(
        lambda row: len(set(row["employee_skills"]).intersection(set(row["project_skills"]))),
        axis=1,
    )

    # overlap ratio (main matching strength)
    featured_df["skill_match_score"] = featured_df.apply(
        lambda row: (
            row["matched_skill_count"] / row["project_skill_count"]
            if row["project_skill_count"] > 0 else 0.0
        ),
        axis=1,
    )

    # employee-side overlap ratio
    featured_df["employee_skill_coverage"] = featured_df.apply(
        lambda row: (
            row["matched_skill_count"] / row["employee_skill_count"]
            if row["employee_skill_count"] > 0 else 0.0
        ),
        axis=1,
    )

    # exact skill match flag
    featured_df["has_any_skill_match"] = featured_df["matched_skill_count"].apply(
        lambda x: 1 if x > 0 else 0
    )

    # strong match flag
    featured_df["strong_skill_match"] = featured_df["skill_match_score"].apply(
        lambda x: 1 if x >= 0.5 else 0
    )

    # normalize experience
    featured_df["experience_score"] = featured_df["experience"].apply(
        lambda x: min(float(x) / 20.0, 1.0)
    )

    # normalize availability
    featured_df["availability_score"] = featured_df["availability"].apply(
        lambda x: min(float(x) / 100.0, 1.0)
    )

    # primary skill match
    if "primary_skill" in featured_df.columns:
        featured_df["primary_skill_match"] = featured_df.apply(
            lambda row: 1 if row["primary_skill"] in row["project_skills"] else 0,
            axis=1,
        )
    else:
        featured_df["primary_skill_match"] = 0

    return featured_df