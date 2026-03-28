from __future__ import annotations

import pandas as pd


def build_pairs_employee_project(employees_sample: pd.DataFrame,projects_sample: pd.DataFrame) -> pd.DataFrame:

    employees_df = employees_sample[["employee_id", "full_name", "skill_name", "experience", "availability"]].copy()

    projects_df = projects_sample[["project_id", "project_name", "skill_name"]].copy()

    pairs_df = employees_df.merge(
        projects_df,
        how="cross",
        suffixes=("_emp", "_proj"))

    pairs_df = pairs_df.rename(columns={
        "skill_name_emp": "employee_skills",
        "skill_name_proj": "project_skills",
    })

    return pairs_df[
        [
            "employee_id",
            "full_name",
            "project_id",
            "project_name",
            "employee_skills",
            "project_skills",
            "experience",
            "availability",
        ]
    ].copy()