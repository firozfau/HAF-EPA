from __future__ import annotations

import pandas as pd


def emp_map_build(
    employee_skills: pd.DataFrame,
    skills: pd.DataFrame,
) -> pd.DataFrame:

    employee_skills_named = employee_skills.merge(
        skills,
        on="skill_id",
        how="left",
    )

    employee_skill_map = (
        employee_skills_named.groupby("employee_id")["skill_name"]
        .apply(list)
        .reset_index()
    )

    return employee_skill_map