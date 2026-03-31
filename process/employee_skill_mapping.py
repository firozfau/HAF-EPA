from __future__ import annotations

import pandas as pd

# Employee_skill_mapping is basically used to match employee with their skills.
# It maps employee skills into a grouped list format like:
# Employee 1 → [Python, SQL] instead of multiple rows like Employee 1 → Python and Employee 1 → SQL.

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