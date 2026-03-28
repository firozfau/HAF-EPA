from __future__ import annotations
import pandas as pd


def employee_map(employees: pd.DataFrame, employee_skill_map: pd.DataFrame ) -> pd.DataFrame:

    employees_enriched = employees.merge(
        employee_skill_map,
        on="employee_id",
        how="left"
    )

    # ensure skill_name is always list
    employees_enriched["skill_name"] = employees_enriched["skill_name"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    return employees_enriched


def project_map( projects: pd.DataFrame, project_skill_map: pd.DataFrame ) -> pd.DataFrame:

    projects_enriched = projects.merge(
        project_skill_map,
        on="project_id",
        how="left"
    )

    # ensure skill_name is always list
    projects_enriched["skill_name"] = projects_enriched["skill_name"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    return projects_enriched