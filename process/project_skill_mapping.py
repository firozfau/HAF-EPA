from __future__ import annotations

import pandas as pd


def extract_skills_from_stack(
    stack_text: object,
    valid_skill_names: set[str],
) -> list[str]:
    """
    Extract skill names from a semicolon-separated technology stack string.

    Example:
    "Python; SQL; React" -> ["Python", "SQL", "React"]
    """
    if pd.isna(stack_text):
        return []

    parts = [x.strip() for x in str(stack_text).split(";")]
    return [x for x in parts if x in valid_skill_names]


def project_map_build(
    projects: pd.DataFrame,
    project_skills: pd.DataFrame,
    skills: pd.DataFrame,
) -> pd.DataFrame:
    
    project_skills_named = project_skills.merge(
        skills,
        on="skill_id",
        how="left",
    )

    project_skill_map_mapped = (
        project_skills_named.groupby("project_id")["skill_name"]
        .apply(list)
        .reset_index()
    )

    valid_skill_names = set(skills["skill_name"].dropna().tolist())

    project_fallback = projects[["project_id", "technology_stack"]].copy()
    project_fallback["skill_name"] = project_fallback["technology_stack"].apply(
        lambda value: extract_skills_from_stack(value, valid_skill_names)
    )

    project_skill_map = project_fallback.merge(
        project_skill_map_mapped,
        on="project_id",
        how="left",
        suffixes=("_fallback", "_mapped"),
    )

    project_skill_map["skill_name"] = project_skill_map.apply(
        lambda row: row["skill_name_mapped"]
        if isinstance(row["skill_name_mapped"], list) and len(row["skill_name_mapped"]) > 0
        else row["skill_name_fallback"],
        axis=1,
    )

    return project_skill_map[["project_id", "skill_name"]]