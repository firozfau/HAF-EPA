from __future__ import annotations
import pandas as pd


def _safe_list(values) -> list:
    if isinstance(values, list):
        return values
    if pd.isna(values):
        return []
    return [values]


def create_employee_project_pairs(
    employees_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    employee_skills_df: pd.DataFrame,
    project_skills_df: pd.DataFrame,
    skills_df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    employee_availability_df: pd.DataFrame,
) -> pd.DataFrame:
    # -----------------------------
    # 1. skill_id -> skill_name map
    # -----------------------------
    skill_map = skills_df[["skill_id", "skill_name"]].drop_duplicates()
    skill_id_to_name = dict(zip(skill_map["skill_id"], skill_map["skill_name"]))

    # -----------------------------------------
    # 2. Employee skill summary
    # employee_id -> [skill names]
    # -----------------------------------------
    emp_sk = employee_skills_df.copy()
    emp_sk["skill_name"] = emp_sk["skill_id"].map(skill_id_to_name)

    employee_skill_summary = (
        emp_sk.groupby("employee_id")["skill_name"]
        .apply(lambda x: sorted(set([v for v in x.dropna()])))
        .reset_index()
        .rename(columns={"skill_name": "employee_skills"})
    )

    employee_exp_summary = (
        emp_sk.groupby("employee_id")["years_experience"]
        .mean()
        .reset_index()
        .rename(columns={"years_experience": "employee_avg_skill_experience"})
    )

    # -----------------------------------------
    # 3. Project skill summary
    # project_id -> all / required / optional
    # -----------------------------------------
    proj_sk = project_skills_df.copy()
    proj_sk["skill_name"] = proj_sk["skill_id"].map(skill_id_to_name)

    all_project_skills = (
        proj_sk.groupby("project_id")["skill_name"]
        .apply(lambda x: sorted(set([v for v in x.dropna()])))
        .reset_index()
        .rename(columns={"skill_name": "project_skills"})
    )

    required_project_skills = (
        proj_sk[proj_sk["required_flag"] == 1]
        .groupby("project_id")["skill_name"]
        .apply(lambda x: sorted(set([v for v in x.dropna()])))
        .reset_index()
        .rename(columns={"skill_name": "required_project_skills"})
    )

    optional_project_skills = (
        proj_sk[proj_sk["required_flag"] == 0]
        .groupby("project_id")["skill_name"]
        .apply(lambda x: sorted(set([v for v in x.dropna()])))
        .reset_index()
        .rename(columns={"skill_name": "optional_project_skills"})
    )

    project_weight_summary = (
        proj_sk.groupby("project_id")["importance_weight"]
        .mean()
        .reset_index()
        .rename(columns={"importance_weight": "avg_project_skill_weight"})
    )

    # -----------------------------------------
    # 4. Task context summary per project
    # -----------------------------------------
    task_context = tasks_df.copy()

    if "required_skill_context" in task_context.columns:
        task_skill_context_summary = (
            task_context.groupby("project_id")["required_skill_context"]
            .apply(lambda x: sorted(set([v for v in x.dropna()])))
            .reset_index()
            .rename(columns={"required_skill_context": "task_required_skill_context"})
        )
    else:
        task_skill_context_summary = pd.DataFrame(
            columns=["project_id", "task_required_skill_context"]
        )

    if "task_domain" in task_context.columns:
        task_domain_summary = (
            task_context.groupby("project_id")["task_domain"]
            .apply(lambda x: sorted(set([v for v in x.dropna()])))
            .reset_index()
            .rename(columns={"task_domain": "task_domains"})
        )
    else:
        task_domain_summary = pd.DataFrame(columns=["project_id", "task_domains"])

    # -----------------------------------------
    # 5. Availability summary
    # -----------------------------------------
    availability_summary = employee_availability_df.copy()

    if "allocation_percent" in availability_summary.columns:
        availability_summary = (
            availability_summary.groupby("employee_id")["allocation_percent"]
            .mean()
            .reset_index()
            .rename(columns={"allocation_percent": "allocation_percent"})
        )
    else:
        availability_summary = pd.DataFrame(
            columns=["employee_id", "allocation_percent"]
        )

    # -----------------------------------------
    # 6. Merge summaries into employee/project
    # -----------------------------------------
    employees_enriched = employees_df.copy()
    employees_enriched = employees_enriched.merge(
        employee_skill_summary, on="employee_id", how="left"
    )
    employees_enriched = employees_enriched.merge(
        employee_exp_summary, on="employee_id", how="left"
    )
    employees_enriched = employees_enriched.merge(
        availability_summary, on="employee_id", how="left"
    )

    projects_enriched = projects_df.copy()
    projects_enriched = projects_enriched.merge(
        all_project_skills, on="project_id", how="left"
    )
    projects_enriched = projects_enriched.merge(
        required_project_skills, on="project_id", how="left"
    )
    projects_enriched = projects_enriched.merge(
        optional_project_skills, on="project_id", how="left"
    )
    projects_enriched = projects_enriched.merge(
        project_weight_summary, on="project_id", how="left"
    )
    projects_enriched = projects_enriched.merge(
        task_skill_context_summary, on="project_id", how="left"
    )
    projects_enriched = projects_enriched.merge(
        task_domain_summary, on="project_id", how="left"
    )

    # -----------------------------------------
    # 7. Cross join => employee x project pairs
    # -----------------------------------------
    employees_enriched = employees_enriched.copy()
    projects_enriched = projects_enriched.copy()

    employees_enriched["__key"] = 1
    projects_enriched["__key"] = 1

    pairs_df = employees_enriched.merge(projects_enriched, on="__key").drop(columns="__key")

    # -----------------------------------------
    # 8. Fill list-like empty values
    # -----------------------------------------
    list_columns = [
        "employee_skills",
        "project_skills",
        "required_project_skills",
        "optional_project_skills",
        "task_required_skill_context",
        "task_domains",
    ]

    for col in list_columns:
        if col in pairs_df.columns:
            pairs_df[col] = pairs_df[col].apply(_safe_list)

    # numeric fallback
    if "employee_avg_skill_experience" in pairs_df.columns:
        pairs_df["employee_avg_skill_experience"] = pairs_df[
            "employee_avg_skill_experience"
        ].fillna(0)

    if "allocation_percent" in pairs_df.columns:
        pairs_df["allocation_percent"] = pairs_df["allocation_percent"].fillna(0)

    if "avg_project_skill_weight" in pairs_df.columns:
        pairs_df["avg_project_skill_weight"] = pairs_df["avg_project_skill_weight"].fillna(0)

    return pairs_df