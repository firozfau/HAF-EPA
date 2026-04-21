from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from data_loader.load_datasets import LoadedData


# -------------------------------
# Helper functions
# -------------------------------

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


def _check_required(df: pd.DataFrame, name: str, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _to_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# -------------------------------
# MAIN FUNCTION
# -------------------------------

def normalize_loaded_data(data: LoadedData) -> LoadedData:

    # --- Clean column names ---
    employees = _clean_columns(data.employees)
    projects = _clean_columns(data.projects)
    tasks = _clean_columns(data.tasks)
    employee_skills = _clean_columns(data.employee_skills)
    project_skills = _clean_columns(data.project_skills)
    skills = _clean_columns(data.skills)
    eph = _clean_columns(data.employee_project_history)
    ea = _clean_columns(data.employee_availability)
    er = _clean_columns(data.employee_relationship)
    ss = _clean_columns(data.skill_similarity)

    # --- Required columns (Professor expectation aligned) ---
    _check_required(employees, "employees", ["employee_id", "full_name"])
    _check_required(projects, "projects", ["project_id", "project_name"])
    _check_required(tasks, "tasks", ["task_id", "project_id"])

    _check_required(employee_skills, "employee_skills", ["employee_id", "skill_id"])
    _check_required(project_skills, "project_skills", ["project_id", "skill_id"])
    _check_required(skills, "skills", ["skill_id", "skill_name"])

    # Professor requirement critical datasets
    _check_required(eph, "employee_project_history",
                    ["employee_id", "project_id", "performance_score"])
    _check_required(ss, "skill_similarity",
                    ["skill_id_1", "skill_id_2", "similarity_score"])
    _check_required(ea, "employee_availability", ["employee_id"])
    _check_required(er, "employee_relationship",
                    ["employee_id_1", "employee_id_2"])

    # --- Clean text ---
    employees = _clean_text(employees)
    projects = _clean_text(projects)
    tasks = _clean_text(tasks)
    employee_skills = _clean_text(employee_skills)
    project_skills = _clean_text(project_skills)
    skills = _clean_text(skills)
    eph = _clean_text(eph)
    ea = _clean_text(ea)
    er = _clean_text(er)
    ss = _clean_text(ss)

    # --- Convert numeric (Professor features need these) ---
    employee_skills = _to_numeric(employee_skills, ["years_experience"])
    project_skills = _to_numeric(project_skills, ["importance_weight", "required_flag"])
    eph = _to_numeric(eph, ["performance_score"])
    ss = _to_numeric(ss, ["similarity_score"])
    ea = _to_numeric(ea, ["allocation_percent"])
    er = _to_numeric(er, ["compatibility_score"])

    # --- Convert dates (needed for availability + recency) ---
    employee_skills = _to_datetime(employee_skills, ["last_used_date"])
    eph = _to_datetime(eph, ["start_date", "end_date"])
    ea = _to_datetime(ea, ["available_from", "available_to"])

    # --- Remove duplicates ---
    employees = employees.drop_duplicates()
    projects = projects.drop_duplicates()
    tasks = tasks.drop_duplicates()
    employee_skills = employee_skills.drop_duplicates()
    project_skills = project_skills.drop_duplicates()
    skills = skills.drop_duplicates()
    eph = eph.drop_duplicates()
    ea = ea.drop_duplicates()
    er = er.drop_duplicates()
    ss = ss.drop_duplicates()

    # --- Return ---
    return LoadedData(
        employees=employees,
        projects=projects,
        tasks=tasks,
        employee_skills=employee_skills,
        project_skills=project_skills,
        skills=skills,
        employee_project_history=eph,
        employee_availability=ea,
        employee_relationship=er,
        skill_similarity=ss,
    )


# Backward compatibility
def normalize_data(data: LoadedData) -> LoadedData:
    return normalize_loaded_data(data)