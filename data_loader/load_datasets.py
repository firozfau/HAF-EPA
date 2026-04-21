from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from config import TRAINING_DATASET_DIR


@dataclass
class LoadedData:
    employees: pd.DataFrame
    projects: pd.DataFrame
    tasks: pd.DataFrame
    employee_skills: pd.DataFrame
    project_skills: pd.DataFrame
    skills: pd.DataFrame
    employee_project_history: pd.DataFrame
    employee_availability: pd.DataFrame
    employee_relationship: pd.DataFrame
    skill_similarity: pd.DataFrame


def load_datasets() -> LoadedData:
    employees = pd.read_csv(TRAINING_DATASET_DIR / "employees.csv")
    projects = pd.read_csv(TRAINING_DATASET_DIR / "projects.csv")
    tasks = pd.read_csv(TRAINING_DATASET_DIR / "tasks.csv")
    employee_skills = pd.read_csv(TRAINING_DATASET_DIR / "employee_skills.csv")
    project_skills = pd.read_csv(TRAINING_DATASET_DIR / "project_skills.csv")
    skills = pd.read_csv(TRAINING_DATASET_DIR / "skills.csv")

    employee_project_history = pd.read_csv(
        TRAINING_DATASET_DIR / "employee_project_history.csv"
    )
    employee_availability = pd.read_csv(
        TRAINING_DATASET_DIR / "employee_availability.csv"
    )
    employee_relationship = pd.read_csv(
        TRAINING_DATASET_DIR / "employee_relationship.csv"
    )
    skill_similarity = pd.read_csv(
        TRAINING_DATASET_DIR / "skill_similarity.csv"
    )

    return LoadedData(
        employees=employees,
        projects=projects,
        tasks=tasks,
        employee_skills=employee_skills,
        project_skills=project_skills,
        skills=skills,
        employee_project_history=employee_project_history,
        employee_availability=employee_availability,
        employee_relationship=employee_relationship,
        skill_similarity=skill_similarity,
    )