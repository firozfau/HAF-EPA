from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from config import (DATASET_1_DIR,DATASET_2_DIR)

# load_datasets is used to load data from multiple dataset sources.

# 1. It loads dataset 1 and dataset 2 CSV files (employees, projects, tasks, employee_skills).
# 2. Then it combines (concat) both datasets to create a single full dataset.

# 3. project_skills and skills are taken from dataset 1.

# Finally, it returns all datasets in a structured format (LoadedData)
# so they can be used in further processing steps.


@dataclass
class LoadedData:
    employees: pd.DataFrame
    projects: pd.DataFrame
    tasks: pd.DataFrame
    employee_skills: pd.DataFrame
    project_skills: pd.DataFrame
    skills: pd.DataFrame


def load_datasets() -> LoadedData:
    employees_dataset_1 = pd.read_csv(DATASET_1_DIR / "employees.csv")
    projects_dataset_1 = pd.read_csv(DATASET_1_DIR / "projects.csv")
    tasks_dataset_1 = pd.read_csv(DATASET_1_DIR / "tasks.csv")
    employee_skills_dataset_1 = pd.read_csv(DATASET_1_DIR / "employee_skills.csv")
    
    project_skills_dataset_1 = pd.read_csv(DATASET_1_DIR / "project_skills.csv")
    skills_dataset_1 = pd.read_csv(DATASET_1_DIR / "skills.csv")

    employees_dataset_2 = pd.read_csv(DATASET_2_DIR / "employees.csv")
    projects_dataset_2 = pd.read_csv(DATASET_2_DIR / "projects.csv")
    tasks_dataset_2 = pd.read_csv(DATASET_2_DIR / "tasks.csv")
    employee_skills_dataset_2 = pd.read_csv(DATASET_2_DIR / "employee_skills.csv")



    employees = pd.concat([employees_dataset_1, employees_dataset_2], ignore_index=True)
    projects = pd.concat([projects_dataset_1, projects_dataset_2], ignore_index=True)
    tasks = pd.concat([tasks_dataset_1, tasks_dataset_2], ignore_index=True)
    employee_skills = pd.concat([employee_skills_dataset_1, employee_skills_dataset_2], ignore_index=True)

    project_skills = project_skills_dataset_1.copy()
    skills = skills_dataset_1.copy()

    return LoadedData(
            employees=employees,
            projects=projects,
            tasks=tasks,
            employee_skills=employee_skills,
            project_skills=project_skills,
            skills=skills,
        )
