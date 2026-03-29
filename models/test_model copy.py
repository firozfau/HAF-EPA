from __future__ import annotations

import pandas as pd

from data_loader.load_datasets import load_datasets
from process.normalize import normalize_datasets
from process.employee_skill_mapping import emp_map_build
from process.project_skill_mapping import project_map_build
from process.mapping import employee_map, project_map
from process.pair_creation import build_pairs_employee_project
from process.feature_engineering import add_features
from process.lebel_employee_project import add_labels
from models.process import load_model
from models.predict import predict_result

from config import (
    EMPLOYEE_SAMPLE_SIZE,
    PROJECT_SAMPLE_SIZE,
    RANDOM_STATE,
    TRAINED_MODEL,
)

# We use it to test the model: starting from loading the data, processing it, creating features, loading the saved model, and generating the final predictions.
# I trained the model using 80% of the dataset and evaluated it on the remaining 20% unseen test data. After that, I use the saved model (**HAF-EPA.joblib**) to make predictions on new employee–project data or specific project candidates.


def _build_pairs_for_single_project(
    employees_df: pd.DataFrame,
    project_row: pd.Series
) -> pd.DataFrame:
    project_id = project_row["project_id"]
    project_name = project_row["project_name"] if "project_name" in project_row.index else project_id
    project_skills = project_row["skill_name"] if "skill_name" in project_row.index else []

    rows = []

    for _, emp_row in employees_df.iterrows():
        rows.append(
            {
                "employee_id": emp_row["employee_id"],
                "full_name": emp_row["full_name"] if "full_name" in emp_row.index else emp_row["employee_id"],
                "project_id": project_id,
                "project_name": project_name,
                "employee_skills": emp_row["skill_name"] if "skill_name" in emp_row.index else [],
                "project_skills": project_skills,
                "experience": emp_row["experience"] if "experience" in emp_row.index else 0,
                "availability": emp_row["availability"] if "availability" in emp_row.index else 0,
            }
        )

    return pd.DataFrame(rows)


def test_model(
    project_id: str | None = None,
    candidate_employee_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    If project_id is None:
        - general sampled test mode

    If project_id is provided:
        - project-specific prediction mode

    If candidate_employee_ids is provided:
        - only those employees are used
        - this helps align ML candidates with KG candidates
    """

    # 1. Load raw datasets
    data = load_datasets()

    # 2. Normalize datasets
    normalized_data = normalize_datasets(data)
    employees = normalized_data["employees"]
    projects = normalized_data["projects"]
    tasks = normalized_data["tasks"]
    employee_skills = normalized_data["employee_skills"]
    project_skills = normalized_data["project_skills"]
    skills = normalized_data["skills"]

    # 3. Skill mapping
    employee_skill_map = emp_map_build(employee_skills, skills)
    project_skill_map = project_map_build(projects, project_skills, skills)

    employees_enriched = employee_map(employees, employee_skill_map)
    projects_enriched = project_map(projects, project_skill_map)

    projects_with_skills = projects_enriched[
        projects_enriched["skill_name"].apply(len) > 0
    ].copy()

    # 4. Pair creation
    if project_id is None:
        employees_sample = employees_enriched.sample(
            n=min(EMPLOYEE_SAMPLE_SIZE, len(employees_enriched)),
            random_state=RANDOM_STATE
        )

        projects_sample = projects_with_skills.sample(
            n=min(PROJECT_SAMPLE_SIZE, len(projects_with_skills)),
            random_state=RANDOM_STATE
        )

        empp_pair_DataFrame = build_pairs_employee_project(
            employees_sample,
            projects_sample
        )

    else:
        selected_project_df = projects_with_skills[
            projects_with_skills["project_id"] == project_id
        ].copy()

        if selected_project_df.empty:
            raise ValueError(f"Project ID not found or has no skills: {project_id}")

        selected_project_row = selected_project_df.iloc[0]

        project_employees_df = employees_enriched.copy()

        if candidate_employee_ids is not None and len(candidate_employee_ids) > 0:
            project_employees_df = project_employees_df[
                project_employees_df["employee_id"].isin(candidate_employee_ids)
            ].copy()

        if project_employees_df.empty:
            raise ValueError(f"No employees found for project-specific test: {project_id}")

        empp_pair_DataFrame = _build_pairs_for_single_project(
            employees_df=project_employees_df,
            project_row=selected_project_row
        )

    # 5. Feature engineering
    featured_DataFrame = add_features(empp_pair_DataFrame)

    # 6. Labeling
    labeled_DataFrame = add_labels(featured_DataFrame, tasks)

    # 7. Load saved model
    model = load_model(TRAINED_MODEL)

    # 8. Predict
    predicted_DataFrame = predict_result(labeled_DataFrame, model)

    return predicted_DataFrame