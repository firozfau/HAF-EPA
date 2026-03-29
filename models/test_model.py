from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from process.normalize import normalize_datasets
from process.employee_skill_mapping import emp_map_build
from process.project_skill_mapping import project_map_build
from process.mapping import employee_map, project_map
from process.feature_engineering import add_features
from process.lebel_employee_project import add_labels
from models.process import load_model
from models.predict import predict_result

from config import TRAINED_MODEL, TEST_DATASET_DIR, TRAINING_THRESHOLD


@dataclass
class ExternalTestArtifacts:
    predicted_df: pd.DataFrame
    accuracy: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    confusion_matrix: list[list[int]] | None
    classification_report: str | None
    threshold: float
    total_rows: int


def load_test_datasets(
    test_dataset_dir: str | Path = TEST_DATASET_DIR
) -> SimpleNamespace:
    """
    Load external test datasets from datasets/test-dataset directory
    and return an attribute-style object compatible with normalize_datasets().
    """
    test_dataset_dir = Path(test_dataset_dir)

    required_files = {
        "employees": "employees.csv",
        "projects": "projects.csv",
        "tasks": "tasks.csv",
        "employee_skills": "employee_skills.csv",
        "project_skills": "project_skills.csv",
        "skills": "skills.csv",
    }

    data: dict[str, pd.DataFrame] = {}

    for key, filename in required_files.items():
        file_path = test_dataset_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing required test dataset file: {file_path}"
            )

        data[key] = pd.read_csv(file_path)

    return SimpleNamespace(**data)


def _build_pairs_for_single_project(
    employees_df: pd.DataFrame,
    project_row: pd.Series
) -> pd.DataFrame:
    project_id = project_row["project_id"]
    project_name = (
        project_row["project_name"]
        if "project_name" in project_row.index
        else project_id
    )
    project_skills = (
        project_row["skill_name"]
        if "skill_name" in project_row.index
        else []
    )

    rows = []

    for _, emp_row in employees_df.iterrows():
        rows.append(
            {
                "employee_id": emp_row["employee_id"],
                "full_name": (
                    emp_row["full_name"]
                    if "full_name" in emp_row.index
                    else emp_row["employee_id"]
                ),
                "project_id": project_id,
                "project_name": project_name,
                "employee_skills": (
                    emp_row["skill_name"]
                    if "skill_name" in emp_row.index
                    else []
                ),
                "project_skills": project_skills,
                "experience": (
                    emp_row["experience"]
                    if "experience" in emp_row.index
                    else 0
                ),
                "availability": (
                    emp_row["availability"]
                    if "availability" in emp_row.index
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def test_model(
    project_id: str,
    candidate_employee_ids: list[str] | None = None,
    test_dataset_dir: str | Path = TEST_DATASET_DIR,
    threshold: float = TRAINING_THRESHOLD,
) -> ExternalTestArtifacts:
    """
    Run prediction and optional external evaluation on unseen external test data.
    """

    # 1. Load external unseen test dataset
    data = load_test_datasets(test_dataset_dir)

    # 2. Normalize test datasets
    normalized_data = normalize_datasets(data)
    employees = normalized_data["employees"]
    projects = normalized_data["projects"]
    tasks = normalized_data["tasks"]
    employee_skills = normalized_data["employee_skills"]
    project_skills = normalized_data["project_skills"]
    skills = normalized_data["skills"]

    # 3. Build skill mappings
    employee_skill_map = emp_map_build(employee_skills, skills)
    project_skill_map = project_map_build(projects, project_skills, skills)

    employees_enriched = employee_map(employees, employee_skill_map)
    projects_enriched = project_map(projects, project_skill_map)

    projects_with_skills = projects_enriched[
        projects_enriched["skill_name"].apply(len) > 0
    ].copy()

    # 4. Select target project
    selected_project_df = projects_with_skills[
        projects_with_skills["project_id"] == project_id
    ].copy()

    if selected_project_df.empty:
        raise ValueError(
            f"Project ID not found or has no skills in external test dataset: {project_id}"
        )

    selected_project_row = selected_project_df.iloc[0]
    project_employees_df = employees_enriched.copy()

    # 5. Optional shortlist filter
    if candidate_employee_ids:
        candidate_ids_str = [str(emp_id) for emp_id in candidate_employee_ids]
        project_employees_df = project_employees_df[
            project_employees_df["employee_id"].astype(str).isin(candidate_ids_str)
        ].copy()

    if project_employees_df.empty:
        raise ValueError(
            f"No employees found for project-specific prediction in external test dataset: {project_id}"
        )

    # 6. Create employee-project pairs
    pair_df = _build_pairs_for_single_project(
        employees_df=project_employees_df,
        project_row=selected_project_row,
    )

    # 7. Feature engineering
    featured_df = add_features(pair_df)

    # 8. Add labels if possible
    labeled_df = add_labels(featured_df, tasks)

    # 9. Load trained model
    model = load_model(TRAINED_MODEL)

    # 10. Predict suitability scores
    predicted_df = predict_result(labeled_df, model)

    # 11. Optional external evaluation
    accuracy = None
    precision = None
    recall = None
    f1 = None
    cm_list = None
    report = None

    if "label" in predicted_df.columns:
        evaluation_df = predicted_df[predicted_df["label"].notna()].copy()

        if not evaluation_df.empty:
            y_true = evaluation_df["label"].astype(int)
            y_prob = evaluation_df["predicted_score"]
            y_pred = (y_prob >= threshold).astype(int)

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            cm = confusion_matrix(y_true, y_pred)
            cm_list = cm.tolist()
            report = classification_report(y_true, y_pred, zero_division=0)

            print("\n External Test Evaluation Metrics:")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1-score : {f1:.4f}")
            print(f"Threshold: {threshold:.2f}")
            print("\nConfusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(report)
        else:
            print("\nLabel column exists but contains no valid values. Metrics cannot be computed.")
    else:
        print("\nNo 'label' column found. Metrics cannot be computed.")

    return ExternalTestArtifacts(
        predicted_df=predicted_df,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm_list,
        classification_report=report,
        threshold=threshold,
        total_rows=len(predicted_df),
    )