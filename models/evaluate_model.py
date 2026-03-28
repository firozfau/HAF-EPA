from __future__ import annotations

from data_loader.load_datasets import load_datasets
from process.normalize import normalize_datasets
from process.employee_skill_mapping import emp_map_build
from process.project_skill_mapping import project_map_build
from process.mapping import employee_map, project_map
from process.pair_creation import build_pairs_employee_project
from process.feature_engineering import add_features
from process.lebel_employee_project import add_labels
from models.train_model import train_random_forest_model

from config import (
    EMPLOYEE_SAMPLE_SIZE,
    PROJECT_SAMPLE_SIZE,
    RANDOM_STATE,
)


def evaluate_model() -> None:
    print("HAF-EPA Validation and Evaluation Starting...")

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

    # 4. Sampling
    employees_sample = employees_enriched.sample(
        n=min(EMPLOYEE_SAMPLE_SIZE, len(employees_enriched)),
        random_state=RANDOM_STATE
    )

    projects_with_skills = projects_enriched[
        projects_enriched["skill_name"].apply(len) > 0
    ]

    projects_sample = projects_with_skills.sample(
        n=min(PROJECT_SAMPLE_SIZE, len(projects_with_skills)),
        random_state=RANDOM_STATE
    )

    # 5. Employee-project pair creation
    empp_pair_DataFrame = build_pairs_employee_project(
        employees_sample,
        projects_sample
    )

    # 6. Feature engineering
    featured_DataFrame = add_features(empp_pair_DataFrame)

    # 7. Label employee-project pair data
    labeled_DataFrame = add_labels(featured_DataFrame, tasks)
    evaluate_labeled_sample= labeled_DataFrame[["employee_id","project_id","skill_match_score","experience_score","availability_score","label",]].head()

    # 8. Train + evaluate
    trained_model_data = train_random_forest_model(labeled_DataFrame)

    # 9. Print evaluation metrics
    print("\nEvaluate traing model:")

    print("\nModel Accuracy:", trained_model_data.accuracy)
    print("Precision:", trained_model_data.precision)
    print("Recall:", trained_model_data.recall)
    print("F1-score:", trained_model_data.f1)
    print("Confusion matrix:")
    print(trained_model_data.confusion_matrix)
    print("Classification report:")
    print(trained_model_data.classification_report)