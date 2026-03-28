from __future__ import annotations

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


def test_model() -> None:
    print("HAF-EPA Test Model Starting...")

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

    # 5. Pair creation
    empp_pair_DataFrame = build_pairs_employee_project(employees_sample, projects_sample)

    # 6. Feature engineering
    featured_DataFrame = add_features(empp_pair_DataFrame)

    # 7. Labeling
    labeled_DataFrame = add_labels(featured_DataFrame, tasks)

    # 8. Load saved model
    model = load_model(TRAINED_MODEL)

    # 9. Predict
    predicted_DataFrame = predict_result(labeled_DataFrame, model)
    prediction_sample = predicted_DataFrame[["employee_id", "project_id", "skill_match_score", "predicted_score",]].head()

    print(f"\nTest model prediction sample : ")
    print(prediction_sample)
 
    return predicted_DataFrame