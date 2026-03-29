from __future__ import annotations

import joblib

from data_loader.load_datasets import load_datasets
from process.normalize import normalize_datasets
from process.employee_skill_mapping import emp_map_build
from process.project_skill_mapping import project_map_build
from process.mapping import employee_map, project_map
from process.pair_creation import build_pairs_employee_project
from process.feature_engineering import add_features
from process.lebel_employee_project import add_labels
from models.train_model import train_random_forest_model
from models.process import save_model

from config import (
    TRAINED_MODEL,
    HELD_OUT_TEST_DATA,
)


def generate_train_model():
    # 1. Load full dataset
    data = load_datasets()

    # 2. Normalize
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

    # 4. Keep only projects that have at least one skill
    projects_with_skills = projects_enriched[
        projects_enriched["skill_name"].apply(len) > 0
    ].copy()

    # 5. Use FULL dataset for pair creation
    pair_df = build_pairs_employee_project(
        employees_enriched,
        projects_with_skills,
    )

    # 6. Feature engineering
    featured_df = add_features(pair_df)

    # 7. Label generation
    labeled_df = add_labels(featured_df, tasks)

    # 8. Train model using 80% and keep 20% for internal held-out test
    trained = train_random_forest_model(labeled_df)

    # 9. Save trained model
    save_model(trained.model, TRAINED_MODEL)

    # 10. Save held-out internal test data
    HELD_OUT_TEST_DATA.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "X_test": trained.X_test,
            "y_test": trained.y_test,
            "threshold": trained.threshold,
        },
        HELD_OUT_TEST_DATA,
    )

    print(f"\nSuccessfully generated training model: {TRAINED_MODEL}")
    print(f"Successfully saved held-out internal test data: {HELD_OUT_TEST_DATA}")
    print(f"Full labeled dataset size: {len(labeled_df)}")
    print(f"Training size (80%): {len(trained.X_train)}")
    print(f"Internal test size (20%): {len(trained.X_test)}\n")

    return trained