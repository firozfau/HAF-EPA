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

from config import (TRAINED_MODEL,HELD_OUT_TEST_DATA,)

# This function is used to generate and train the machine learning model.

# 1. First, it loads all raw datasets.
# 2. Then it normalizes and cleans the datasets.

# 3. It builds employee and project skill mappings.
# 4. It enriches employees and projects with their skill lists.

# 5. It keeps only projects that have at least one skill.

# 6. Then it creates employee-project pairs using full dataset.

# 7. Feature engineering is applied to create useful matching features.

# 8. Labels are generated using historical task data (suitable or not).

# 9. The model is trained using 80% data and 20% is kept for testing.

# 10. The trained model is saved for future use.

# 11. The 20% test data (held-out data) is also saved for evaluation.

# Finally, it prints dataset size and confirms model generation.

def generate_train_model():
    
    data = load_datasets()

    # Normalize and clean all loaded datasets
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

    print(f"Full dataset size : {len(labeled_df)}")
    print(f"Successfully generated training model: HAF-EPA.joblib and this model uses 80% data of the full dataset.")
    print(f"Successfully generated training model: HAF-EPA-TEST.joblib and this model uses 20% data of the full dataset.")
    
  


    return trained