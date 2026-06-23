from __future__ import annotations

import pandas as pd

from data_loader.load_datasets import load_datasets
from process.normalize import normalize_loaded_data
from process.pair_creation import create_employee_project_pairs
from process.feature_engineering import add_features
from process.lebel_employee_project import add_labels


def prepare_labeled_dataset(performance_threshold: float = 7.0) -> pd.DataFrame:
    data = load_datasets()
    data = normalize_loaded_data(data)

    pairs_df = create_employee_project_pairs(
        employees_df=data.employees,
        projects_df=data.projects,
        employee_skills_df=data.employee_skills,
        project_skills_df=data.project_skills,
        skills_df=data.skills,
        tasks_df=data.tasks,
        employee_availability_df=data.employee_availability,
    )

    features_df = add_features(
        pairs_df=pairs_df,
        employee_skills_df=data.employee_skills,
        project_skills_df=data.project_skills,
        skills_df=data.skills,
        employee_project_history_df=data.employee_project_history,
        employee_availability_df=data.employee_availability,
        employee_relationship_df=data.employee_relationship,
        skill_similarity_df=data.skill_similarity,
    )

    labeled_df = add_labels(
        features_df=features_df,
        employee_project_history_df=data.employee_project_history,
        performance_threshold=performance_threshold,
    )

    return labeled_df