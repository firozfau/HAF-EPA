from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import OUTPUT_DIR


FEATURE_COLUMNS = [
    "matched_skill_count",
    "matched_required_skill_count",
    "matched_optional_skill_count",
    "employee_skill_count",
    "project_skill_count",
    "required_skill_count",
    "optional_skill_count",
    "skill_match_score",
    "employee_skill_coverage",
    "missing_required_skill_count",
    "has_any_skill_match",
    "strong_skill_match",
    "weighted_skill_match_score",
    "related_skill_match_score",
    "avg_experience_on_required_skills",
    "avg_past_performance_score",
    "availability_fit_score",
    "task_context_match_score",
    "soft_skill_compatibility_score",
]

TARGET_COLUMN = "label"
MODEL_FILENAME = "HAF-EPA.joblib"


def train_haf_epa_model(
    balanced_train_df: pd.DataFrame,
    random_state: int = 42,
):
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required_columns if c not in balanced_train_df.columns]
    if missing:
        raise ValueError(f"Missing columns for training: {missing}")

    X_train = balanced_train_df[FEATURE_COLUMNS].copy()
    y_train = balanced_train_df[TARGET_COLUMN].copy()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / MODEL_FILENAME
    joblib.dump(model, model_path)

    feature_importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    return model, feature_importance_df, model_path