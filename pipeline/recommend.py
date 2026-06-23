from __future__ import annotations

import pandas as pd

from models.train_model import FEATURE_COLUMNS


def build_recommendation_reason(row: pd.Series) -> str:
    reasons = []

    if row.get("matched_required_skill_count", 0) > 0:
        reasons.append(f"matched required skills: {int(row.get('matched_required_skill_count', 0))}")

    if row.get("weighted_skill_match_score", 0) > 0:
        reasons.append(f"weighted skill match: {row.get('weighted_skill_match_score', 0):.2f}")

    if row.get("related_skill_match_score", 0) > 0:
        reasons.append(f"related skill score: {row.get('related_skill_match_score', 0):.2f}")

    if row.get("avg_experience_on_required_skills", 0) > 0:
        reasons.append(f"required-skill experience: {row.get('avg_experience_on_required_skills', 0):.2f} years")

    if row.get("avg_past_performance_score", 0) > 0:
        reasons.append(f"past performance: {row.get('avg_past_performance_score', 0):.2f}")

    if row.get("availability_fit_score", 0) > 0:
        reasons.append(f"availability fit: {row.get('availability_fit_score', 0):.2f}")

    if row.get("task_context_match_score", 0) > 0:
        reasons.append(f"task context match: {row.get('task_context_match_score', 0):.2f}")

    if row.get("soft_skill_compatibility_score", 0) > 0:
        reasons.append(f"soft-skill compatibility: {row.get('soft_skill_compatibility_score', 0):.2f}")

    if not reasons:
        return "Recommendation based on combined hybrid recommendation features."

    return "; ".join(reasons)


def generate_test_recommendations(model, test_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in test_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for recommendation: {missing}")

    recommendation_df = test_df.copy()

    X_test = recommendation_df[FEATURE_COLUMNS].copy()
    recommendation_df["predicted_label"] = model.predict(X_test)
    recommendation_df["match_probability"] = model.predict_proba(X_test)[:, 1]
    recommendation_df["recommendation_reason"] = recommendation_df.apply(
        build_recommendation_reason,
        axis=1,
    )

    recommendation_df = recommendation_df.sort_values(
        by=["match_probability", "weighted_skill_match_score", "avg_past_performance_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    recommendation_columns = [
        "employee_id",
        "project_id",
        "predicted_label",
        "label",
        "match_probability",
        "matched_skill_count",
        "matched_required_skill_count",
        "matched_optional_skill_count",
        "weighted_skill_match_score",
        "related_skill_match_score",
        "avg_experience_on_required_skills",
        "avg_past_performance_score",
        "availability_fit_score",
        "task_context_match_score",
        "soft_skill_compatibility_score",
        "recommendation_reason",
    ]

    existing_columns = [c for c in recommendation_columns if c in recommendation_df.columns]
    return recommendation_df[existing_columns].copy()