from __future__ import annotations

import joblib
import pandas as pd

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


def load_trained_model(model_path=None):
    """
    Load trained model from OUTPUT_DIR by default.
    """
    if model_path is None:
        model_path = OUTPUT_DIR / "trained_model.pkl"

    model = joblib.load(model_path)
    return model


def prepare_prediction_data(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only model feature columns in correct order.
    """
    missing_features = [col for col in FEATURE_COLUMNS if col not in features_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns for prediction: {missing_features}")

    X = features_df[FEATURE_COLUMNS].copy()
    X = X.fillna(0)

    return X


def build_explanation(row: pd.Series) -> str:
    """
    Create a simple human-readable explanation for each recommendation.
    """
    reasons = []

    if row.get("matched_required_skill_count", 0) > 0:
        reasons.append(
            f"matched required skills: {int(row.get('matched_required_skill_count', 0))}"
        )

    if row.get("weighted_skill_match_score", 0) > 0:
        reasons.append(
            f"weighted skill match: {row.get('weighted_skill_match_score', 0):.2f}"
        )

    if row.get("related_skill_match_score", 0) > 0:
        reasons.append(
            f"related skill score: {row.get('related_skill_match_score', 0):.2f}"
        )

    if row.get("avg_experience_on_required_skills", 0) > 0:
        reasons.append(
            f"required-skill experience: {row.get('avg_experience_on_required_skills', 0):.2f} years"
        )

    if row.get("avg_past_performance_score", 0) > 0:
        reasons.append(
            f"past performance: {row.get('avg_past_performance_score', 0):.2f}"
        )

    if row.get("availability_fit_score", 0) > 0:
        reasons.append(
            f"availability fit: {row.get('availability_fit_score', 0):.2f}"
        )

    if row.get("task_context_match_score", 0) > 0:
        reasons.append(
            f"task context match: {row.get('task_context_match_score', 0):.2f}"
        )

    if row.get("soft_skill_compatibility_score", 0) > 0:
        reasons.append(
            f"soft-skill compatibility: {row.get('soft_skill_compatibility_score', 0):.2f}"
        )

    if not reasons:
        return "Recommendation based on overall feature combination."

    return "; ".join(reasons)


def predict_recommendations(
    features_df: pd.DataFrame,
    model=None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Predict employee-project suitability and return ranked recommendations.
    """
    if model is None:
        model = load_trained_model()

    X = prepare_prediction_data(features_df)

    result_df = features_df.copy()

    # Binary prediction
    result_df["predicted_label"] = model.predict(X)

    # Probability if available
    if hasattr(model, "predict_proba"):
        result_df["match_probability"] = model.predict_proba(X)[:, 1]
    else:
        result_df["match_probability"] = 0.0

    # Explanation
    result_df["recommendation_reason"] = result_df.apply(build_explanation, axis=1)

    # Sort by probability first, then by weighted match / performance
    sort_columns = ["match_probability"]
    ascending = [False]

    if "weighted_skill_match_score" in result_df.columns:
        sort_columns.append("weighted_skill_match_score")
        ascending.append(False)

    if "avg_past_performance_score" in result_df.columns:
        sort_columns.append("avg_past_performance_score")
        ascending.append(False)

    ranked_df = result_df.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)

    if top_n is not None:
        ranked_df = ranked_df.head(top_n).copy()

    return ranked_df


def predict_for_single_project(
    features_df: pd.DataFrame,
    project_id: str,
    model=None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Filter recommendation candidates for a single project and rank employees.
    """
    project_df = features_df[features_df["project_id"] == project_id].copy()

    if project_df.empty:
        raise ValueError(f"No rows found for project_id={project_id}")

    ranked_df = predict_recommendations(
        features_df=project_df,
        model=model,
        top_n=top_n,
    )

    selected_columns = [
        "employee_id",
        "project_id",
        "predicted_label",
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

    existing_columns = [col for col in selected_columns if col in ranked_df.columns]
    return ranked_df[existing_columns].copy()


def save_predictions(predictions_df: pd.DataFrame, filename: str = "predictions.csv") -> str:
    """
    Save predictions into OUTPUT_DIR.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    predictions_df.to_csv(output_path, index=False)
    return str(output_path)