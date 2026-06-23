from pathlib import Path
import ast
import joblib
import pandas as pd

from src.pdf_parser import extract_text_from_pdf
from src.project_parser import extract_skills_from_text
from src.pdf_context_validation import validate_project_pdf_or_raise

# This module extracts project requirements from a PDF
# and recommends the best-matching employees using a trained model.

# 1. Define file paths for the trained model and employee dataset.
# 2. Specify the feature columns required for prediction.

# 3. Parse employee skills into a clean list format.

# 4. Extract project skills from PDF text.
#    NOTE: uploaded PDF project description / required skills section
#    is used as the source of required project skills.
# 5. Build employee-project matching features:
#    - skill matching metrics
#    - experience and availability scores
#    - programming score and management score
#    - programming/management project-required skill matching
#    - primary skill matching

# 6. Load trained model and employee dataset.
# 7. Validate required employee data columns.

# 8. Generate features based on extracted project text.
# 9. Predict suitability scores for each employee.

# 10. Filter employees with at least one required skill match.

# 11. Rank employees based on:
#     - match percentage
#     - employee required-skill percentage
#     - matched required skill count
#     - skill match score

# 12. Return top recommended employees with matching details.


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent

MODEL_PATH = ROOT_DIR / "output" / "HAF-EPA.joblib"
EMPLOYEES_PATH = ROOT_DIR / "output" / "HAF-EPA_employee_reference.csv"

# IMPORTANT: these columns must stay identical to the columns used when
# training output/HAF-EPA.joblib. Scikit-learn stores feature_names_in_ and
# will reject prediction data that contains different names.
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

PROGRAMMING_SKILLS = {
    "python", "java", "javascript", "typescript", "php", "laravel",
    "react", "angular", "vue", "node.js", "node", "django", "flask",
    "spring boot", "html", "css", "sql", "mysql", "postgresql",
    "mongodb", "docker", "kubernetes", "aws", "azure", "git",
    "linux", "machine learning", "deep learning", "data analysis",
    "data science", "ui/ux design", "go", "rust", "scala",
    "android", "ios", "tensorflow", "pytorch", "devops",
    "automation testing", "cybersecurity", "testing", "c++", "c#",
    "rest api", "graphql", "microservices", "ci/cd", "firebase",
    "bootstrap", "tailwind css", "kotlin", "swift"
}

MANAGEMENT_SKILLS = {
    "project management", "agile", "scrum", "leadership",
    "team management", "communication", "planning", "coordination",
    "risk management", "time management", "stakeholder management",
    "resource management", "budgeting", "documentation",
    "presentation", "decision making", "problem solving",
    "requirement analysis", "business analysis", "quality management",
    "client communication", "project planning", "sprint planning",
    "task management", "team leadership", "conflict resolution",
    "strategic thinking", "negotiation", "process improvement"
}


def validate_project_pdf_or_raise(project_text: str):
    text = project_text.lower()

    # required sections (at least some must exist)
    required_sections = [
        "project overview",
        "project description",
        "technical requirements",
        "technology requirements",
        "required skills",
        "objectives",
        "expected outcome",
        "modules",
    ]

    # non-project indicators (wrong document signals)
    invalid_indicators = [
        "thesis",
        "declaration",
        "signature",
        "matriculation",
        "regulation",
        "policy",
        "i hereby confirm",
    ]

    # check how many valid sections exist
    section_match_count = sum(1 for sec in required_sections if sec in text)

    # check invalid hints
    invalid_match_count = sum(1 for word in invalid_indicators if word in text)

    # ❌ reject conditions
    if section_match_count < 2 or "required skills" not in text or invalid_match_count >= 2:
        raise ValueError("Uploaded pdf file are not able to recognaize HAF-EPA training model")


def parse_employee_skills(value) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(skill).strip() for skill in parsed if str(skill).strip()]
    except Exception:
        pass

    return [item.strip() for item in text.split(",") if item.strip()]


def normalize_skill(skill: str) -> str:
    return str(skill).strip().lower()


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def count_category_match(employee_skill_set, project_skill_set, category_set):
    # project_skill_set = uploaded PDF required skills
    # category_set = programming or management skill category
    # Result means: employee matched X skills out of PDF required category skills.
    required_category_skills = project_skill_set.intersection(category_set)
    matched_category_skills = employee_skill_set.intersection(required_category_skills)

    required_count = len(required_category_skills)
    matched_count = len(matched_category_skills)

    percentage = (matched_count / required_count) * 100 if required_count else 0.0

    return matched_count, required_count, round(percentage, 2)


def build_employee_features(employee_df: pd.DataFrame, project_text: str) -> pd.DataFrame:
    # Extract required skills from uploaded PDF project description.
    # In current web upload flow, project_parser returns one extracted skill list.
    # That list is treated as required project skills.
    project_skills = extract_skills_from_text(project_text)

    if not project_skills:
        raise ValueError("No recognizable required skills found from the uploaded PDF.")

    # Remove duplicate required skills while keeping original order.
    required_project_skills = list(dict.fromkeys(project_skills))

    # Optional skills are not separately extracted from PDF in current flow.
    # Keep optional skill list empty so model feature names remain compatible.
    optional_project_skills = []

    required_skill_set = {normalize_skill(skill) for skill in required_project_skills}
    optional_skill_set = {normalize_skill(skill) for skill in optional_project_skills}
    project_skill_set = required_skill_set.union(optional_skill_set)

    rows = []

    for _, row in employee_df.iterrows():
        employee_skills = parse_employee_skills(row.get("employee_skills", ""))

        employee_skill_set = {normalize_skill(skill) for skill in employee_skills}

        matched_skill_count = len(employee_skill_set.intersection(project_skill_set))
        matched_required_skill_count = len(employee_skill_set.intersection(required_skill_set))
        matched_optional_skill_count = len(employee_skill_set.intersection(optional_skill_set))

        employee_skill_count = len(employee_skills)
        project_skill_count = len(project_skill_set)
        required_skill_count = len(required_skill_set)
        optional_skill_count = len(optional_skill_set)

        skill_match_score = matched_skill_count / project_skill_count if project_skill_count else 0.0
        employee_skill_coverage = matched_skill_count / employee_skill_count if employee_skill_count else 0.0
        missing_required_skill_count = max(required_skill_count - matched_required_skill_count, 0)
        has_any_skill_match = 1 if matched_skill_count > 0 else 0
        strong_skill_match = 1 if skill_match_score >= 0.5 else 0

        required_part = (
            matched_required_skill_count / required_skill_count
            if required_skill_count else 0.0
        )

        optional_part = (
            matched_optional_skill_count / optional_skill_count
            if optional_skill_count else 0.0
        )

        weighted_skill_match_score = (0.7 * required_part) + (0.3 * optional_part)

        experience = safe_float(row.get("experience", 0))
        availability = safe_float(row.get("availability", 0))

        avg_experience_on_required_skills = experience if matched_required_skill_count > 0 else 0.0
        avg_past_performance_score = 0.0
        availability_fit_score = max(0.0, min(availability / 100.0, 1.0))
        related_skill_match_score = 0.0
        task_context_match_score = 0.0
        soft_skill_compatibility_score = 0.0

        # Percent values for graph/table display.
        # Employee Skill % = matched required skills / required skills from uploaded PDF.
        # Required Skill % = 100%, because required skills come from the uploaded project PDF.
        employee_skill_percentage = round(required_part * 100, 2)
        required_skill_percentage = 100.00 if required_skill_count else 0.00

        # These values are generated automatically in output/HAF-EPA_employee_reference.csv
        # by pipeline/export_top_employees.py after running python main.py.
        # They are employee general profile scores, not direct PDF-required skill match.
        programming_score = safe_float(row.get("programming_score", 0))
        management_score = safe_float(row.get("management_score", 0))
        programming_skill_percentage = safe_float(row.get("programming_skill_percentage", 0))
        management_skill_percentage = safe_float(row.get("management_skill_percentage", 0))
        estimated_project_working_count = safe_float(row.get("estimated_project_working_count", 0))

        # PDF-based category match:
        # This is the important graph/table value.
        # Example: employee has 9 out of 10 required management skills = 90%.
        programming_matched_count, programming_required_count, programming_project_match_percentage = (
            count_category_match(
                employee_skill_set,
                required_skill_set,
                PROGRAMMING_SKILLS,
            )
        )

        management_matched_count, management_required_count, management_project_match_percentage = (
            count_category_match(
                employee_skill_set,
                required_skill_set,
                MANAGEMENT_SKILLS,
            )
        )

        overall_project_match_percentage = employee_skill_percentage

        # Explainability details for the web application.
        # These lists make it clear which project skills were extracted from the PDF,
        # which of them matched the employee profile, and which ones are still missing.
        matched_project_skills = [
            skill for skill in required_project_skills
            if normalize_skill(skill) in employee_skill_set
        ]
        missing_project_skills = [
            skill for skill in required_project_skills
            if normalize_skill(skill) not in employee_skill_set
        ]

        rows.append({
            "employee_id": row["employee_id"],
            "full_name": row["full_name"],
            "employee_skills": employee_skills,
            "project_skills": required_project_skills,
            "matched_project_skills": matched_project_skills,
            "missing_project_skills": missing_project_skills,

            "matched_skill_count": matched_skill_count,
            "matched_required_skill_count": matched_required_skill_count,
            "matched_optional_skill_count": matched_optional_skill_count,

            "employee_skill_count": employee_skill_count,
            "project_skill_count": project_skill_count,
            "required_skill_count": required_skill_count,
            "optional_skill_count": optional_skill_count,

            "employee_skill_percentage": employee_skill_percentage,
            "required_skill_percentage": required_skill_percentage,

            "programming_score": programming_score,
            "management_score": management_score,
            "programming_skill_percentage": programming_skill_percentage,
            "management_skill_percentage": management_skill_percentage,
            "estimated_project_working_count": estimated_project_working_count,

            "programming_matched_count": programming_matched_count,
            "programming_required_count": programming_required_count,
            "programming_project_match_percentage": programming_project_match_percentage,

            "management_matched_count": management_matched_count,
            "management_required_count": management_required_count,
            "management_project_match_percentage": management_project_match_percentage,

            "overall_project_match_percentage": overall_project_match_percentage,

            "experience": experience,
            "availability": availability,

            "skill_match_score": skill_match_score,
            "employee_skill_coverage": employee_skill_coverage,
            "missing_required_skill_count": missing_required_skill_count,
            "has_any_skill_match": has_any_skill_match,
            "strong_skill_match": strong_skill_match,
            "weighted_skill_match_score": weighted_skill_match_score,
            "related_skill_match_score": related_skill_match_score,

            "avg_experience_on_required_skills": avg_experience_on_required_skills,
            "avg_past_performance_score": avg_past_performance_score,
            "availability_fit_score": availability_fit_score,
            "task_context_match_score": task_context_match_score,
            "soft_skill_compatibility_score": soft_skill_compatibility_score,
        })

    return pd.DataFrame(rows)


def build_recommendation_explanation(row: pd.Series) -> str:
    """
    Create a short human-readable reason for the recommendation.

    This is not used by the ML model itself. It is used only in the web app to
    make the final ranking transparent for the supervisor/user.
    """
    matched_count = int(row.get("matched_required_skill_count", 0))
    required_count = int(row.get("required_skill_count", 0))
    project_match = safe_float(row.get("overall_project_match_percentage", 0))
    ml_score = safe_float(row.get("ml_prediction_percentage", 0))
    experience = safe_float(row.get("experience", 0))
    availability = safe_float(row.get("availability", 0))

    reasons = [
        f"matched {matched_count} of {required_count} extracted project skills",
        f"project-skill match {project_match:.2f}%",
        f"ML suitability score {ml_score:.2f}%",
    ]

    if experience > 0:
        reasons.append(f"experience value {experience:.1f}")

    if availability > 0:
        reasons.append(f"availability {availability:.2f}%")

    return "; ".join(reasons)


def recommend_top_employees_from_pdf(pdf_path: str, top_k: int = 10):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not EMPLOYEES_PATH.exists():
        raise FileNotFoundError(f"Employee file not found: {EMPLOYEES_PATH}")

    model = joblib.load(MODEL_PATH)
    employee_df = pd.read_csv(EMPLOYEES_PATH)

    required_columns = {"employee_id", "full_name", "employee_skills", "experience", "availability"}
    missing_columns = required_columns - set(employee_df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns in employee_reference.csv: {sorted(missing_columns)}")

    project_text = extract_text_from_pdf(pdf_path)
    validate_project_pdf_or_raise(project_text)

    feature_df = build_employee_features(employee_df, project_text)

    expected_features = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    missing_features = [col for col in expected_features if col not in feature_df.columns]

    if missing_features:
        raise ValueError(f"Missing prediction features: {missing_features}")

    # Select exactly the model's training columns, in the same order, so pandas
    # does not pass unseen/misordered feature names to scikit-learn.
    X = feature_df[expected_features].copy()

    # ML prediction stage: this is the actual Random Forest prediction used by HAF-EPA.
    if hasattr(model, "predict_proba"):
        feature_df["predicted_score"] = model.predict_proba(X)[:, 1]
    else:
        feature_df["predicted_score"] = model.predict(X)

    # Keep the ML score visible as a separate value.
    # Supervisor issue fixed: the previous UI used only this probability as "Match %",
    # which made the best employees look like 8-10% matches. For user-facing display,
    # "Match %" now means extracted project-skill coverage, while this field shows
    # the ML model's suitability probability separately.
    feature_df["ml_prediction_percentage"] = (feature_df["predicted_score"] * 100).round(2)

    # Human-readable project match percentage from extracted PDF skills.
    feature_df["match_percentage"] = feature_df["overall_project_match_percentage"].round(2)

    # Only employees with actual required skill match are allowed in final ranking.
    filtered_df = feature_df[feature_df["matched_required_skill_count"] > 0].copy()

    if filtered_df.empty:
        raise ValueError("No employees matched the required skills extracted from the uploaded PDF.")

    # Professional ranking:
    # 1. project skill match percentage from uploaded PDF
    # 2. ML suitability prediction
    # 3. matched required skill count
    # 4. project skill match score
    top_df = (
        filtered_df
        .sort_values(
            [
                "match_percentage",
                "ml_prediction_percentage",
                "matched_required_skill_count",
                "skill_match_score",
            ],
            ascending=[False, False, False, False],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    top_df["recommendation_rank"] = top_df.index + 1
    top_df["recommendation_reason"] = top_df.apply(build_recommendation_explanation, axis=1)

    recommendations = top_df[
        [
            "recommendation_rank",
            "employee_id",
            "full_name",
            "employee_skills",
            "project_skills",
            "matched_project_skills",
            "missing_project_skills",

            "matched_skill_count",
            "matched_required_skill_count",
            "matched_optional_skill_count",

            "employee_skill_count",
            "project_skill_count",
            "required_skill_count",
            "optional_skill_count",

            "employee_skill_percentage",
            "required_skill_percentage",

            "programming_score",
            "management_score",
            "programming_skill_percentage",
            "management_skill_percentage",
            "estimated_project_working_count",

            "programming_matched_count",
            "programming_required_count",
            "programming_project_match_percentage",

            "management_matched_count",
            "management_required_count",
            "management_project_match_percentage",

            "overall_project_match_percentage",

            "experience",
            "availability",
            "avg_experience_on_required_skills",
            "skill_match_score",
            "predicted_score",
            "ml_prediction_percentage",
            "match_percentage",
            "recommendation_reason",
        ]
    ].to_dict(orient="records")

    return recommendations, project_text
