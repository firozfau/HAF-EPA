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
# 5. Build employee-project matching features:
#    - skill matching metrics
#    - experience and availability scores
#    - primary skill matching

# 6. Load trained model and employee dataset.
# 7. Validate required employee data columns.

# 8. Generate features based on extracted project text.
# 9. Predict suitability scores for each employee.

# 10. Filter employees with at least one matching skill.

# 11. Rank employees based on:
#     - matched skill count
#     - skill match score
#     - predicted score

# 12. Return top recommended employees with matching details.


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent

MODEL_PATH = ROOT_DIR / "output" / "HAF-EPA.joblib"
EMPLOYEES_PATH = ROOT_DIR / "output" / "employee_reference.csv"

FEATURE_COLUMNS = [
    "matched_skill_count",
    "employee_skill_count",
    "project_skill_count",
    "skill_match_score",
    "employee_skill_coverage",
    "has_any_skill_match",
    "strong_skill_match",
    "experience_score",
    "availability_score",
    "primary_skill_match",
]

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


def build_employee_features(employee_df: pd.DataFrame, project_text: str) -> pd.DataFrame:
    project_skills = extract_skills_from_text(project_text)

    if not project_skills:
        raise ValueError("No recognizable skills found in the uploaded PDF.")

    rows = []

    for _, row in employee_df.iterrows():
        employee_skills = parse_employee_skills(row.get("employee_skills", ""))

        matched_skill_count = len(set(employee_skills).intersection(set(project_skills)))
        employee_skill_count = len(employee_skills)
        project_skill_count = len(project_skills)

        skill_match_score = matched_skill_count / project_skill_count if project_skill_count else 0.0
        employee_skill_coverage = matched_skill_count / employee_skill_count if employee_skill_count else 0.0
        has_any_skill_match = 1 if matched_skill_count > 0 else 0
        strong_skill_match = 1 if skill_match_score >= 0.5 else 0

        experience = float(row.get("experience", 0))
        availability = float(row.get("availability", 0))

        experience_score = min(experience / 20.0, 1.0)
        availability_score = min(availability / 100.0, 1.0)

        primary_skill_match = 0
        if employee_skills:
            primary_skill_match = 1 if employee_skills[0] in project_skills else 0

        rows.append({
            "employee_id": row["employee_id"],
            "full_name": row["full_name"],
            "employee_skills": employee_skills,
            "project_skills": project_skills,
            "matched_skill_count": matched_skill_count,
            "employee_skill_count": employee_skill_count,
            "project_skill_count": project_skill_count,
            "skill_match_score": skill_match_score,
            "employee_skill_coverage": employee_skill_coverage,
            "has_any_skill_match": has_any_skill_match,
            "strong_skill_match": strong_skill_match,
            "experience_score": experience_score,
            "availability_score": availability_score,
            "primary_skill_match": primary_skill_match,
        })

    return pd.DataFrame(rows)


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

# now start work on pdf 
    project_text = extract_text_from_pdf(pdf_path)
    # validate project PDF before processing
    validate_project_pdf_or_raise(project_text)
    feature_df = build_employee_features(employee_df, project_text)

    X = feature_df[FEATURE_COLUMNS].copy()

    # apply model 
    if hasattr(model, "predict_proba"):
        feature_df["predicted_score"] = model.predict_proba(X)[:, 1]
    else:
        feature_df["predicted_score"] = model.predict(X)

    feature_df["match_percentage"] = (feature_df["predicted_score"] * 100).round(2)

    # only employees with actual skill match are allowed in final ranking
    filtered_df = feature_df[feature_df["matched_skill_count"] > 0].copy()

    if filtered_df.empty:
        raise ValueError("No employees matched the project skills extracted from the uploaded PDF.")

    # Professional ranking:
    # 1. matched skills
    # 2. project skill match score
    # 3. model predicted score
    top_df = (
        filtered_df
        .sort_values(["matched_skill_count", "skill_match_score", "predicted_score"], ascending=[False, False, False]
        ) .head(top_k).reset_index(drop=True)
    )

    recommendations = top_df[
        [
            "employee_id",
            "full_name",
            "employee_skills",
            "project_skills",
            "matched_skill_count",
            "skill_match_score",
            "match_percentage",
        ]
    ].to_dict(orient="records")

    return recommendations, project_text