from __future__ import annotations

import ast
import pandas as pd
from config import OUTPUT_DIR


def create_top_employee_summary(
    recommendation_df: pd.DataFrame,
    employees_df: pd.DataFrame,
    employee_skills_df: pd.DataFrame,
    skills_df: pd.DataFrame,
    employee_project_history_df: pd.DataFrame,
    employee_availability_df: pd.DataFrame,
    employee_relationship_df: pd.DataFrame,
    top_n: int = 200,
) -> pd.DataFrame:
    """
    Create a final employee-level summary from recommendation results.
    Employees are ranked by avg_score (average match_probability).
    """

    # -----------------------------
    # 1. Employee basic info
    # -----------------------------
    employee_info_cols = [
        "employee_id",
        "full_name",
        "department",
        "job_title",
        "seniority_level",
        "location",
        "employment_type",
        "primary_language",
        "status",
    ]
    existing_employee_info_cols = [c for c in employee_info_cols if c in employees_df.columns]
    employee_info_df = employees_df[existing_employee_info_cols].drop_duplicates().copy()

    # -----------------------------
    # 2. Employee skills -> skill names
    # -----------------------------
    skill_map_df = skills_df[["skill_id", "skill_name"]].drop_duplicates().copy()

    emp_skill_df = employee_skills_df.merge(skill_map_df, on="skill_id", how="left")

    skill_summary_df = (
        emp_skill_df.groupby("employee_id")["skill_name"]
        .apply(lambda x: ", ".join(sorted(set(str(v).strip() for v in x.dropna()))))
        .reset_index()
        .rename(columns={"skill_name": "skills"})
    )

    experience_summary_df = (
        emp_skill_df.groupby("employee_id")["years_experience"]
        .mean()
        .reset_index()
        .rename(columns={"years_experience": "experience"})
    )

    # -----------------------------
    # 3. Total project work + past performance
    # -----------------------------
    history_summary_df = (
        employee_project_history_df.groupby("employee_id")
        .agg(
            total_project_work=("project_id", "nunique"),
            avg_past_performance_score=("performance_score", "mean"),
        )
        .reset_index()
    )

    # -----------------------------
    # 4. Availability summary
    # -----------------------------
    if "allocation_percent" in employee_availability_df.columns:
        availability_summary_df = (
            employee_availability_df.groupby("employee_id")["allocation_percent"]
            .mean()
            .reset_index()
            .rename(columns={"allocation_percent": "avg_allocation_percent"})
        )
    else:
        availability_summary_df = pd.DataFrame(columns=["employee_id", "avg_allocation_percent"])

    # -----------------------------
    # 5. Compatibility summary
    # -----------------------------
    if "compatibility_score" in employee_relationship_df.columns:
        rel_rows = []

        for _, row in employee_relationship_df.iterrows():
            e1 = row.get("employee_id_1")
            e2 = row.get("employee_id_2")
            score = row.get("compatibility_score")

            if pd.notna(e1) and pd.notna(score):
                rel_rows.append({"employee_id": e1, "avg_compatibility_score": score})
            if pd.notna(e2) and pd.notna(score):
                rel_rows.append({"employee_id": e2, "avg_compatibility_score": score})

        if rel_rows:
            compatibility_summary_df = (
                pd.DataFrame(rel_rows)
                .groupby("employee_id")["avg_compatibility_score"]
                .mean()
                .reset_index()
            )
        else:
            compatibility_summary_df = pd.DataFrame(columns=["employee_id", "avg_compatibility_score"])
    else:
        compatibility_summary_df = pd.DataFrame(columns=["employee_id", "avg_compatibility_score"])

    # -----------------------------
    # 6. Recommendation summary per employee
    # -----------------------------
    required_recommendation_cols = ["employee_id", "match_probability"]
    missing_cols = [c for c in required_recommendation_cols if c not in recommendation_df.columns]
    if missing_cols:
        raise ValueError(f"recommendation_df missing required columns: {missing_cols}")

    agg_dict = {
        "match_probability": "mean",
    }

    optional_aggs = {
        "project_id": "nunique",
        "weighted_skill_match_score": "mean",
        "related_skill_match_score": "mean",
        "avg_experience_on_required_skills": "mean",
        "avg_past_performance_score": "mean",
        "availability_fit_score": "mean",
        "task_context_match_score": "mean",
        "soft_skill_compatibility_score": "mean",
        "predicted_label": "sum",
    }

    for col, agg_fn in optional_aggs.items():
        if col in recommendation_df.columns:
            agg_dict[col] = agg_fn

    rec_summary_df = recommendation_df.groupby("employee_id").agg(agg_dict).reset_index()

    rename_map = {
        "match_probability": "avg_score",
        "project_id": "recommended_project_count",
        "weighted_skill_match_score": "avg_weighted_skill_match_score",
        "related_skill_match_score": "avg_related_skill_match_score",
        "avg_experience_on_required_skills": "avg_required_skill_experience",
        "availability_fit_score": "avg_availability_fit_score",
        "task_context_match_score": "avg_task_context_match_score",
        "soft_skill_compatibility_score": "avg_soft_skill_score",
        "predicted_label": "predicted_positive_count",
    }
    rec_summary_df = rec_summary_df.rename(columns=rename_map)

    # -----------------------------
    # 7. Merge everything
    # -----------------------------
    final_df = rec_summary_df.merge(employee_info_df, on="employee_id", how="left")
    final_df = final_df.merge(skill_summary_df, on="employee_id", how="left")
    final_df = final_df.merge(experience_summary_df, on="employee_id", how="left")
    final_df = final_df.merge(history_summary_df, on="employee_id", how="left")
    final_df = final_df.merge(availability_summary_df, on="employee_id", how="left")
    final_df = final_df.merge(compatibility_summary_df, on="employee_id", how="left")

    # -----------------------------
    # 8. Fill blanks
    # -----------------------------
    fill_zero_cols = [
        "experience",
        "total_project_work",
        "avg_past_performance_score",
        "avg_allocation_percent",
        "avg_compatibility_score",
        "recommended_project_count",
        "avg_weighted_skill_match_score",
        "avg_related_skill_match_score",
        "avg_required_skill_experience",
        "avg_availability_fit_score",
        "avg_task_context_match_score",
        "avg_soft_skill_score",
        "predicted_positive_count",
    ]
    for col in fill_zero_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)

    if "skills" in final_df.columns:
        final_df["skills"] = final_df["skills"].fillna("")

    # -----------------------------
    # 9. Sorting
    # -----------------------------
    sort_cols = ["avg_score"]
    ascending = [False]

    if "avg_past_performance_score" in final_df.columns:
        sort_cols.append("avg_past_performance_score")
        ascending.append(False)

    if "recommended_project_count" in final_df.columns:
        sort_cols.append("recommended_project_count")
        ascending.append(False)

    final_df = final_df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

    # -----------------------------
    # 10. Reorder columns
    # -----------------------------
    final_columns = [
        "employee_id",
        "full_name",
        "department",
        "job_title",
        "seniority_level",
        "location",
        "employment_type",
        "primary_language",
        "status",
        "skills",
        "experience",
        "total_project_work",
        "recommended_project_count",
        "predicted_positive_count",
        "avg_score",
        "avg_past_performance_score",
        "avg_weighted_skill_match_score",
        "avg_related_skill_match_score",
        "avg_required_skill_experience",
        "avg_availability_fit_score",
        "avg_task_context_match_score",
        "avg_soft_skill_score",
        "avg_allocation_percent",
        "avg_compatibility_score",
    ]
    final_columns = [c for c in final_columns if c in final_df.columns]
    final_df = final_df[final_columns].copy()

    return final_df.head(top_n)


def save_top_employee_summary_excel(
    top_employee_df: pd.DataFrame,
    filename: str = "HAF-EPA_top_200_employees.xlsx",
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    top_employee_df.to_excel(output_path, index=False)
    return output_path


def save_employee_reference_csv(
    top_employee_df: pd.DataFrame,
    filename: str = "HAF-EPA_employee_reference.csv",
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    employee_reference_df = top_employee_df.copy()

    # employee free time calculation: availability = 100% - avg_allocation_percent
    employee_reference_df["availability"] = 100 - employee_reference_df["avg_allocation_percent"]

    #need to rename skills column to employee_skills for webapp compatibility
    employee_reference_df = employee_reference_df.rename(columns={
        "skills": "employee_skills"
    })

    # -----------------------------
    # Programming and management skill groups
    # -----------------------------
    # Programming score and management score must not use the same logic.
    # Programming score depends mainly on technical/programming skills.
    # Management score depends on management skills + experience + availability + project work.
    programming_skills = {
        "python", "java", "javascript", "typescript", "php", "laravel",
        "react", "angular", "vue", "node.js", "node", "django", "flask",
        "spring boot", "html", "css", "sql", "mysql", "postgresql",
        "mongodb", "docker", "kubernetes", "aws", "azure", "git",
        "linux", "machine learning", "deep learning", "data analysis",
        "data science", "ui/ux design", "go", "rust", "scala",
        "android", "ios", "tensorflow", "pytorch", "devops",
        "automation testing", "cybersecurity", "testing", "c++", "c#"
    }

    management_skills = {
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

    def parse_skill_list(value):
        if pd.isna(value):
            return []

        if isinstance(value, list):
            return [str(skill).strip() for skill in value if str(skill).strip()]

        text = str(value).strip()

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(skill).strip() for skill in parsed if str(skill).strip()]
        except Exception:
            pass

        return [
            str(skill).strip()
            for skill in text.split(",")
            if str(skill).strip()
        ]

    def count_category_skills(skill_text, category_set):
        skills = parse_skill_list(skill_text)

        return sum(
            1
            for skill in skills
            if skill.lower() in category_set
        )

    def percentage(part, total):
        if total == 0:
            return 0.0

        return round((part / total) * 100, 2)

    employee_reference_df["total_skill_count"] = employee_reference_df["employee_skills"].apply(
        lambda value: len(parse_skill_list(value))
    )

    employee_reference_df["programming_skill_count"] = employee_reference_df["employee_skills"].apply(
        lambda value: count_category_skills(value, programming_skills)
    )

    employee_reference_df["management_skill_count"] = employee_reference_df["employee_skills"].apply(
        lambda value: count_category_skills(value, management_skills)
    )

    employee_reference_df["programming_skill_percentage"] = employee_reference_df.apply(
        lambda row: percentage(row["programming_skill_count"], row["total_skill_count"]),
        axis=1,
    )

    employee_reference_df["management_skill_percentage"] = employee_reference_df.apply(
        lambda row: percentage(row["management_skill_count"], row["total_skill_count"]),
        axis=1,
    )

    # Estimated project working count comes from employee history summary.
    # If the column is missing, use 0 safely.
    if "total_project_work" not in employee_reference_df.columns:
        employee_reference_df["total_project_work"] = 0

    employee_reference_df["estimated_project_working_count"] = employee_reference_df[
        "total_project_work"
    ].fillna(0)

    # Programming score mostly depends on programming skills.
    employee_reference_df["programming_score"] = employee_reference_df[
        "programming_skill_percentage"
    ].round(2)

    # Management score should not be same as programming score.
    # It depends on management skills, experience, availability, and project history.
    employee_reference_df["experience_score"] = employee_reference_df["experience"].apply(
        lambda value: min((float(value) / 10) * 100, 100)
    )

    employee_reference_df["availability_score"] = employee_reference_df["availability"].apply(
        lambda value: max(0.0, min(float(value), 100.0))
    )

    employee_reference_df["project_work_score"] = employee_reference_df[
        "estimated_project_working_count"
    ].apply(
        lambda value: min((float(value) / 20) * 100, 100)
    )

    employee_reference_df["management_score"] = (
        employee_reference_df["management_skill_percentage"] * 0.45
        + employee_reference_df["experience_score"] * 0.25
        + employee_reference_df["availability_score"] * 0.15
        + employee_reference_df["project_work_score"] * 0.15
    ).round(2)

    reference_columns = [
        "employee_id",
        "full_name",
        "employee_skills",
        "experience",
        "availability",
        "total_skill_count",
        "programming_skill_count",
        "management_skill_count",
        "programming_skill_percentage",
        "management_skill_percentage",
        "estimated_project_working_count",
        "programming_score",
        "management_score",
    ]

    employee_reference_df = employee_reference_df[reference_columns].copy()

    output_path = OUTPUT_DIR / filename
    employee_reference_df.to_csv(output_path, index=False)

    return output_path