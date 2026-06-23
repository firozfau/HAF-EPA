from __future__ import annotations
import pandas as pd


def _to_set(value) -> set:
    if isinstance(value, list):
        return set(v for v in value if pd.notna(v))
    if pd.isna(value):
        return set()
    return {value}


def _normalize_text_set(values: set) -> set:
    normalized = set()
    for v in values:
        if pd.isna(v):
            continue
        text = str(v).strip().lower()
        if text:
            normalized.add(text)
    return normalized


def _build_skill_name_to_id(skills_df: pd.DataFrame) -> dict:
    return dict(
        zip(
            skills_df["skill_name"].astype(str).str.strip().str.lower(),
            skills_df["skill_id"],
        )
    )


def _build_similarity_lookup(skill_similarity_df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in skill_similarity_df.iterrows():
        s1 = row["skill_id_1"]
        s2 = row["skill_id_2"]
        score = row["similarity_score"]

        if pd.isna(s1) or pd.isna(s2) or pd.isna(score):
            continue

        lookup[(str(s1).strip(), str(s2).strip())] = float(score)
    return lookup


def _employee_skill_experience_map(employee_skills_df: pd.DataFrame) -> dict:
    """
    Returns:
        {
            employee_id: {
                skill_id: years_experience
            }
        }
    """
    result = {}
    for _, row in employee_skills_df.iterrows():
        emp = row["employee_id"]
        skill = row["skill_id"]
        exp = row.get("years_experience", 0)

        if pd.isna(emp) or pd.isna(skill):
            continue

        result.setdefault(emp, {})
        result[emp][skill] = 0 if pd.isna(exp) else float(exp)

    return result


def _employee_performance_map(employee_project_history_df: pd.DataFrame) -> dict:
    """
    Returns average performance per employee
    """
    grouped = (
        employee_project_history_df.groupby("employee_id")["performance_score"]
        .mean()
        .reset_index()
    )
    return dict(zip(grouped["employee_id"], grouped["performance_score"]))


def _employee_soft_skill_map(employee_relationship_df: pd.DataFrame) -> dict:
    """
    Returns average compatibility per employee
    """
    long_rows = []

    for _, row in employee_relationship_df.iterrows():
        e1 = row.get("employee_id_1")
        e2 = row.get("employee_id_2")
        score = row.get("compatibility_score")

        if pd.isna(e1) or pd.isna(e2) or pd.isna(score):
            continue

        long_rows.append({"employee_id": e1, "compatibility_score": score})
        long_rows.append({"employee_id": e2, "compatibility_score": score})

    if not long_rows:
        return {}

    rel_df = pd.DataFrame(long_rows)
    grouped = (
        rel_df.groupby("employee_id")["compatibility_score"]
        .mean()
        .reset_index()
    )
    return dict(zip(grouped["employee_id"], grouped["compatibility_score"]))


def _calculate_related_skill_score(
    employee_skill_names: set,
    project_skill_names: set,
    skill_name_to_id: dict,
    similarity_lookup: dict,
) -> float:
    """
    Exact matches are not counted here.
    Only related skill matches based on skill_similarity table.
    """
    employee_skill_ids = []
    project_skill_ids = []

    for skill in employee_skill_names:
        sid = skill_name_to_id.get(skill)
        if sid:
            employee_skill_ids.append(sid)

    for skill in project_skill_names:
        sid = skill_name_to_id.get(skill)
        if sid:
            project_skill_ids.append(sid)

    related_scores = []

    for emp_sid in employee_skill_ids:
        for proj_sid in project_skill_ids:
            if emp_sid == proj_sid:
                continue
            sim = similarity_lookup.get((emp_sid, proj_sid), 0)
            if sim > 0:
                related_scores.append(sim)

    if not related_scores:
        return 0.0

    return float(sum(related_scores) / len(related_scores))


def _task_context_match_score(employee_skill_names: set, task_contexts: set) -> float:
    """
    Very simple text-based context match.
    If any employee skill text appears inside task context text, count as match.
    """
    if not task_contexts:
        return 0.0

    context_text = " | ".join(task_contexts).lower()
    matches = 0

    for skill in employee_skill_names:
        if skill in context_text:
            matches += 1

    if len(employee_skill_names) == 0:
        return 0.0

    return matches / len(employee_skill_names)


def add_features(
    pairs_df: pd.DataFrame,
    employee_skills_df: pd.DataFrame,
    project_skills_df: pd.DataFrame,
    skills_df: pd.DataFrame,
    employee_project_history_df: pd.DataFrame,
    employee_availability_df: pd.DataFrame,
    employee_relationship_df: pd.DataFrame,
    skill_similarity_df: pd.DataFrame,
) -> pd.DataFrame:
    df = pairs_df.copy()

    skill_name_to_id = _build_skill_name_to_id(skills_df)
    similarity_lookup = _build_similarity_lookup(skill_similarity_df)
    employee_exp_map = _employee_skill_experience_map(employee_skills_df)
    employee_perf_map = _employee_performance_map(employee_project_history_df)
    employee_soft_map = _employee_soft_skill_map(employee_relationship_df)

    # -----------------------------
    # Basic match features
    # -----------------------------
    matched_skill_count_list = []
    matched_required_skill_count_list = []
    matched_optional_skill_count_list = []
    employee_skill_count_list = []
    project_skill_count_list = []
    required_skill_count_list = []
    optional_skill_count_list = []
    skill_match_score_list = []
    employee_skill_coverage_list = []
    missing_required_skill_count_list = []
    has_any_skill_match_list = []
    strong_skill_match_list = []
    weighted_skill_match_score_list = []
    related_skill_match_score_list = []
    avg_experience_on_required_skills_list = []
    avg_past_performance_score_list = []
    availability_fit_score_list = []
    task_context_match_score_list = []
    soft_skill_compatibility_score_list = []

    for _, row in df.iterrows():
        employee_id = row["employee_id"]

        employee_skills = _normalize_text_set(_to_set(row.get("employee_skills", [])))
        project_skills = _normalize_text_set(_to_set(row.get("project_skills", [])))
        required_project_skills = _normalize_text_set(
            _to_set(row.get("required_project_skills", []))
        )
        optional_project_skills = _normalize_text_set(
            _to_set(row.get("optional_project_skills", []))
        )
        task_contexts = _normalize_text_set(
            _to_set(row.get("task_required_skill_context", []))
        )

        exact_matches = employee_skills.intersection(project_skills)
        required_matches = employee_skills.intersection(required_project_skills)
        optional_matches = employee_skills.intersection(optional_project_skills)

        matched_skill_count = len(exact_matches)
        matched_required_skill_count = len(required_matches)
        matched_optional_skill_count = len(optional_matches)

        employee_skill_count = len(employee_skills)
        project_skill_count = len(project_skills)
        required_skill_count = len(required_project_skills)
        optional_skill_count = len(optional_project_skills)

        skill_match_score = (
            matched_skill_count / project_skill_count if project_skill_count > 0 else 0.0
        )
        employee_skill_coverage = (
            matched_skill_count / employee_skill_count if employee_skill_count > 0 else 0.0
        )

        missing_required_skill_count = max(
            required_skill_count - matched_required_skill_count, 0
        )

        has_any_skill_match = 1 if matched_skill_count > 0 else 0
        strong_skill_match = 1 if skill_match_score >= 0.5 else 0

        weighted_skill_match_score = 0.0
        if required_skill_count > 0 or optional_skill_count > 0:
            required_part = (
                matched_required_skill_count / required_skill_count
                if required_skill_count > 0
                else 0.0
            )
            optional_part = (
                matched_optional_skill_count / optional_skill_count
                if optional_skill_count > 0
                else 0.0
            )
            weighted_skill_match_score = (0.7 * required_part) + (0.3 * optional_part)

        related_skill_match_score = _calculate_related_skill_score(
            employee_skill_names=employee_skills,
            project_skill_names=project_skills,
            skill_name_to_id=skill_name_to_id,
            similarity_lookup=similarity_lookup,
        )

        # skill-specific experience on required skills
        employee_skill_exp = employee_exp_map.get(employee_id, {})
        required_skill_exp_values = []

        for req_skill_name in required_project_skills:
            req_skill_id = skill_name_to_id.get(req_skill_name)
            if req_skill_id and req_skill_id in employee_skill_exp:
                required_skill_exp_values.append(employee_skill_exp[req_skill_id])

        avg_experience_on_required_skills = (
            sum(required_skill_exp_values) / len(required_skill_exp_values)
            if required_skill_exp_values
            else 0.0
        )

        # average past performance
        avg_past_performance_score = employee_perf_map.get(employee_id, 0.0)
        if pd.isna(avg_past_performance_score):
            avg_past_performance_score = 0.0

        # availability
        allocation_percent = row.get("allocation_percent", 0)
        if pd.isna(allocation_percent):
            allocation_percent = 0.0

        # lower allocation = more available
        availability_fit_score = max(0.0, 1.0 - (float(allocation_percent) / 100.0))

        # task context
        task_context_score = _task_context_match_score(employee_skills, task_contexts)

        # soft skill
        soft_skill_score = employee_soft_map.get(employee_id, 0.0)
        if pd.isna(soft_skill_score):
            soft_skill_score = 0.0

        matched_skill_count_list.append(matched_skill_count)
        matched_required_skill_count_list.append(matched_required_skill_count)
        matched_optional_skill_count_list.append(matched_optional_skill_count)
        employee_skill_count_list.append(employee_skill_count)
        project_skill_count_list.append(project_skill_count)
        required_skill_count_list.append(required_skill_count)
        optional_skill_count_list.append(optional_skill_count)
        skill_match_score_list.append(skill_match_score)
        employee_skill_coverage_list.append(employee_skill_coverage)
        missing_required_skill_count_list.append(missing_required_skill_count)
        has_any_skill_match_list.append(has_any_skill_match)
        strong_skill_match_list.append(strong_skill_match)
        weighted_skill_match_score_list.append(weighted_skill_match_score)
        related_skill_match_score_list.append(related_skill_match_score)
        avg_experience_on_required_skills_list.append(avg_experience_on_required_skills)
        avg_past_performance_score_list.append(avg_past_performance_score)
        availability_fit_score_list.append(availability_fit_score)
        task_context_match_score_list.append(task_context_score)
        soft_skill_compatibility_score_list.append(soft_skill_score)

    df["matched_skill_count"] = matched_skill_count_list
    df["matched_required_skill_count"] = matched_required_skill_count_list
    df["matched_optional_skill_count"] = matched_optional_skill_count_list
    df["employee_skill_count"] = employee_skill_count_list
    df["project_skill_count"] = project_skill_count_list
    df["required_skill_count"] = required_skill_count_list
    df["optional_skill_count"] = optional_skill_count_list
    df["skill_match_score"] = skill_match_score_list
    df["employee_skill_coverage"] = employee_skill_coverage_list
    df["missing_required_skill_count"] = missing_required_skill_count_list
    df["has_any_skill_match"] = has_any_skill_match_list
    df["strong_skill_match"] = strong_skill_match_list
    df["weighted_skill_match_score"] = weighted_skill_match_score_list
    df["related_skill_match_score"] = related_skill_match_score_list
    df["avg_experience_on_required_skills"] = avg_experience_on_required_skills_list
    df["avg_past_performance_score"] = avg_past_performance_score_list
    df["availability_fit_score"] = availability_fit_score_list
    df["task_context_match_score"] = task_context_match_score_list
    df["soft_skill_compatibility_score"] = soft_skill_compatibility_score_list

    return df