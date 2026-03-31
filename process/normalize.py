import pandas as pd


# normalize_datasets is basically used to clean and prepare all raw datasets
# before using them in mapping, feature engineering, training, or recommendation.

# 1. First, it checks whether all required columns are available in each dataset.
#    If any important column is missing, it raises an error.

# 2. Then it converts all ID columns into string format,
#    so all IDs stay consistent for matching and merging.

# 3. After that, it cleans text columns by removing extra spaces.

# 4. It removes rows where important ID fields are blank, empty, or invalid.

# 5. Then it removes duplicate rows from each dataset
#    to avoid repeated or incorrect data.

# 6. It keeps only valid skill links,
#    meaning employee_skills and project_skills must match valid skill_id from skills table.

# 7. It also keeps only valid task, employee, and project links,
#    so no broken relationship stays in the data.

# Finally, it returns all cleaned and normalized datasets
# in a structured format for next processing steps.

REQUIRED_COLUMNS = {
    "employees": ["employee_id"],
    "projects": ["project_id","project_name"],
    "tasks": ["task_id", "employee_id", "project_id"],
    "employee_skills": ["employee_id", "skill_id"],
    "project_skills": ["project_id", "skill_id"],
    "skills": ["skill_id", "skill_name"],
}


def _validate_columns(name: str, df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS[name] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")


def _convert_id_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    id_columns = [col for col in df.columns if col.endswith("_id")]
    for col in id_columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    text_columns = df.select_dtypes(include=["object"]).columns
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def _drop_blank_ids(df: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    for col in key_columns:
        df = df[df[col].notna()]
        df = df[df[col].astype(str).str.strip() != ""]
        df = df[df[col].astype(str).str.lower() != "nan"]
    return df


def normalize_datasets(data):

    employees = data.employees.copy()
    projects = data.projects.copy()
    tasks = data.tasks.copy()
    employee_skills = data.employee_skills.copy()
    project_skills = data.project_skills.copy()
    skills = data.skills.copy()

    # 1. Validate required columns
    _validate_columns("employees", employees)
    _validate_columns("projects", projects)
    _validate_columns("tasks", tasks)
    _validate_columns("employee_skills", employee_skills)
    _validate_columns("project_skills", project_skills)
    _validate_columns("skills", skills)

    # 2. Convert IDs to strings
    employees = _convert_id_columns_to_string(employees)
    projects = _convert_id_columns_to_string(projects)
    tasks = _convert_id_columns_to_string(tasks)
    employee_skills = _convert_id_columns_to_string(employee_skills)
    project_skills = _convert_id_columns_to_string(project_skills)
    skills = _convert_id_columns_to_string(skills)

    # 3. Clean text columns
    employees = _clean_text_columns(employees)
    projects = _clean_text_columns(projects)
    tasks = _clean_text_columns(tasks)
    employee_skills = _clean_text_columns(employee_skills)
    project_skills = _clean_text_columns(project_skills)
    skills = _clean_text_columns(skills)

    # 4. Drop blank IDs
    employees = _drop_blank_ids(employees, ["employee_id"])
    projects = _drop_blank_ids(projects, ["project_id","project_name"])
    tasks = _drop_blank_ids(tasks, ["task_id", "employee_id", "project_id"])
    employee_skills = _drop_blank_ids(employee_skills, ["employee_id", "skill_id"])
    project_skills = _drop_blank_ids(project_skills, ["project_id", "skill_id"])
    skills = _drop_blank_ids(skills, ["skill_id"])

    # 5. Remove duplicates
    employees = employees.drop_duplicates(subset=["employee_id"])
    projects = projects.drop_duplicates(subset=["project_id","project_name"])
    tasks = tasks.drop_duplicates(subset=["task_id"])
    employee_skills = employee_skills.drop_duplicates(subset=["employee_id", "skill_id"])
    project_skills = project_skills.drop_duplicates(subset=["project_id", "skill_id"])
    skills = skills.drop_duplicates(subset=["skill_id"])

    # 6. Keep valid skill links
    valid_skill_ids = set(skills["skill_id"])
    employee_skills = employee_skills[employee_skills["skill_id"].isin(valid_skill_ids)]
    project_skills = project_skills[project_skills["skill_id"].isin(valid_skill_ids)]

    # 7. Keep valid task/entity links
    valid_employee_ids = set(employees["employee_id"])
    valid_project_ids = set(projects["project_id"])

    tasks = tasks[
        tasks["employee_id"].isin(valid_employee_ids) &
        tasks["project_id"].isin(valid_project_ids)
    ]

    employee_skills = employee_skills[employee_skills["employee_id"].isin(valid_employee_ids)]
    project_skills = project_skills[project_skills["project_id"].isin(valid_project_ids)]

    return {
        "employees": employees.reset_index(drop=True),
        "projects": projects.reset_index(drop=True),
        "tasks": tasks.reset_index(drop=True),
        "employee_skills": employee_skills.reset_index(drop=True),
        "project_skills": project_skills.reset_index(drop=True),
        "skills": skills.reset_index(drop=True),
    }