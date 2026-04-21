from data_loader.load_datasets import load_datasets
from process.normalize import normalize_loaded_data
from process.pair_creation import create_employee_project_pairs
from process.feature_engineering import add_features

# -------------------------
# Step 1: Load + Normalize
# -------------------------
data = load_datasets()
data = normalize_loaded_data(data)

# -------------------------
# Step 2: Pair Creation
# -------------------------
pairs = create_employee_project_pairs(
    employees_df=data.employees,
    projects_df=data.projects,
    employee_skills_df=data.employee_skills,
    project_skills_df=data.project_skills,
    skills_df=data.skills,
    tasks_df=data.tasks,
    employee_availability_df=data.employee_availability,
)

print("\n=== PAIRS CREATED ===")
print(pairs.shape)

# -------------------------
# Step 3: Feature Engineering
# -------------------------
features = add_features(
    pairs_df=pairs,
    employee_skills_df=data.employee_skills,
    project_skills_df=data.project_skills,
    skills_df=data.skills,
    employee_project_history_df=data.employee_project_history,
    employee_availability_df=data.employee_availability,
    employee_relationship_df=data.employee_relationship,
    skill_similarity_df=data.skill_similarity,
)

print("\n=== FEATURES CREATED ===")
print(features.shape)

print(
    features[
        [
            "employee_id",
            "project_id",
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
        ]
    ].head()
)

print("\nDONE STEP 5 ✅")