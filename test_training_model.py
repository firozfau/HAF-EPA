from data_loader.load_datasets import load_datasets
from process.normalize import normalize_loaded_data
from process.pair_creation import create_employee_project_pairs
from process.feature_engineering import add_features
from process.lebel_employee_project import add_labels
from models.train_model import train_model


# -----------------------------------
# Step 1: Load + Normalize
# -----------------------------------
print("\n=== STEP 1: LOAD + NORMALIZE ===")

data = load_datasets()
data = normalize_loaded_data(data)

print("Employees loaded:", data.employees.shape)
print("Skills loaded:", data.skills.shape)
print("History loaded:", data.employee_project_history.shape)


# -----------------------------------
# Step 2: Pair Creation
# -----------------------------------
print("\n=== STEP 2: PAIR CREATION ===")

pairs = create_employee_project_pairs(
    employees_df=data.employees,
    projects_df=data.projects,
    employee_skills_df=data.employee_skills,
    project_skills_df=data.project_skills,
    skills_df=data.skills,
    tasks_df=data.tasks,
    employee_availability_df=data.employee_availability,
)

print("Pairs shape:", pairs.shape)

# Bigger sample so we get more positive labels
pairs = pairs.sample(20000, random_state=42)

print("Sampled pairs shape:", pairs.shape)


# -----------------------------------
# Step 3: Feature Engineering
# -----------------------------------
print("\n=== STEP 3: FEATURE ENGINEERING ===")

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

print("Features shape:", features.shape)

print("\nSample feature columns:")
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


# -----------------------------------
# Step 4: Label Creation
# -----------------------------------
print("\n=== STEP 4: LABEL GENERATION ===")

labeled = add_labels(
    features_df=features,
    employee_project_history_df=data.employee_project_history,
    performance_threshold=7.0,
)

print("Labeled shape:", labeled.shape)

print("\nSample labeled data:")
print(
    labeled[
        [
            "employee_id",
            "project_id",
            "avg_past_performance_score",
            "label",
        ]
    ].head(10)
)

print("\nOriginal label distribution:")
print(labeled["label"].value_counts())


# -----------------------------------
# Step 5: Final Check Before Training
# -----------------------------------
print("\n=== STEP 5: FINAL CHECK ===")

assert "label" in labeled.columns
assert "weighted_skill_match_score" in labeled.columns
assert "related_skill_match_score" in labeled.columns
assert "avg_past_performance_score" in labeled.columns

print("All checks passed ✅")


# -----------------------------------
# Step 6: Balanced Model Training
# -----------------------------------
print("\n=== STEP 6: BALANCED MODEL TRAINING ===")

model, feature_importance, balanced_df = train_model(
    df=labeled,
    negative_multiplier=3,
    random_state=42,
)

print("\nBalanced dataset preview:")
print(
    balanced_df[
        [
            "employee_id",
            "project_id",
            "label",
            "weighted_skill_match_score",
            "related_skill_match_score",
            "avg_past_performance_score",
        ]
    ].head(10)
)

print("\nBalanced label distribution (returned dataframe):")
print(balanced_df["label"].value_counts())

print("\nTop 10 important features:")
print(feature_importance.head(10))

print("\nPIPELINE SUCCESSFULLY COMPLETED 🚀")