from __future__ import annotations

from data_loader.load_datasets import load_datasets 
from process.normalize import normalize_datasets
from knowledge_graph.build import kg_build
from knowledge_graph.representation import kg_recommendation
from process.employee_skill_mapping import emp_map_build
from process.project_skill_mapping import project_map_build
from process.mapping import employee_map,project_map
from process.pair_creation import build_pairs_employee_project
from process.feature_engineering import add_features
from process.lebel_employee_project import add_labels
from models.train_model import train_random_forest_model
from models.process import save_model,load_model
from models.predict import predict_result


from config import (KG_RECOMMENDED_LIMIT,
                    LIMIT_NUMBER,OUTPUT_DIR,
                    KNOWLEDGE_RECOMMENDED_EXCEL,
                    EMPLOYEE_SAMPLE_SIZE,
                    PROJECT_SAMPLE_SIZE,
                    RANDOM_STATE,
                    TRAINED_MODEL
                    )
import os


def main() -> None:
    print("HAF-EPA Pipeline Starting...")
    
    # 1. Load raw datasets
    data = load_datasets() 

    # 2. Normalize datasets
    normalized_data = normalize_datasets(data)
    employees = normalized_data["employees"]
    projects = normalized_data["projects"]
    tasks = normalized_data["tasks"]
    employee_skills = normalized_data["employee_skills"]
    project_skills = normalized_data["project_skills"]
    skills = normalized_data["skills"]

     # 3. represet knowledge graph from datasets
    kg_nodes,  kg_edges = kg_build(
        employees,
        projects,
        tasks,
        employee_skills,
        project_skills,
        skills,
    )

    print("\n--- Projects sample (check employee_id) ---")  
    kgr_data = kg_recommendation(nodes_df=kg_nodes, edges_df=kg_edges,  top_k=KG_RECOMMENDED_LIMIT)
   
    sorted_kgr_data = kgr_data.sort_values(["project_id", "match_score"], ascending=[True, False])
    group_kgr_data = sorted_kgr_data.groupby("project_id")
    top_emp_kgr_data = group_kgr_data.head(LIMIT_NUMBER);

    file_path = os.path.join(OUTPUT_DIR, KNOWLEDGE_RECOMMENDED_EXCEL)
    top_emp_kgr_data.to_excel(file_path, index=False)

    print("save:",KNOWLEDGE_RECOMMENDED_EXCEL)
   
    # 4. Skill mapping
    employee_skill_map = emp_map_build(employee_skills, skills)
    project_skill_map = project_map_build(projects, project_skills, skills)
   
    employees_enriched = employee_map(data.employees, employee_skill_map)
    projects_enriched = project_map(data.projects, project_skill_map)

    # 5. Sampling

    #a. employees sample 
    employees_sample = employees_enriched.sample(
        n=min(EMPLOYEE_SAMPLE_SIZE, len(employees_enriched)),
        random_state=RANDOM_STATE
    )

    #b. filter projects with skills
    projects_with_skills = projects_enriched[
        projects_enriched["skill_name"].apply(len) > 0
    ]

    #c. projects sample 
    projects_sample = projects_with_skills.sample(
        n=min(PROJECT_SAMPLE_SIZE, len(projects_with_skills)),
        random_state=RANDOM_STATE
    )

    #6. Employee-project pair creation
    #pairs_df = all employee-project combinations + their data (features)
    empp_pair_DataFrame = build_pairs_employee_project(employees_sample, projects_sample)

    #7. Feature engineering

    featured_DataFrame = add_features(empp_pair_DataFrame)
    FEATURE_COLUMNS_1 = featured_DataFrame[["employee_id", "project_id","employee_skills", "project_skills","skill_match_score"]].head()
    FEATURE_COLUMNS_2 = featured_DataFrame[["employee_id","project_id","skill_match_score","experience","experience_score","availability","availability_score"]].head()
 
    #8. Label employee-project pair data
    labeled_DataFrame = add_labels(featured_DataFrame, data.tasks)
    input_features = labeled_DataFrame[["skill_match_score","experience_score","availability_score",]]
    target_label = labeled_DataFrame["label"]

     #9. Sice whe are ready, Train model
    trained_model_data = train_random_forest_model(labeled_DataFrame)
    save_model(trained_model_data.model, TRAINED_MODEL)
    
    predicted_DataFrame = predict_result(labeled_DataFrame,trained_model_data.model)
    prediction_sample = predicted_DataFrame[["employee_id","project_id","skill_match_score", "predicted_score",]].head()
    print(prediction_sample)


if __name__ == "__main__":
    main()
