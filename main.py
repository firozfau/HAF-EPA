from __future__ import annotations

from data_loader.load_datasets import load_datasets 
from process.normalize import normalize_datasets
from knowledge_graph.build import kg_build
from knowledge_graph.representation import kg_recommendation

from config import (KG_RECOMENDED_LIMIT,LIMIT_NUMBER,OUTPUT_DIR,KNOWLEDGE_RECOMMENDED_EXCEL)
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
    kgr_data = kg_recommendation(nodes_df=kg_nodes, edges_df=kg_edges,  top_k=KG_RECOMENDED_LIMIT)
   
    sorted_kgr_data = kgr_data.sort_values(["project_id", "match_score"], ascending=[True, False])
    group_kgr_data = sorted_kgr_data.groupby("project_id")
    top_emp_kgr_data = group_kgr_data.head(LIMIT_NUMBER);

    file_path = os.path.join(OUTPUT_DIR, KNOWLEDGE_RECOMMENDED_EXCEL)
    top_emp_kgr_data.to_excel(file_path, index=False)

    print("save:",KNOWLEDGE_RECOMMENDED_EXCEL)
   


if __name__ == "__main__":
    main()
