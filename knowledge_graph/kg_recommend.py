from __future__ import annotations

from data_loader.load_datasets import load_datasets
from process.normalize import normalize_datasets
from knowledge_graph.build import kg_build
from knowledge_graph.representation import kg_recommendation

from config import (
    KG_RECOMMENDED_LIMIT,
    LIMIT_NUMBER,  
) 
#work flow=> Employee → Skills → Projects → Relationships

def kg_recommend() -> None:
   
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

    # 3. Build knowledge graph
    kg_nodes, kg_edges = kg_build(
        employees,
        projects,
        tasks,
        employee_skills,
        project_skills,
        skills,
    )

    # 4. KG recommendation
    kgr_data = kg_recommendation(
        nodes_df=kg_nodes,
        edges_df=kg_edges,
        top_k=KG_RECOMMENDED_LIMIT
    )

    sorted_kgr_data = kgr_data.sort_values(
        ["project_id", "match_score"],
        ascending=[True, False]
    )

    group_kgr_data = sorted_kgr_data.groupby("project_id")
    top_emp_kgr_data = group_kgr_data.head(LIMIT_NUMBER)

    return top_emp_kgr_data