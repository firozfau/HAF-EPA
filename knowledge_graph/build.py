import pandas as pd
from typing import Tuple


def kg_build(
    employees: pd.DataFrame,
    projects: pd.DataFrame,
    tasks: pd.DataFrame,
    employee_skills: pd.DataFrame,
    project_skills: pd.DataFrame,
    skills: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build knowledge graph nodes and edges from normalized datasets.
    """

    nodes = []
    edges = []

    employee_has_full_name = "full_name" in employees.columns
    project_has_project_name = "project_name" in projects.columns
    skill_has_skill_name = "skill_name" in skills.columns

    # Employee nodes
    for _, row in employees.iterrows():
        name = row["full_name"] if employee_has_full_name else row["employee_id"]
        nodes.append(
            {
                "node_id": row["employee_id"],
                "node_type": "Employee",
                "name": name,
            }
        )

    # Project nodes
    for _, row in projects.iterrows():
        name = row["project_name"] if project_has_project_name else row["project_id"]
        nodes.append(
            {
                "node_id": row["project_id"],
                "node_type": "Project",
                "name": name,
            }
        )

    # Skill nodes
    for _, row in skills.iterrows():
        name = row["skill_name"] if skill_has_skill_name else row["skill_id"]
        nodes.append(
            {
                "node_id": row["skill_id"],
                "node_type": "Skill",
                "name": name,
            }
        )

    # Employee -> Skill edges
    for _, row in employee_skills.iterrows():
        edges.append(
            {
                "source": row["employee_id"],
                "target": row["skill_id"],
                "relation": "HAS_SKILL",
            }
        )

    # Project -> Skill edges
    for _, row in project_skills.iterrows():
        edges.append(
            {
                "source": row["project_id"],
                "target": row["skill_id"],
                "relation": "REQUIRES_SKILL",
            }
        )

    # Employee -> Project edges
    task_pairs = tasks[["employee_id", "project_id"]].drop_duplicates()
    for _, row in task_pairs.iterrows():
        edges.append(
            {
                "source": row["employee_id"],
                "target": row["project_id"],
                "relation": "WORKED_ON",
            }
        )

    nodes_df = pd.DataFrame(nodes).drop_duplicates(subset=["node_id"]).reset_index(drop=True)
    edges_df = pd.DataFrame(edges).drop_duplicates().reset_index(drop=True)

    return nodes_df, edges_df