import pandas as pd


def kg_recommendation(nodes_df, edges_df, top_k=5):
    """
    Graph-based recommendation for ALL projects
    """

    edges_df = edges_df.copy()
    nodes_df = nodes_df.copy()

    # -------------------
    # Project → Skills
    # -------------------
    project_skill_map = (
        edges_df[edges_df["relation"] == "REQUIRES_SKILL"]
        .groupby("source")["target"]
        .apply(set)
        .to_dict()
    )

    # -------------------
    # Employee → Skills
    # -------------------
    employee_skill_map = (
        edges_df[edges_df["relation"] == "HAS_SKILL"]
        .groupby("source")["target"]
        .apply(set)
        .to_dict()
    )

    results = []

    # -------------------
    # Matching
    # -------------------
    for project_id, proj_skills in project_skill_map.items():

        for emp_id, emp_skills in employee_skill_map.items():

            match_score = len(proj_skills & emp_skills)

            if match_score > 0:
                results.append({
                    "project_id": project_id,
                    "employee_id": emp_id,
                    "match_score": match_score
                })

    # ✅ IMPORTANT FIX
    if len(results) == 0:
        return pd.DataFrame(columns=["project_id","project_name", "employee_id", "employee_name", "match_score"])

    result_df = pd.DataFrame(results)

    # -------------------
    # Add employee names
    # -------------------
    employee_nodes = nodes_df[nodes_df["node_type"] == "Employee"][
        ["node_id", "name"]
    ].copy()

    employee_nodes.columns = ["employee_id", "employee_name"]

    result_df = result_df.merge(employee_nodes, on="employee_id", how="left")

    # -------------------
    # Add project names
    # -------------------
    project_nodes = nodes_df[nodes_df["node_type"] == "Project"][
        ["node_id", "name"]
    ].copy()

    project_nodes.columns = ["project_id", "project_name"]

    result_df = result_df.merge(project_nodes, on="project_id", how="left")

    # -------------------
    # Sort + Top K per project
    # -------------------
    result_df = result_df.sort_values(
        ["project_id", "match_score"],
        ascending=[True, False]
    )

    result_df = result_df.groupby("project_id").head(top_k).reset_index(drop=True)
    result_df = result_df[["project_id", "project_name", "employee_id", "employee_name", "match_score"]]

    return result_df