import pandas as pd

#Hybrid approach for single project use project ID
# -->ML and Graph each have strengths + weaknesses, and combining them gives better, more reliable recommendations.
# Generate hybrid employee recommendations for a single project by combining machine learning prediction scores with graph-based matching scores.

# ML says: “This employee usually performs well on similar projects”
# Graph says: “This employee has the required skills”
#“This employee both has the skills AND historically performs well”

def hybrid_recommendation(
    predicted_df: pd.DataFrame,
    graph_rec: pd.DataFrame,
    project_id: str,
    top_k: int = 5,
    ml_weight: float = 0.7,
    graph_weight: float = 0.3,
) -> pd.DataFrame:
    

    # 1. Filter ML prediction for the target project
    ml_df = predicted_df[predicted_df["project_id"] == project_id].copy()

    if ml_df.empty:
        return pd.DataFrame(
            columns=[
                "employee_id",
                "full_name",
                "project_id",
                "predicted_score",
                "graph_score",
                "hybrid_score",
            ]
        )

    # 2. Filter graph recommendation for the target project
    graph_df = graph_rec[graph_rec["project_id"] == project_id].copy()

    # 3. Normalize graph score to 0-1
    if not graph_df.empty:
        max_score = graph_df["match_score"].max()
        if max_score > 0:
            graph_df["graph_score"] = graph_df["match_score"] / max_score
        else:
            graph_df["graph_score"] = 0.0
    else:
        graph_df = pd.DataFrame(columns=["project_id", "employee_id", "graph_score"])

   # 4. Merge ML and KG
    merged = ml_df.merge(
        graph_df[["project_id", "employee_id", "graph_score"]],
        on=["project_id", "employee_id"],
        how="left",
    )

    merged["graph_score"] = pd.to_numeric(
        merged["graph_score"], errors="coerce"
    ).fillna(0.0)

    # 5. Hybrid score
    merged["hybrid_score"] = (
        merged["predicted_score"] * ml_weight +
        merged["graph_score"] * graph_weight
    )

    # 6. Sort and keep top_k
    merged = merged.sort_values(
        by="hybrid_score",
        ascending=False
    ).reset_index(drop=True)

    output_columns = [
        "employee_id",
        "full_name",
        "project_id",
        "predicted_score",
        "graph_score",
        "hybrid_score",
    ]

    available_columns = [col for col in output_columns if col in merged.columns]

    return merged[available_columns].head(top_k)