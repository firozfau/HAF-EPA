from __future__ import annotations

import pandas as pd


def hybrid_recommendation(
    predicted_df: pd.DataFrame,
    graph_rec: pd.DataFrame,
    project_id: str,
    top_k: int = 5,
    ml_weight: float = 0.7,
    graph_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Hybrid recommendation for one project using OUTER merge.
    """

    ml_df = predicted_df[predicted_df["project_id"] == project_id].copy()
    graph_df = graph_rec[graph_rec["project_id"] == project_id].copy()

    if ml_df.empty and graph_df.empty:
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

    if not ml_df.empty:
        ml_keep_cols = [c for c in ["employee_id", "full_name", "project_id", "predicted_score"] if c in ml_df.columns]
        ml_df = ml_df[ml_keep_cols].copy()
        if "full_name" not in ml_df.columns:
            ml_df["full_name"] = None
    else:
        ml_df = pd.DataFrame(columns=["employee_id", "full_name", "project_id", "predicted_score"])

    if not graph_df.empty:
        max_score = graph_df["match_score"].max()
        if max_score > 0:
            graph_df["graph_score"] = graph_df["match_score"] / max_score
        else:
            graph_df["graph_score"] = 0.0

        graph_df = graph_df[["employee_id", "employee_name", "project_id", "graph_score"]].copy()
        graph_df = graph_df.rename(columns={"employee_name": "full_name"})
    else:
        graph_df = pd.DataFrame(columns=["employee_id", "full_name", "project_id", "graph_score"])

    merged = pd.merge(
        ml_df,
        graph_df,
        on=["employee_id", "project_id"],
        how="outer",
        suffixes=("_ml", "_kg"),
    )

    if "full_name_ml" in merged.columns and "full_name_kg" in merged.columns:
        merged["full_name"] = merged["full_name_ml"].fillna(merged["full_name_kg"])
        merged = merged.drop(columns=["full_name_ml", "full_name_kg"])
    elif "full_name_ml" in merged.columns:
        merged["full_name"] = merged["full_name_ml"]
        merged = merged.drop(columns=["full_name_ml"])
    elif "full_name_kg" in merged.columns:
        merged["full_name"] = merged["full_name_kg"]
        merged = merged.drop(columns=["full_name_kg"])

    merged["predicted_score"] = pd.to_numeric(
        merged.get("predicted_score"), errors="coerce"
    ).fillna(0.0)

    merged["graph_score"] = pd.to_numeric(
        merged.get("graph_score"), errors="coerce"
    ).fillna(0.0)

    merged["hybrid_score"] = (
        merged["predicted_score"] * ml_weight
        + merged["graph_score"] * graph_weight
    )

    merged = merged.sort_values(
        by=["hybrid_score", "predicted_score", "graph_score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return merged[
        ["employee_id", "full_name", "project_id", "predicted_score", "graph_score", "hybrid_score"]
    ].head(top_k)