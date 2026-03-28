import pandas as pd

# Generate top-K employee recommendations per project based on predicted score.

def generate_final_recommendation(predicted_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
 
    # Sort by project and score
    sorted_df = predicted_df.sort_values(
        ["project_id", "predicted_score"],
        ascending=[True, False]
    )

    # Group by project
    grouped = sorted_df.groupby("project_id")

    # Take top K per project
    top_recommendations = grouped.head(top_k).reset_index(drop=True)

    return top_recommendations