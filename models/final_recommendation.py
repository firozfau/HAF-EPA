import pandas as pd

# This function is used to generate final top employee recommendations for each project.

# 1. It sorts the data by project_id and predicted_score (highest score first).
# 2. Then it groups the data by project.
# 3. For each project, it selects top K employees based on score.
# 4. Finally, it returns the top recommended employees for each project.

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