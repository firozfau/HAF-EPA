from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "label"


def split_train_test(
    labeled_df: pd.DataFrame,
    feature_columns: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
):
    required_columns = feature_columns + [TARGET_COLUMN]
    missing = [c for c in required_columns if c not in labeled_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for split: {missing}")

    model_df = labeled_df.dropna(subset=required_columns).reset_index(drop=True)

    train_df, test_df = train_test_split(
        model_df,
        test_size=test_size,
        random_state=random_state,
        stratify=model_df[TARGET_COLUMN],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)