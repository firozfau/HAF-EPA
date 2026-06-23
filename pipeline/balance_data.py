from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "label"


def balance_training_data(
    train_df: pd.DataFrame,
    negative_multiplier: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    positive_df = train_df[train_df[TARGET_COLUMN] == 1].copy()
    negative_df = train_df[train_df[TARGET_COLUMN] == 0].copy()

    if positive_df.empty:
        raise ValueError("No positive samples found in training set.")

    negative_sample_size = min(len(negative_df), len(positive_df) * negative_multiplier)

    sampled_negative_df = negative_df.sample(
        n=negative_sample_size,
        random_state=random_state,
    )

    balanced_df = pd.concat([positive_df, sampled_negative_df], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return balanced_df