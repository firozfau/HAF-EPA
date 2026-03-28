from __future__ import annotations

import pandas as pd

from models.train_model import FEATURE_COLUMNS


def predict_result(df: pd.DataFrame, model) -> pd.DataFrame:
    predicted_df = df.copy()
    X = predicted_df[FEATURE_COLUMNS]
    predicted_df["predicted_score"] = model.predict_proba(X)[:, 1]
    return predicted_df
