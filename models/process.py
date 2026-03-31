from pathlib import Path
import joblib

# save_model and load_model are used to store and retrieve the trained machine learning model.

# save_model saves the trained model into a file for future use.
# load_model loads the saved model from file so it can be used for prediction.

def save_model(model, filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(filepath: str):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {filepath}")
    return joblib.load(path)