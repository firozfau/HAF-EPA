from pathlib import Path
import joblib


def save_model(model, filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(filepath: str):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {filepath}")
    return joblib.load(path)