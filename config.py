from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
 
TRAINING_DATASET_DIR = DATASET_DIR / "training-dataset"
TEST_DATASET_DIR = DATASET_DIR / "test-dataset"
 
OUTPUT_DIR = BASE_DIR / "output"

RANDOM_STATE = 42 
TEST_SIZE = 0.2

TRAINED_MODEL = OUTPUT_DIR / "HAF-EPA.joblib"
HELD_OUT_TEST_DATA = OUTPUT_DIR / "HAF-EPA-TEST.joblib"
