from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_1_DIR = DATASET_DIR / "dataset-1"
DATASET_2_DIR = DATASET_DIR / "dataset-2"
TEST_DATASET_DIR = DATASET_DIR / "test-dataset"
OUTPUT_DIR = BASE_DIR / "output"

RANDOM_STATE = 42 
TEST_SIZE = 0.2
MAX_EXPERIENCE_YEARS = 20
KG_RECOMMENDED_LIMIT = 10
LIMIT_NUMBER = 10

KNOWLEDGE_RECOMMENDED_EXCEL = OUTPUT_DIR / "knowledge_recommended_data.xlsx"
FINAL_RECOMMENDED_EXCEL = OUTPUT_DIR / "final_recommendations.xlsx"
HYBRID_RECOMMENDED_EXCEL = OUTPUT_DIR / "hybrid_recommendations.xlsx"

TRAINED_MODEL = OUTPUT_DIR / "HAF-EPA.joblib"
HELD_OUT_TEST_DATA = OUTPUT_DIR / "held_out_test_data.joblib"

TRAINING_THRESHOLD = 0.35