from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_1_DIR = DATASET_DIR / "dataset-1"
DATASET_2_DIR = DATASET_DIR / "dataset-2"
OUTPUT_DIR = BASE_DIR / "output"

 
RANDOM_STATE = 42               # Ensures reproducibility (used in train-test split, models, and sampling)

EMPLOYEE_SAMPLE_SIZE = 100      # Maximum number of employees to include in processing
PROJECT_SAMPLE_SIZE = 50        # Maximum number of projects to include in processing

TEST_SIZE = 0.2                 # Proportion of dataset used for testing (20% test, 80% train)

MAX_EXPERIENCE_YEARS = 20       # Upper bound for normalizing employee experience (caps at 20 years)

KG_RECOMMENDED_LIMIT = 10       # Maximum number of KG-based recommendations per project

LIMIT_NUMBER = 10               # Generic limit for number of items (e.g., employees, projects, etc.)


KNOWLEDGE_RECOMMENDED_EXCEL = "knowledge_recommended_data.xlsx"       
TRAINED_MODEL = OUTPUT_DIR/"HAF-EPA.joblib"


