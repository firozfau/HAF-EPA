from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_1_DIR = DATASET_DIR / "dataset-1"
DATASET_2_DIR = DATASET_DIR / "dataset-2"
OUTPUT_DIR = BASE_DIR / "output"

 
RANDOM_STATE = 42               # Random state is used to ensure reproducibility of results by fixing randomness in data splitting and model training [Train-test split, Random Forest ,Sampling].
EMPLOYEE_SAMPLE_SIZE = 100      # Set Limit number of employees used in processing
PROJECT_SAMPLE_SIZE = 50        # Set Limit number of project used in processing
TEST_SIZE = 0.2                 # Split dataset into: [ 80% training and 20% testing] 
MAX_EXPERIENCE_YEARS = 20       # Normalize emplyee experience score, it fix max 20 year

KG_RECOMENDED_LIMIT = 10        # set recomended kg graph data  per project
LIMIT_NUMBER = 10                      # limit set any case  for number emp or number of project etc 

KNOWLEDGE_RECOMMENDED_EXCEL = "knowledge_recommended_data.xlsx"       



