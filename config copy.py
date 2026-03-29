from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
DATASET_1_DIR = DATASET_DIR / "dataset-1"
DATASET_2_DIR = DATASET_DIR / "dataset-2"
OUTPUT_DIR = BASE_DIR / "output"

 
RANDOM_STATE = 42               # Ensures reproducibility ,  42 → then random numbers are generated following a pattern based on 42, The first random number comes from 42, and the subsequent ones depend on the previous value. exmple: r1 = f(42)  then r2 = f(r1) then r3 = f(r2)


EMPLOYEE_SAMPLE_SIZE = 100      # Maximum number of employees to include in processing
PROJECT_SAMPLE_SIZE = 50        # Maximum number of projects to include in processing

TEST_SIZE = 0.2                 # Proportion of dataset used for testing (20% test, 80% train)

MAX_EXPERIENCE_YEARS = 20       # Upper bound for normalizing employee experience (caps at 20 years)

KG_RECOMMENDED_LIMIT = 10       # Maximum number of KG-based recommendations per project

LIMIT_NUMBER = 10               # Generic limit for number of items (e.g., employees, projects, etc.)


#RandomForestClassifier
NUMBER_OF_ESTIMATORS = 400              # How many decision tree will be held 
MAX_DEPTH = 12                          # How many level will allow in single Tree
MINIMUM_SAMPLES_SPLIT = 6               # A node must have at least 6 samples before it can be split.
MINIMUM_SAMPLES_LEAF = 2                # A leaf node must have at least 2 samples
CLASS_WEIGHT = "balanced_subsample"     # Handle Class imbalance   
N_JOBS = -1                             # It will use the CPU for parallel processing.
TRAING_THRESHOLD = 0.35                 # The threshold for deciding the final class is set to 0.35. it define predictio true or false based on the condition 



KNOWLEDGE_RECOMMENDED_EXCEL = OUTPUT_DIR/"knowledge_recommended_data.xlsx"       
FINAL_RECOMMENDED_EXCEL = OUTPUT_DIR/"final_recommendations.xlsx"
HYBRID_RECOMMENDED_EXCEL = OUTPUT_DIR/"hybrid_recommendations.xlsx"
 
TRAINED_MODEL = OUTPUT_DIR/"HAF-EPA.joblib"



