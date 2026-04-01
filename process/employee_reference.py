import ast
import pandas as pd
from config import EMPLOYEE_REFERENCE_INFORMATION


# create_employee_reference is used to create a clean employee reference dataset.

# 1. It selects important employee details like id, name, skills, experience, and availability.

# 2. It cleans the employee_skills column and converts it into a readable format
#    like "Python, SQL" instead of list or raw text.

# 3. It removes duplicate employees.

# 4. It normalizes availability into a score (0 to 1).

# 5. It saves the final cleaned employee reference data into a CSV file.

# Finally, it returns the cleaned employee dataframe.

def create_employee_reference(final_rec: pd.DataFrame) -> pd.DataFrame:
    
    employee_df = final_rec[
        ["employee_id", "full_name", "employee_skills", "experience", "availability"]
    ].copy()

    # Clean skills column
    def clean_skills(value):
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, list):
                return ", ".join(str(s).strip() for s in parsed)
        except:
            pass

        text = str(value).replace("[", "").replace("]", "").replace("'", "")
        return ", ".join([x.strip() for x in text.split(",") if x.strip()])

    employee_df["employee_skills"] = employee_df["employee_skills"].apply(clean_skills)

    # Remove duplicates
    employee_df = employee_df.drop_duplicates(subset=["employee_id"]).reset_index(drop=True)

    # Normalize availability
    employee_df["availability_score"] = employee_df["availability"] / 100.0

    # Ensure folder exists
    EMPLOYEE_REFERENCE_INFORMATION.parent.mkdir(parents=True, exist_ok=True)

    # Save file
    employee_df.to_csv(EMPLOYEE_REFERENCE_INFORMATION, index=False)

    print(f"Saved to: {EMPLOYEE_REFERENCE_INFORMATION}")

    return employee_df