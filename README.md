# HAF-EPA: Hybrid AI Framework for Employee Project Allocation

HAF-EPA is a machine-learning-based employee recommendation framework for project allocation.
It trains a Random Forest model on employee-project feature vectors and then uses the trained model to recommend the most suitable employees for a new project description.

The project also includes a Flask web application that demonstrates how the trained model can be used in a practical scenario by uploading a project PDF and showing the Top-10 recommended employees.

---

## Main Purpose

The main purpose of this repository is to demonstrate the complete HAF-EPA workflow:

```text
Employee and project datasets
        ↓
Data preprocessing
        ↓
Employee-project pair generation
        ↓
Feature engineering
        ↓
Suitability labelling
        ↓
Train/test split
        ↓
Class balancing
        ↓
Random Forest model training
        ↓
Model evaluation
        ↓
Top-N recommendation generation
        ↓
Flask web application demonstration
```

The web application is not the main thesis contribution. It is a demonstration platform that shows how the trained HAF-EPA model can be applied to a real-world-style project description.

---

## Current Implementation Status

This repository currently implements:

- Employee-project pair generation
- Feature engineering with 19 engineered features
- Binary suitability labelling
- Train/test split
- Class balancing
- Random Forest model training
- Model evaluation
- Top-N recommendation export
- Employee reference export for the web application
- Flask-based PDF upload and recommendation demo
- Explainable recommendation display:
  - extracted project skills
  - matched employee skills
  - project match percentage
  - machine-learning suitability percentage
  - recommendation reason

Important note: the current runnable code uses a Random Forest model for the prediction component. Knowledge graph reasoning is discussed in the thesis/framework concept, but there is no separate runnable `knowledge_graph/` module in this repository.

---

## Project Structure

```text
HAF-EPA/
├── config.py
├── main.py
├── requirements.txt
├── README.md
│
├── datasets/
│   ├── training-dataset/
│   └── test-dataset/
│
├── data_loader/
│   └── load_datasets.py
│
├── process/
│   ├── normalize.py
│   ├── pair_creation.py
│   ├── feature_engineering.py
│   └── lebel_employee_project.py
│
├── pipeline/
│   ├── prepare_dataset.py
│   ├── split_data.py
│   ├── balance_data.py
│   ├── evaluate.py
│   ├── recommend.py
│   ├── export_results.py
│   └── export_top_employees.py
│
├── models/
│   ├── train_model.py
│   └── predict.py
│
├── helper/
│   ├── loader.py
│   └── model_required.py
│
├── webapp/
│   ├── app.py
│   ├── src/
│   │   ├── predictor.py
│   │   ├── pdf_parser.py
│   │   ├── project_parser.py
│   │   └── pdf_context_validation.py
│   ├── templates/
│   │   ├── index.html
│   │   └── graph.html
│   └── static/
│       ├── style.css
│       ├── graph.css
│       └── graph.js
│
├── webapp-test-pdf-file/
│   ├── StockPro.pdf
│   └── StockPro_Detailed_Project_Brief.pdf
│
├── project_explanation/
└── output/
```

---

## Dataset

The project uses a synthetic software-development employee-project dataset.

The dataset includes:

- employees
- projects
- tasks
- skills
- employee-skill mapping
- project-skill mapping
- employee availability
- employee feedback
- employee-project history
- relationship data
- skill similarity data

Dataset source:

```text
https://www.kaggle.com/datasets/firozfau/software-development-employee-project-dataset
```

The dataset is synthetic and is used for research and experimental purposes only. It does not contain real personal or organizational data.

---

## Machine Learning Model

The project uses:

```text
RandomForestClassifier
```

Model configuration:

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

The model is trained using 19 engineered features, including:

- matched_skill_count
- matched_required_skill_count
- matched_optional_skill_count
- employee_skill_count
- project_skill_count
- required_skill_count
- optional_skill_count
- skill_match_score
- employee_skill_coverage
- missing_required_skill_count
- has_any_skill_match
- strong_skill_match
- weighted_skill_match_score
- related_skill_match_score
- avg_experience_on_required_skills
- avg_past_performance_score
- availability_fit_score
- task_context_match_score
- soft_skill_compatibility_score

The target label is:

```text
1 = Suitable
0 = Not Suitable
```

---

## Installation

### 1. Clone or unzip the project

```bash
cd HAF-EPA
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Run the Full Training Pipeline

From the project root directory:

```bash
python main.py
```

This command performs the full model pipeline:

1. prepares the labelled dataset
2. splits train/test data
3. balances the training dataset
4. trains the Random Forest model
5. evaluates the model
6. generates test recommendations
7. exports CSV/Excel/report files
8. creates the employee reference file required by the web application

Generated files are saved in:

```text
output/
```

Important generated files:

```text
output/HAF-EPA.joblib
output/HAF-EPA_employee_reference.csv
output/HAF-EPA_model_evaluation.txt
output/HAF-EPA_feature_importance.csv
output/HAF-EPA_balanced_training_dataset.csv
output/HAF-EPA_test_predictions.csv
output/HAF-EPA_test_recommendations.xlsx
output/HAF-EPA_top_200_employees.xlsx
```

---

## Run the Web Application

Important: run `python main.py` first. The web app requires:

```text
output/HAF-EPA.joblib
output/HAF-EPA_employee_reference.csv
```

Then start the Flask application:

```bash
cd webapp
python app.py
```

Open the browser:

```text
http://127.0.0.1:5000
```

You can test the application using the sample PDFs in:

```text
webapp-test-pdf-file/
```

---

## Web Application Workflow

```text
Upload project PDF
        ↓
Extract project text
        ↓
Validate project PDF
        ↓
Extract required project skills
        ↓
Generate employee-project feature vectors
        ↓
Load trained HAF-EPA model
        ↓
Predict ML suitability score
        ↓
Calculate project-skill match percentage
        ↓
Rank employees
        ↓
Show Top-10 recommendations
```

The web application displays:

- uploaded file name
- extracted project skills
- Top-10 employees
- project match percentage
- ML suitability percentage
- matched project skills
- recommendation reason
- bar chart visualization

---

## Explainability Features

The web application improves recommendation transparency by showing:

- extracted skills from the uploaded PDF
- employee skills
- matched project skills
- number of matched skills
- project match percentage
- ML suitability percentage
- reason why the employee was recommended

The current chart view intentionally uses only a bar chart because employee recommendation scores are independent ranking values. Line charts and pie charts are not suitable for this type of comparison.

---

## Difference Between Project Match and ML Suitability

The web app separates two different values:

### Project Match %

This is calculated from extracted PDF skills.

Example:

```text
Project skills: Data Analysis, SQL, Testing
Employee matched: 3 of 3 skills
Project Match = 100%
```

### ML Suitability %

This is predicted by the trained Random Forest model.

Example:

```text
Feature vector
        ↓
HAF-EPA.joblib
        ↓
predict_proba()
        ↓
ML Suitability Score
```

The ML score is not the same as the direct skill match score. It represents the model's predicted suitability probability based on the full employee-project feature vector.

---

## Example Web App Output

For `StockPro.pdf`, the extracted project skills may be:

```text
Data Analysis
SQL
Testing
```

Example recommendation table:

| Rank | Employee | Project Match | ML Suitability | Matched Skills |
|---:|---|---:|---:|---|
| 1 | Christopher Bass | 100.00% | 10.00% | Data Analysis, SQL, Testing |
| 2 | James Collins | 100.00% | 9.00% | Data Analysis, SQL, Testing |
| 3 | Donna Jordan | 100.00% | 8.67% | Data Analysis, SQL, Testing |

When multiple employees have the same project-skill match, the ML suitability score is used to rank them.

---

## Troubleshooting

### Model file not found

If you see:

```text
Model file not found: output/HAF-EPA.joblib
```

Run:

```bash
python main.py
```

### Employee reference file not found

If you see:

```text
Employee file not found: output/HAF-EPA_employee_reference.csv
```

Run:

```bash
python main.py
```

### Uploaded PDF is not recognized

The web app expects a project description PDF containing project-related sections such as:

- project overview
- project description
- required skills
- technical requirements
- objectives
- expected outcome

Use the sample PDFs in:

```text
webapp-test-pdf-file/
```

---

## Known Limitations

- The dataset is synthetic.
- The current web app extracts explicit skills from the project PDF.
- Hidden skill inference is not implemented.
- Transfer learning is not implemented.
- The current runnable model is Random Forest based.
- The trained model is most reliable when the new employee/project data are similar to the training dataset.
- If a company has a very different workforce or project domain, retraining with company-specific data is recommended.

---

## Recommended Future Improvements

- Add SHAP or LIME for feature-level explainability.
- Improve PDF requirement extraction using NLP or transformer-based models.
- Add hidden skill inference from project descriptions.
- Add transfer learning or fine-tuning for cross-company model portability.
- Add model monitoring and periodic retraining.
- Add authentication and database persistence for production deployment.


---

## Academic Summary

HAF-EPA is a research and thesis-oriented framework for employee-project allocation.
The core contribution is the model creation, training, evaluation, and recommendation workflow.
The Flask web application demonstrates how the trained model can be used in a practical setting.


## 👨‍💻 Author

**MD Firozur Rahman**
ID: 22975954


---

## 👨‍🎓 Academic Information

<table>
<tr>
<td align="center">

**Student**  
Md Firozur Rahman  
MSc in Data Science  
FAU  

</td>

<td align="center">

**Supervisor**  
Robert Bauer  
Managing Director  
TW-Legal Tech  

</td>

<td align="center">

**Professor**  
Prof. Frauke Liers  
Head of the Data Science Department  
FAU  

</td>
</tr>
</table>