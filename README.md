# HAF-EPA: Hybrid AI Framework for Employee Project Allocation

---

## 📖 Overview

HAF-EPA (Hybrid AI Framework for Employee Project Allocation) is an intelligent system designed to automatically recommend the most suitable employees for a given project.

The framework integrates:

*-**Machine Learning (Random Forest)**
*-**Knowledge Graph Reasoning**
*-**Rule-Based Filtering**

to improve accuracy, efficiency, and fairness in employee allocation.

---

## Objectives

* Automate employee-to-project assignment
* Reduce manual bias and inefficiency
* Improve decision accuracy
* Recommend **Top-K best employees** for each project
* Support real-world decision-making systems

---

## 🏗️ System Architecture

```text
Raw Data
→ Data Preprocessing & Normalization
→ Employee-Project Pair Creation
→ Feature Engineering
→ Label Generation
→ Train/Test Split (80/20)
→ Machine Learning Model (Random Forest)
→ Model Evaluation
→ External Testing
→ Prediction Pipeline
→ Knowledge Graph Recommendation
→ Final Recommendation (ML)
→ Hybrid Recommendation (ML + KG)
```

---

## 📂 Project Structure

```text
HAF-EPA/
├── data_loader/
├── process/
├── models/
├── knowledge_graph/
├── output/
├── config.py
├── main.py
└── README.md
```

---
---
## 📊 Dataset

This project uses a **synthetic dataset** designed to simulate real-world employee–project allocation scenarios.

The dataset includes:

* Employees
* Projects
* Tasks
* Skills
* Employee-Skill Mapping
* Project-Skill Mapping

### 🔗 Data Source

The dataset is publicly available on Kaggle:

👉 https://www.kaggle.com/datasets/firozfau/software-development-employee-project-dataset

### ⚠️ Note

* The dataset is **synthetically generated** for research and experimental purposes.
* It does not contain real personal or organizational data.
* It is designed to reflect realistic software development project environments.


---
## ⚙️ Workflow

### 1️⃣ Data Loading

* Employees
* Projects
* Tasks
* Skills

---

### 2️⃣ Data Preprocessing

* Cleaning
* Validation
* Normalization
* Skill mapping

---

### 3️⃣ Employee-Project Pair Creation

Each row represents:

```
(Employee + Project)
```

---

### 4️⃣ Feature Engineering

* matched skill count
* skill match score
* experience score
* availability score
* skill coverage
* primary skill match

---

### 5️⃣ Label Generation

* `1` → Suitable
* `0` → Not Suitable

---

## 🤖 Machine Learning Model

### Model Used

* Random Forest Classifier

### Why Random Forest?

* Works well with tabular data
* Handles non-linear relationships
* Reduces overfitting
* No heavy preprocessing needed

---

## 📊 Training & Testing

### 🔹 Train-Test Split Implementation

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

* 80% → Training
* 20% → Internal Testing (Unseen Data)

---

### 🔹 Training Phase

* Model learns from full dataset
* Saved as:

```
HAF-EPA.joblib
```

---

### 🔹 Internal Testing (Held-out 20%)

* Uses unseen test data
* Metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

## 🌍 External Testing (NEW)

The system also supports **external unseen dataset evaluation**:

* Uses `datasets/test-dataset/`
* Completely independent from training data
* Simulates real-world scenario

👉 If labels exist:

* metrics are calculated

👉 If labels do not exist:

* only prediction is performed

---

## 🔮 Prediction Pipeline

* Load trained model
* Generate features
* Predict suitability score

```
predicted_score (0 → 1)
```

---

## Knowledge Graph Recommendation

* Uses full dataset
* Builds relationships:

  * Employee ↔ Skills ↔ Projects

---

## 🏆 Recommendation System

### ✅ ML Recommendation

* Top-K based on predicted score

### 🔗 Hybrid Recommendation

* ML + Knowledge Graph combined

---

## ▶️ How to Run

### Train Model

```python
RUN_GENERATE_TRAIN_MODEL = True
```

---

### Evaluate Model (20%)

```python
RUN_EVALUATE_MODEL = True
```

---

### External Testing

```python
RUN_TEST_MODEL = True
```

---

### Recommendation

```python
RUN_KG_RECOMMEND = True
RUN_FINAL_RECOMMENDATION = True
RUN_HYBRID_RECOMMENDATION = True
```

---

## ⚠️ Important Design Principles

### ✔ Separation of Pipeline

| Stage         | Data         |
| ------------- | ------------ |
| Training      | 80%          |
| Internal Test | 20%          |
| External Test | New dataset  |
| Prediction    | Model output |

---

### ✔ Modular Execution

Each stage can be turned ON/OFF using flags:

* Avoid retraining
* Fast demo for viva
* Efficient pipeline

---

### ✔ Model Validation

Before inference:

* Check if model exists
* Check if test data exists

---

## 📌 Example Output

| Employee | Score |
| -------- | ----- |
| Emp1     | 0.94  |
| Emp2     | 0.91  |
| Emp3     | 0.88  |

---

## 🧪 Key Contributions

* Hybrid AI (ML + KG)
* External testing support
* Modular pipeline
* Real-world simulation
* Explainable system


---

## Final Note

This project demonstrates a **production-ready hybrid AI system** combining:

* Machine Learning
* Knowledge Graph
* Real-world testing
 
---
---

### 👥 Real-World Employee Allocation

* Uses the full employee dataset
* Automatically matches employees with project requirements
* Generates **Top best-fit employees** based on:

  * Skill matching
  * Experience
  * Availability
  * ML prediction score

---
---
### WebApp Project Structure (HAF-EPA)
---
HAF-EPA/
│
├── webapp/
│   ├── src/
│   │   ├── *.py
│   │
│   ├── templates
│   │   ├── index.html
│   │   └── graph.html
│   │
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │
│   ├── uploads/
│   │
│   ├── app.py
│   │
│   └── requirements.txt
│
├── output/
    ├── HAF-EPA.joblib
    └── employee_reference.csv


---

### 🌐 Web Application Integration

* A user-friendly web interface will be developed
* Features include:

  * Project upload (PDF / form input)
  * Real-time prediction results
  * Interactive dashboards

---

### 📊 Visualization & User Interaction

* Visual graphs and analytics for:

  * Employee ranking
  * Skill matching score
  * Recommendation explanation
* Enhances interpretability and decision-making

---
## Current Web Application Workflow

Upload Project PDF
→ Extract PDF Text
→ Validate Project PDF Structure
→ Extract Required Skills
→ Load Employee Dataset
→ Build Employee Matching Features
→ Load Trained HAF-EPA Model
→ Predict Employee Suitability Score
→ Filter Invalid Matches
→ Rank Top Employees
→ Show Results in Web UI
→ Visualize Results with Charts

---
## 📊 Available Graph Visualizations

The new graph page supports multiple chart types for better visualization of recommendation results:

* Bar Chart
* Horizontal Bar Chart
* Line Chart
* Pie Chart
* Doughnut Chart
* Radar Chart
* Polar Area Chart

The graph page also allows users to select different metrics, such as:

* Match Percentage
* Matched Skill Count
* Skill Match Score

---
---
# Run code 
## For Traing and Test Model 
  python3 main.py

## For Webapp
  python3 webapp/app.py

---
---
 ## Future Work

* Deep learning integration 
* Real-time prediction
* Explainable dashboard

---
## 📚 Academic Summary

> The HAF-EPA framework integrates machine learning and knowledge graph reasoning for employee-project allocation. The system uses a supervised learning approach with a Random Forest classifier trained on 80% of the data and evaluated on 20% unseen data. Additionally, an external dataset is used to simulate real-world prediction scenarios. The final output combines ML predictions and knowledge graph reasoning.

---

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
