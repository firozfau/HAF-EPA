# HAF-EPA: Hybrid AI Framework for Employee Project Allocation

---

## 📖 Overview

HAF-EPA (Hybrid AI Framework for Employee Project Allocation) is an intelligent system designed to automatically recommend the most suitable employees for a given project.

The framework integrates:

* 🤖 **Machine Learning (Random Forest)**
* 🧠 **Knowledge Graph Reasoning**
* ⚙️ **Rule-Based Filtering**

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

## 🧠 Knowledge Graph Recommendation

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

## 📚 Academic Summary

> The HAF-EPA framework integrates machine learning and knowledge graph reasoning for employee-project allocation. The system uses a supervised learning approach with a Random Forest classifier trained on 80% of the data and evaluated on 20% unseen data. Additionally, an external dataset is used to simulate real-world prediction scenarios. The final output combines ML predictions and knowledge graph reasoning.

---

## 👨‍💻 Author

**MD Firozur Rahman**
ID: 22975954

---

## ⭐ Final Note

This project demonstrates a **production-ready hybrid AI system** combining:

* Machine Learning
* Knowledge Graph
* Real-world testing
 
---

 ## 🚀 Future Work

* Deep learning integration 
* Real-time prediction
* Explainable dashboard

---
---
## 🚀 Ongoing Work / Future Extension

The HAF-EPA system is being extended into a real-world intelligent web application with the following capabilities:

### 📌 Project-Based Prediction

* Users can upload or define a new project (e.g., PDF, form input).
* The system extracts project requirements such as:

  * Required skills
  * Project domain
  * Complexity level
* The trained model predicts and ranks the most suitable employees.

---

### 👥 Real-World Employee Allocation

* Uses the full employee dataset
* Automatically matches employees with project requirements
* Generates **Top-5 best-fit employees** based on:

  * Skill matching
  * Experience
  * Availability
  * ML prediction score

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

### 🎯 Goal

To transform the system into a **fully automated intelligent decision-support platform** that can be used in real-world HR and project management systems.

---

