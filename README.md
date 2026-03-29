# 📌 HAF-EPA: Hybrid AI Framework for Employee Project Allocation

---

## 📖 Overview

HAF-EPA (Hybrid AI Framework for Employee Project Allocation) is an intelligent system designed to automatically recommend the most suitable employees for a given project.

The framework integrates:

* 🤖 **Machine Learning (Random Forest)**
* 🧠 **Knowledge Graph Reasoning**
* ⚙️ **Rule-Based Filtering**

to improve accuracy, efficiency, and fairness in employee allocation.

---

## 🎯 Objectives

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
→ Prediction Pipeline
→ Knowledge Graph Recommendation
→ Final Recommendation (ML)
→ Hybrid Recommendation (ML + KG)
```

---

## 📂 Project Structure

```text
HAF-EPA/
│
├── data_loader/
│   └── load_datasets.py
│
├── process/
│   ├── normalize.py
│   ├── employee_skill_mapping.py
│   ├── project_skill_mapping.py
│   ├── mapping.py
│   ├── pair_creation.py
│   ├── feature_engineering.py
│   └── lebel_employee_project.py
│
├── models/
│   ├── train_model.py
│   ├── generate_train_model.py
│   ├── evaluate_model.py
│   ├── test_model.py
│   ├── predict.py
│   ├── final_recommendation.py
│   └── hybrid_recommendation.py
│
├── knowledge_graph/
│   └── kg_recommend.py
│
├── output/
│   ├── HAF-EPA.joblib
│   ├── held_out_test_data.joblib
│   ├── knowledge_recommended_data.xlsx
│   ├── final_recommendations.xlsx
│   └── hybrid_recommendations.xlsx
│
├── config.py
├── main.py
└── README.md
```

---

## ⚙️ Workflow

### 1️⃣ Data Loading

* Load datasets:

  * Employees
  * Projects
  * Tasks
  * Skills

---

### 2️⃣ Data Preprocessing

* Data cleaning
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

Key features:

* matched skill count
* skill match score
* experience score
* availability score
* skill coverage
* primary skill match

---

### 5️⃣ Label Generation

* `1` → Suitable employee
* `0` → Not suitable

---

## 🤖 Machine Learning Model

### Model Used

* **Random Forest Classifier**

### Why Random Forest?

* Works well with structured/tabular data
* Handles **non-linear relationships**
* Reduces overfitting (ensemble learning)
* Requires minimal preprocessing

---

## 📊 Training & Testing

### Train-Test Split

* **80% Training Data**
* **20% Unseen Testing Data**

### Training Phase

* Model learns from employee-project features
* Model saved as:

```
HAF-EPA.joblib
```

---

### Testing Phase (Unseen Data)

* Uses **20% unseen data**
* Ensures fair evaluation
* Prevents data leakage

---

## 📈 Model Evaluation Metrics

| Metric           | Description                          |
| ---------------- | ------------------------------------ |
| Accuracy         | Overall correctness                  |
| Precision        | Correct positive predictions         |
| Recall           | Ability to detect suitable employees |
| F1-score         | Balance between precision & recall   |
| Confusion Matrix | TP, TN, FP, FN analysis              |

---

## 🔮 Prediction Pipeline

After training and evaluation:

* Load trained model
* Generate features for employee-project pairs
* Predict suitability score

Output:

```
predicted_score (0 → 1)
```

---

## 🧠 Knowledge Graph Recommendation

* Captures relationships between:

  * employees
  * skills
  * projects

* Helps identify:

  * domain experts
  * related experience
  * skill connections

---

## 🏆 Recommendation System

### ✅ Final Recommendation (ML)

* Ranked employees based on predicted score
* Top-K selection

---

### 🔗 Hybrid Recommendation (ML + KG)

Combines:

* Machine Learning prediction
* Knowledge Graph reasoning

Final Output:

```
Best possible employee recommendations
```

---

## ▶️ How to Run

### Step 1: Train Model

```bash
python main.py
```

Enable:

```python
RUN_GENERATE_TRAIN_MODEL = True
```

---

### Step 2: Evaluate Model

```bash
python main.py
```

Enable:

```python
RUN_EVALUATE_MODEL = True
```

---

### Step 3: Run Prediction

```bash
python main.py
```

Enable:

```python
RUN_TEST_MODEL = True
```

---

### Step 4: Generate Recommendations

Enable:

```python
RUN_KG_RECOMMEND = True
RUN_FINAL_RECOMMENDATION = True
RUN_HYBRID_RECOMMENDATION = True
```

---

## 📌 Example Output

| Employee | Score |
| -------- | ----- |
| Emp1     | 0.94  |
| Emp2     | 0.91  |
| Emp3     | 0.88  |

---

## ⚠️ Important Design Principle

The system strictly separates:

### ✔ Training

* Uses 80% data

### ✔ Testing

* Uses 20% unseen data

### ✔ Recommendation

* Uses trained model after testing

👉 This ensures:

* No data leakage
* Fair evaluation
* Real-world applicability

---

## 🧪 Key Contributions

* Hybrid AI system (ML + KG)
* Feature-based employee matching
* Scalable recommendation system
* Explainable AI pipeline

---

## 📚 Academic Summary

> The HAF-EPA framework integrates machine learning and knowledge graph reasoning for employee-project allocation. The system uses a supervised learning approach with a Random Forest classifier trained on 80% of the data and evaluated on 20% unseen data. After evaluation, the model is used for project-specific prediction and combined with knowledge graph outputs to generate hybrid recommendations.

---

## 👨‍💻 Author

**MD Firozur Rahman**
ID: 22975954

---

## 🚀 Future Work

* Deep learning integration
* Real-time API deployment
* Explainable AI dashboard
* Dynamic skill updates

---

## ⭐ Final Note

This project demonstrates a **real-world hybrid AI solution** for intelligent employee allocation combining:

* Machine Learning
* Knowledge Graph
* Rule-based reasoning

---
 
