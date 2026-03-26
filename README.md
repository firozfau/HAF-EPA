# 🚀 HAF-EPA

**Hybrid AI Framework for Employee Project Allocation**

---

## 📌 Overview

HAF-EPA is a hybrid AI-based system designed to intelligently recommend the most suitable employees for a given project.

This system combines:

* 🤖 Machine Learning (Random Forest)
* 🧠 Knowledge Graph
* 🔗 Hybrid Recommendation Logic

It enables:

* Data-driven employee selection
* Smart project allocation
* Automated recommendation from PDF input
* End-to-end prediction via web interface

---

## 🎯 Key Features

* 📊 Employee–Project suitability prediction
* 🤖 ML-based ranking (Random Forest)
* 🧠 Knowledge Graph integration
* 🔗 Hybrid scoring (ML + Graph)
* 📄 PDF-based project input
* 🌐 Web application interface
* 📈 Visualization (Pie Chart + Bar Chart)

---

## 🏗️ Project Structure

```
HAF-EPA/
│
├── data_loader/
├── models/
├── inference/
├── testing_dataset/
├── webapp/
│   ├── templates/
│   ├── services/
│   └── app.py
│
├── saved_outputs/
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python 3
* Pandas, NumPy
* Scikit-learn
* Flask
* Matplotlib
* Joblib

---

## 🧠 Machine Learning Pipeline

1. Dataset Loading
2. Skill Mapping & Enrichment
3. Employee–Project Pair Generation
4. Feature Engineering
5. Label Generation
6. Model Training (Random Forest)
7. Evaluation (Accuracy, Precision, Recall, F1-score)

---

## 🔗 Knowledge Graph

The system builds a knowledge graph with:

**Nodes:**

* Employees
* Projects
* Skills

**Edges:**

* HAS_SKILL
* WORKED_ON

Used for:

* Graph-based recommendations
* Hybrid scoring enhancement

---

## 📄 PDF-Based Inference

The system supports real-world usage via PDF input.

### Workflow:

1. Upload project PDF
2. Extract:

   * Project title
   * Type
   * Required skills
   * Complexity
3. Generate employee candidates
4. Predict suitability using trained model
5. Rank Top 10 employees

---

## 🏆 Output Example

* Top 10 recommended employees
* Predicted score
* Suggested role (Frontend / Backend / etc.)

---

## ▶️ How to Run

### 1️⃣ Train the Model

```bash
python3 main.py
```

### 2️⃣ Run PDF Recommendation

```bash
python3 inference/recommend_from_pdf.py
```

---

## 🌐 Run Web Application

```bash
cd webapp
python3 app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

👉 Upload a PDF → Get ranked employee recommendations instantly.

---

## 📂 Saved Outputs

* trained_model.joblib
* kg_nodes.csv
* kg_edges.csv
* top5_*.csv
* hybrid_recommendations_*.csv

---

## 📅 Project Timeline

**Duration:** April 2026 – June 2026 (3 Months)

* Data Layer: Weeks 2–4
* ML Layer: Weeks 5–8
* Knowledge Graph: Weeks 9–10
* Inference: Week 11
* Web App: Week 12

---

## 🎓 Academic Context

This project is developed as part of a Master's Thesis at
**Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)**.

**Thesis Title:**
*A Knowledge-Driven Framework for Intelligent Employee Project Allocation*

---

## 👤 Author

**MD FIROZUR RAHMAN**
MSc in Data Science
Friedrich-Alexander-Universität Erlangen-Nürnberg

---

## 🤝 Acknowledgment

* Prof. Frauke Liers 
Head of the Data Science Department
Friedrich-Alexander-Universität Erlangen-Nürnberg

* Robert Bauer
Managing Director
TW Legal Tech Rechtsanwaltsgesellschaft mbH

---

## 📜 License

This project is intended for academic and research purposes.

---

## ⭐ Future Improvements

* Deep Learning model integration
* Real-time employee availability tracking
* REST API deployment
* Cloud-based system (AWS/GCP)

---
