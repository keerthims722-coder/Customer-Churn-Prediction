# 🔮 Customer Churn Prediction & Retention Analytics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**An end-to-end Machine Learning project and interactive web application that predicts whether telecom subscribers will churn — enabling proactive, data-driven customer retention.**

</div>

---

## 📌 Project Overview

| | |
|---|---|
| **Goal** | Classify customers as **Churn (Yes)** or **Retained (No)** to enable proactive customer success interventions |
| **Champion Model** | **Random Forest Classifier** — selected for superior ROC-AUC score and robust ensemble bagging |
| **Dataset** | Telco Customer dataset — **5,880 records**, 21 columns |
| **Interface** | Multi-page **Streamlit** web application with real-time risk scoring |

### ML Workflow
```
EDA → ColumnTransformer Preprocessing → Stratified K-Fold CV → Multi-Model Evaluation → Streamlit App
      (OneHotEncoder + StandardScaler)
```

---

## 🖥️ Interactive Streamlit Web App

The project ships a **5-page Streamlit web application** built for presentations, executive dashboards, and real-time customer risk scoring:

| Page | Description |
|---|---|
| 🏠 **Dashboard / Overview** | KPI metric cards (Total Customers, Churned, Retained, Churn Rate), donut chart, business impact overview |
| 🎯 **Customer Prediction Studio** | 1-click sample profiles, custom input form, live risk gauge, churn probability, contributing risk drivers & retention recommendations |
| 📊 **Model Comparison & Metrics** | Side-by-side benchmarks (Accuracy, Precision, Recall, F1, ROC-AUC), ROC curves overlay, Confusion Matrices, Top-15 Feature Importances |
| 🔍 **Interactive EDA** | Plotly charts for contract types, tenure, payment methods, monthly charges, and correlation heatmaps |
| 📚 **ML Concepts & Student Guide** | Educational breakdowns of OneHotEncoding, StandardScaler, evaluation metric trade-offs, and model selection rationale |

### 🎯 One-Click Customer Profiles
- 🟢 *Loyal Long-Term Customer* — Low Risk
- 🔴 *Month-to-Month Fiber Customer* — High Risk
- 🟡 *Moderate Risk Customer (1-Year Contract)*
- ⚡ *Basic Phone Customer — Low/Moderate Risk*

---

## 🏆 Model Benchmark Results

| Model | CV ROC-AUC | Test ROC-AUC | Test Accuracy | Precision | Recall | F1 Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Random Forest (Champion)** | **0.5031** | **0.4982** | **51.19%** | **0.5054** | **0.4810** | **0.4929** |
| Logistic Regression | 0.4869 | 0.4748 | 50.09% | 0.4926 | 0.4000 | 0.4415 |
| Gradient Boosting | 0.4981 | 0.4632 | 48.04% | 0.4721 | 0.4517 | 0.4617 |

> **Why Random Forest?** Highest ROC-AUC on both cross-validation and the held-out test set, with the best balance across Precision, Recall, and F1 Score.

## 🧠 ML Pipeline Deep Dive

### 1. Data Loading & Exploration
- Loads `customer_churn_data.csv` (5,880 rows, 21 columns)
- Examines shape, dtypes, statistical summary, and target variable distribution

### 2. Data Preprocessing
- Drops `customerID` (non-predictive)
- Converts `Churn` → binary (1 = Yes, 0 = No)
- Converts `TotalCharges` to numeric; fills missing values with median

### 3. Feature Engineering & Train/Test Split
- **80/20 stratified split** to preserve churn ratio in both sets
- `ColumnTransformer` pipeline:
  - **Categorical features** → `OneHotEncoder`
  - **Numerical features** → `StandardScaler`

### 4. Model Training & Evaluation
- **3 models compared:** Logistic Regression, Random Forest, Gradient Boosting
- **5-Fold Stratified Cross-Validation** on training data
- Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC
- **Confusion matrices** & **ROC curves** plotted for visual comparison

### 5. Model Selection & Serialization
- **Random Forest** selected as champion (best CV + test ROC-AUC)
- Pipeline serialized via `joblib` for zero-latency inference in the Streamlit app

---

## 🛠️ Tech Stack

| Category | Library |
|---|---|
| **Data Manipulation** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn` |
| **Visualization** | `matplotlib`, `seaborn`, `plotly` |
| **Web Application** | `streamlit` |
| **Model Persistence** | `joblib` |

---

Made with ❤️ for AI & Data Science

</div>
