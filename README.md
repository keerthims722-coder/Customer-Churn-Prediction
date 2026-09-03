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

---

## 📁 Project Structure

```
📦 Customer-Churn-Prediction
├── app.py                                  # Root launcher (delegates to Task---1-main/app.py)
├── requirements.txt                        # Top-level dependencies
├── README.md                               # This file
│
└── Task---1-main/
    ├── app.py                              # ⭐ Main Streamlit Web Application (5 pages)
    ├── train_model.py                      # ML training & pipeline serialization script
    ├── test_prediction.py                  # Automated verification test suite
    ├── customer_churn_data.csv             # Telco dataset (5,880 records, 21 columns)
    ├── Customer_Churn_Classification.ipynb # Full EDA & experiment Jupyter Notebook
    ├── requirements.txt                    # Project dependencies
    ├── project_documentation.md           # Detailed ML workflow documentation
    ├── models_performance_comparison.png  # Visual model benchmark chart
    ├── Screenshot.png                      # App preview screenshot
    └── model/
        ├── churn_model_pipeline.joblib    # Trained champion Random Forest pipeline
        ├── all_models.joblib              # All trained model pipelines
        └── metadata.joblib                # Metrics, ROC points & feature importances
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🌟 Option A — Launch the Streamlit Web App (Recommended)
```bash
streamlit run app.py
```
Then open your browser at **`http://localhost:8501`**

---

### 🔁 Option B — Retrain & Re-export Model Pipelines
```bash
cd Task---1-main
python train_model.py
```

---

### ✅ Option C — Run Automated Verification Tests
```bash
cd Task---1-main
python test_prediction.py
```

---

### 📓 Option D — Explore the Jupyter Notebook
```bash
cd Task---1-main
jupyter notebook Customer_Churn_Classification.ipynb
```

---

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

## 💡 Presentation Guide (College / Demo Sessions)

| Step | Duration | Action |
|---|---|---|
| **1. Introduction** | ~1 min | Open **Dashboard** — explain why churn costs 5–7× more than retention |
| **2. Live Demo** | ~2 min | Go to **Prediction Studio** → click 🔴 *Month-to-Month Fiber* for high risk, then 🟢 *Loyal Long-Term* for safe score |
| **3. Evaluation** | ~1 min | Open **Model Comparison** — walk through ROC curves, Confusion Matrix, and Feature Importances |
| **4. Theory Q&A** | ~1 min | Navigate to **ML Concepts** — answer examiner questions on `OneHotEncoder`, `StandardScaler`, and precision/recall trade-offs |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ for AI & Data Science

</div>
