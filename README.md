# 🚀 Employee Attrition Predictor

A machine learning pipeline and interactive dashboard for predicting employee attrition risk, built with Python, scikit-learn, and Streamlit.

---

## 📸 Dashboard Preview

> Run the app locally to see the full interactive dashboard (see setup below).

**Tab 1 — Single Employee Analysis**
Enter an employee's details and get an instant attrition risk score with the top 5 risk drivers visualised using approximate SHAP values.

**Tab 2 — Bulk Upload**
Upload a CSV of employees and download a scored file flagging who is "At Risk" vs "Stay".

**Tab 3 — Cohort Analytics**
Department-level risk breakdown, risk score distribution, precision/recall curve, and a calibration curve to validate model reliability.

---

## 🧠 Model Architecture

The pipeline uses a **stacking ensemble** with four base learners and a Logistic Regression meta-learner:

| Layer | Model |
|---|---|
| Base learner 1 | XGBoost |
| Base learner 2 | LightGBM |
| Base learner 3 | Random Forest |
| Base learner 4 | Logistic Regression |
| Meta-learner | Logistic Regression |
| Calibration | Isotonic (CalibratedClassifierCV) |
| Class imbalance | SMOTE + TomekLinks (85:15 ratio) |

---

## ⚙️ Feature Engineering

Five custom features built on top of the raw IBM HR dataset:

| Feature | Description |
|---|---|
| `TenureRatio` | Years since last promotion / (years at company + 1) |
| `OT_x_JobSat` | Overtime (binary) × Job Satisfaction — captures burnout signal |
| `AvgSatisfaction` | Mean of Environment, Job, and Relationship Satisfaction |
| `RoleAttritionRate` | Historical attrition rate for each job role |
| `YearsPerCompany` | Years at company / (number of companies worked + 1) |

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | 86.7% |
| ROC AUC | 0.807 |
| PR AUC | 0.520 |
| Optimal F1 Threshold | 0.324 |
| Cost-optimised Threshold | 0.15 |

> Thresholds are tuned separately for F1, minimum cost, recall ≥51%, and maximum accuracy. The dashboard lets you adjust the threshold interactively.

---

## 🗂️ Project Structure

```
├── employee_attrition_project.py   # Training pipeline
├── app-2.py                        # Streamlit dashboard
├── artifacts/
│   ├── attrition_model.pkl         # Trained + calibrated model
│   └── thresholds.json             # Optimal thresholds
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn xgboost lightgbm imbalanced-learn joblib altair matplotlib
```

### 2. Train the model

```bash
python employee_attrition_project.py
```

This downloads the IBM HR dataset, trains the stacking ensemble, calibrates it, and saves the model and thresholds to `artifacts/`.

> ⚠️ Training takes a few minutes due to GridSearchCV with 5-fold cross-validation.

### 3. Run the dashboard

```bash
streamlit run app-2.py
```

Opens at `http://localhost:8501`

---

## 📦 Dataset

[IBM HR Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — publicly available, 1,470 employee records, 35 features, 16% attrition rate.

---

## 🛠️ Tech Stack

`Python` `scikit-learn` `XGBoost` `LightGBM` `imbalanced-learn` `Streamlit` `Altair` `Matplotlib` `pandas` `NumPy` `joblib`

---

## 👩‍💻 Author

**Aura Sintha** — MIT Student, Whitireia | WelTec, Wellington NZ

[LinkedIn](https://www.linkedin.com/in/aura-pradnya-0b24901bb) · [GitHub](https://github.com/aura080203)

---

## ⚠️ Notes

- This project uses a publicly available dataset for learning purposes
- The model is intended as a demonstration of ML pipeline design, not for production HR decisions
- Bulk upload (Tab 2) expects a CSV with the same schema as the IBM HR dataset
