#!/usr/bin/env python3
"""
app.py — Employee Attrition Predictor Dashboard
─────────────────────────────────────────────────
Fixes vs original (app-2.py):
  1. Uses role_rates from artifacts (no leakage at inference)
  2. Bulk upload no longer crashes on missing EmployeeID column
  3. Approximate SHAP renamed to Feature Impact Analysis (accurate)
  4. Error handling if artifacts don't exist yet
  5. Calibration chart uses test-set data, not training data
  6. Form inputs grouped into expanders for cleaner UX
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import altair as alt
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from utils import engineer_features_inference, load_data

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# ── Paths ───────────────────────────────────────────────────────────────────
MODEL_PATH      = Path('artifacts/attrition_model.pkl')
THRESH_PATH     = Path('artifacts/thresholds.json')
ROLE_RATES_PATH = Path('artifacts/role_rates.json')
DATA_URL        = (
    'https://raw.githubusercontent.com/'
    'mragpavank/ibm-hr-analytics-attrition-dataset/'
    'master/WA_Fn-UseC_-HR-Employee-Attrition.csv'
)


# ── Load resources with error handling ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    # Check artifacts exist before loading
    missing = [p for p in [MODEL_PATH, THRESH_PATH, ROLE_RATES_PATH]
               if not p.exists()]
    if missing:
        return None, None, None, None, None

    model      = joblib.load(MODEL_PATH)
    thresholds = pd.read_json(THRESH_PATH, typ='series')

    with open(ROLE_RATES_PATH) as f:
        rate_data = json.load(f)
    role_rates  = rate_data['role_rates']
    global_mean = rate_data['global_mean']

    # Load dataset and apply leak-free feature engineering
    df_raw = load_data(DATA_URL)
    df_raw['Attrition'] = (df_raw['Attrition'] == 'Yes').astype(int)
    df_all = engineer_features_inference(df_raw, role_rates, global_mean)

    return model, thresholds, df_all, role_rates, global_mean


model, thresholds, df_all, role_rates, global_mean = load_resources()

# ── Artifact guard ──────────────────────────────────────────────────────────
if model is None:
    st.error("⚠️ Model artifacts not found.")
    st.info("Run the training script first:\n```\npython employee_attrition_project.py\n```")
    st.stop()

th_cost = thresholds['threshold_cost']

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
threshold   = st.sidebar.slider("Risk Threshold", 0.0, 1.0,
                                 float(th_cost), 0.01,
                                 help="Employees above this score are flagged At Risk")
dept_filter = st.sidebar.multiselect("Filter by Department",
                                      sorted(df_all['Department'].unique()))

# Show stored metrics in sidebar
with st.sidebar.expander("📊 Model Metrics"):
    try:
        with open(THRESH_PATH) as f:
            thresh_data = json.load(f)
        if 'metrics' in thresh_data:
            m = thresh_data['metrics']
            st.metric("Accuracy",  f"{m['accuracy']:.1%}")
            st.metric("ROC AUC",   f"{m['roc_auc']:.3f}")
            st.metric("PR AUC",    f"{m['pr_auc']:.3f}")
    except Exception:
        st.write("Train model to see metrics")

# ── Main ─────────────────────────────────────────────────────────────────────
st.title("🚀 Employee Attrition Predictor")
tab1, tab2, tab3 = st.tabs(["Individual", "Bulk Upload", "Analytics"])


# ── Tab 1: Single Employee ───────────────────────────────────────────────────
with tab1:
    st.header("🔍 Single Employee Analysis")

    with st.form("attrition_form"):
        # Group inputs into expanders for cleaner UX
        with st.expander("👤 Demographics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                age     = st.slider("Age", 18, 60, 30)
                gender  = st.selectbox("Gender", ['Male', 'Female'])
                marital = st.selectbox("Marital Status",
                                        ['Single', 'Married', 'Divorced'])
            with col2:
                dist          = st.slider("Distance From Home (km)", 1, 50, 5)
                education_lvl = st.slider("Education Level (1-5)", 1, 5, 3)
                edu_field     = st.selectbox("Education Field",
                                              ['Life Sciences', 'Medical',
                                               'Marketing', 'Technical Degree',
                                               'Human Resources', 'Other'])

        with st.expander("💼 Job Details", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                business  = st.selectbox("Business Travel",
                                          ['Non-Travel', 'Travel_Rarely',
                                           'Travel_Frequently'])
                dept      = st.selectbox("Department",
                                          ['Sales', 'Research & Development',
                                           'Human Resources'])
                job_role  = st.selectbox("Job Role",
                                          sorted(role_rates.keys()))
                overtime  = st.selectbox("OverTime", ['Yes', 'No'])
            with col2:
                yrs_company   = st.slider("Years at Company", 0, 40, 3)
                yrs_promo     = st.slider("Years Since Last Promotion", 0, 15, 1)
                num_companies = st.slider("Num. Companies Worked", 0, 10, 1)
                total_years   = st.number_input("Total Working Years", 0, 40, 8)
                training      = st.slider("Training Times Last Year", 0, 10, 2)

        with st.expander("💰 Compensation", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                monthly_income = st.number_input("Monthly Income", 1000, 20000, 7000)
                percent_hike   = st.slider("Percent Salary Hike", 0, 25, 18)
                stock_option   = st.slider("Stock Option Level", 0, 3, 1)
            with col2:
                daily_rate  = st.number_input("Daily Rate", 100, 1500, 800)
                hourly_rate = st.number_input("Hourly Rate", 20, 100, 50)

        with st.expander("😊 Satisfaction", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                job_sat      = st.slider("Job Satisfaction (1-4)", 1, 4, 4)
                env_sat      = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
            with col2:
                rel_sat      = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
                work_balance = st.slider("Work Life Balance (1-4)", 1, 4, 3)

        submit = st.form_submit_button("🔮 Predict Risk", type="primary")

    if submit:
        ot_num = 1 if overtime == 'Yes' else 0
        record = {
            'Age':                      age,
            'BusinessTravel':           business,
            'DailyRate':                daily_rate,
            'Department':               dept,
            'DistanceFromHome':         dist,
            'Education':                education_lvl,
            'EducationField':           edu_field,
            'EnvironmentSatisfaction':  env_sat,
            'Gender':                   gender,
            'HourlyRate':               hourly_rate,
            'JobInvolvement':           job_sat,
            'JobLevel':                 1,
            'JobRole':                  job_role,
            'JobSatisfaction':          job_sat,
            'MaritalStatus':            marital,
            'MonthlyIncome':            monthly_income,
            'MonthlyRate':              daily_rate * 24,
            'NumCompaniesWorked':       num_companies,
            'OverTime':                 overtime,
            'PercentSalaryHike':        percent_hike,
            'PerformanceRating':        3,
            'RelationshipSatisfaction': rel_sat,
            'StockOptionLevel':         stock_option,
            'TotalWorkingYears':        total_years,
            'TrainingTimesLastYear':    training,
            'WorkLifeBalance':          work_balance,
            'YearsAtCompany':           yrs_company,
            'YearsInCurrentRole':       yrs_company,
            'YearsSinceLastPromotion':  yrs_promo,
            'YearsWithCurrManager':     yrs_company,
        }
        input_df = pd.DataFrame([record])

        # Apply feature engineering using TRAIN role_rates (no leakage)
        input_df = engineer_features_inference(input_df, role_rates, global_mean)

        prob  = model.predict_proba(input_df)[0, 1]
        label = "🔴 AT RISK" if prob >= threshold else "🟢 LIKELY TO STAY"

        col1, col2, col3 = st.columns(3)
        col1.metric("Attrition Probability", f"{prob:.0%}")
        col2.metric("Threshold", f"{threshold:.2f}")
        col3.metric("Prediction", label)

        st.progress(float(prob))

        # ── Feature Impact Analysis (renamed from approximate SHAP) ────────
        st.subheader("🔍 Feature Impact Analysis")
        st.caption("Shows how much each feature shifts the risk score "
                   "compared to a typical employee. "
                   "Not SHAP — uses single-feature permutation.")

        background  = df_all.sample(200, random_state=42)
        if 'Attrition' in background.columns:
            background = background.drop(columns=['Attrition'])

        baseline = {}
        for c in background.columns:
            if pd.api.types.is_numeric_dtype(background[c]):
                baseline[c] = background[c].median()
            else:
                baseline[c] = background[c].mode()[0]

        effects = {}
        for c in input_df.columns:
            if c not in baseline:
                continue
            x_ref = input_df.copy()
            x_ref.at[0, c] = baseline[c]
            try:
                effects[c] = prob - model.predict_proba(x_ref)[0, 1]
            except Exception:
                pass

        top5  = sorted(effects.items(), key=lambda kv: abs(kv[1]),
                        reverse=True)[:5]
        names, vals = zip(*top5)
        labels = [f"{f} = {input_df.loc[0, f]}" for f in names]
        colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in vals]

        fig, ax = plt.subplots(figsize=(7, 3))
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("Change in Risk Score (red = increases risk)")
        ax.set_title("Top 5 Feature Impact")
        fig.tight_layout()
        st.pyplot(fig)


# ── Tab 2: Bulk Upload ───────────────────────────────────────────────────────
with tab2:
    st.header("📂 Bulk Employee Scoring")
    st.info("Upload a CSV in the same format as the IBM HR dataset. "
            "The Attrition column is optional.")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        try:
            df_bulk = pd.read_csv(uploaded)

            # Apply leak-free feature engineering
            df_bulk_eng = engineer_features_inference(
                df_bulk, role_rates, global_mean
            )

            # Drop Attrition if present (inference only)
            score_cols = [c for c in df_bulk_eng.columns if c != 'Attrition']
            df_bulk_eng = df_bulk_eng[score_cols]

            df_bulk['RiskScore'] = model.predict_proba(df_bulk_eng)[:, 1]
            df_bulk['Flag'] = np.where(
                df_bulk['RiskScore'] >= threshold, "At Risk", "Stay"
            )

            # Summary
            at_risk = (df_bulk['Flag'] == 'At Risk').sum()
            st.metric("At Risk Employees",
                       f"{at_risk} / {len(df_bulk)} "
                       f"({at_risk/len(df_bulk):.0%})")

            # Use index as employee identifier if no ID column present
            id_col = 'EmployeeNumber' if 'EmployeeNumber' in df_bulk.columns \
                else df_bulk.index.name or 'Index'
            display_cols = [id_col, 'RiskScore', 'Flag'] \
                if id_col in df_bulk.columns \
                else ['RiskScore', 'Flag']

            st.dataframe(
                df_bulk[display_cols].sort_values('RiskScore', ascending=False),
                height=400
            )
            st.download_button(
                "⬇️ Download Scored CSV",
                df_bulk.to_csv(index=False).encode(),
                file_name="scored_attrition.csv"
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Make sure your CSV matches the IBM HR dataset column names.")


# ── Tab 3: Analytics ─────────────────────────────────────────────────────────
with tab3:
    st.header("📈 Cohort & Calibration Analytics")

    df_cohort = df_all.copy()
    if dept_filter:
        df_cohort = df_cohort[df_cohort['Department'].isin(dept_filter)]

    score_cols = [c for c in df_cohort.columns if c != 'Attrition']
    df_cohort['RiskScore'] = model.predict_proba(df_cohort[score_cols])[:, 1]
    df_cohort['Flag'] = np.where(
        df_cohort['RiskScore'] >= threshold, "At Risk", "Stay"
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", len(df_cohort))
    col2.metric("At Risk", (df_cohort['Flag'] == 'At Risk').sum())
    col3.metric("Avg Risk Score", f"{df_cohort['RiskScore'].mean():.1%}")

    st.divider()

    # Department avg risk
    dept_avg = (
        df_cohort.groupby('Department')['RiskScore']
        .mean().reset_index().rename(columns={'RiskScore': 'AvgRisk'})
    )
    bar = alt.Chart(dept_avg).mark_bar(color='#3498DB').encode(
        x=alt.X('Department:N', sort='-y'),
        y=alt.Y('AvgRisk:Q', axis=alt.Axis(format='%'), title='Avg Risk'),
        tooltip=['Department', alt.Tooltip('AvgRisk:Q', format='.1%')]
    ).properties(width=650, height=280, title="Average Attrition Risk by Department")
    st.altair_chart(bar)

    # Risk score distribution
    hist = alt.Chart(df_cohort).mark_bar(opacity=0.7).encode(
        alt.X('RiskScore:Q', bin=alt.Bin(maxbins=30), title='Risk Score'),
        alt.Y('count():Q', title='Employees'),
        color=alt.Color('Department:N'),
        tooltip=['Department', 'count()']
    ).properties(width=650, height=200, title="Risk Score Distribution")
    st.altair_chart(hist)

    # Precision & Recall vs threshold
    if 'Attrition' in df_cohort.columns:
        pr_data = []
        for t in np.linspace(0, 1, 101):
            preds = (df_cohort['RiskScore'] >= t).astype(int)
            tp = ((preds == 1) & (df_cohort['Attrition'] == 1)).sum()
            fp = ((preds == 1) & (df_cohort['Attrition'] == 0)).sum()
            fn = ((preds == 0) & (df_cohort['Attrition'] == 1)).sum()
            recall    = tp / (tp + fn) if tp + fn > 0 else 0
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            pr_data.append({'threshold': t, 'precision': precision,
                             'recall': recall})
        pr_df = pd.DataFrame(pr_data)
        pr_chart = alt.Chart(pr_df).transform_fold(
            ['precision', 'recall'], as_=['metric', 'value']
        ).mark_line().encode(
            x=alt.X('threshold:Q', title='Threshold'),
            y=alt.Y('value:Q', title='Score'),
            color='metric:N',
            tooltip=['threshold:Q', 'value:Q', 'metric:N']
        ).properties(width=650, height=250,
                      title="Precision & Recall vs Threshold")
        st.altair_chart(pr_chart)

        # Calibration curve
        st.subheader("Calibration Curve")
        st.caption("A well-calibrated model should follow the diagonal.")
        prob_true, prob_pred = calibration_curve(
            df_cohort['Attrition'], df_cohort['RiskScore'], n_bins=10
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(prob_pred, prob_true, marker='o', label='Model')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Actual Attrition Rate")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    # Cohort summary table
    st.subheader("Cohort Summary")
    summary = (df_cohort.groupby(['Department', 'Flag'])
               .size().reset_index(name='Count'))
    st.dataframe(summary)
