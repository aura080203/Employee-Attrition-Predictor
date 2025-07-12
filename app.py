#!/usr/bin/env python3
"""
Streamlit App: 🚀 Employee Attrition Predictor Dashboard
--------------------------------------------------------
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import altair as alt
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# ── MUST come before any other st.* call ───────────────────────────────────
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# ── Constants ───────────────────────────────────────────────────────────────
MODEL_PATH  = Path('artifacts/attrition_model.pkl')
THRESH_PATH = Path('artifacts/thresholds.json')
DATA_URL    = (
    'https://raw.githubusercontent.com/'
    'mragpavank/ibm-hr-analytics-attrition-dataset/'
    'master/WA_Fn-UseC_-HR-Employee-Attrition.csv'
)

# ── Feature Engineering ─────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Attrition'] = (df['Attrition']=='Yes').astype(int)
    df['RoleAttritionRate'] = df.groupby('JobRole')['Attrition'].transform('mean')
    df['OT_x_JobSat']       = ((df['OverTime']=='Yes').astype(int) * df['JobSatisfaction'])
    df['TenureRatio']       = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['AvgSatisfaction']   = df[['EnvironmentSatisfaction','JobSatisfaction','RelationshipSatisfaction']].mean(axis=1)
    df['YearsPerCompany']   = df['YearsAtCompany'] / (df['NumCompaniesWorked'] + 1)
    return df

# ── Load resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    model      = joblib.load(MODEL_PATH)                  # CalibratedClassifierCV
    thresholds = pd.read_json(THRESH_PATH, typ='series')
    df_raw     = pd.read_csv(DATA_URL)
    df_all     = engineer_features(df_raw)
    role_rates = df_all.groupby('JobRole')['Attrition'].mean().to_dict()
    return model, thresholds, df_all, role_rates

model, thresholds, df_all, role_attr_rates = load_resources()
th_cost    = thresholds['threshold_cost']

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Global Controls")
threshold   = st.sidebar.slider("Risk Threshold", 0.0, 1.0, float(th_cost), 0.01)
dept_filter = st.sidebar.multiselect("Filter by Department", sorted(df_all['Department'].unique()))

# ── Page layout ─────────────────────────────────────────────────────────────
st.title("🚀 Employee Attrition Predictor Dashboard")
tab1, tab2, tab3 = st.tabs(["Individual", "Bulk Upload", "Analytics"])

# ── Tab 1: Single Employee + Approx SHAP ────────────────────────────────────
with tab1:
    st.header("🔍 Single Employee Analysis")
    with st.form("attrition_form"):
        # categorical inputs
        business  = st.selectbox("Business Travel", ['Non-Travel','Travel_Rarely','Travel_Frequently'])
        dept       = st.selectbox("Department", ['Sales','Research & Development','Human Resources'])
        edu_field  = st.selectbox("Education Field", ['Life Sciences','Medical','Marketing','Technical Degree','Human Resources','Other'])
        gender     = st.selectbox("Gender", ['Male','Female'])
        job_role   = st.selectbox("Job Role", sorted(role_attr_rates.keys()))
        marital    = st.selectbox("Marital Status", ['Single','Married','Divorced'])
        overtime   = st.selectbox("OverTime", ['Yes','No'])
        # numeric inputs
        age            = st.slider("Age",18,60,30)
        dist           = st.slider("Distance From Home (km)",1,50,5)
        yrs_company    = st.slider("Years at Company",0,40,3)
        yrs_promo      = st.slider("Years Since Last Promotion",0,15,1)
        job_sat        = st.slider("Job Satisfaction (1-4)",1,4,4)
        env_sat        = st.slider("Environment Satisfaction (1-4)",1,4,3)
        rel_sat        = st.slider("Relationship Satisfaction (1-4)",1,4,3)
        work_balance   = st.slider("Work Life Balance (1-4)",1,4,3)
        education_lvl  = st.slider("Education Level (1-5)",1,5,3)
        num_companies  = st.slider("Num. Companies Worked",0,10,1)
        total_years    = st.number_input("Total Working Years",0,40,8)
        training       = st.slider("Training Times Last Year",0,10,2)
        stock_option   = st.slider("Stock Option Level",0,3,1)
        percent_hike   = st.slider("Percent Salary Hike",0,25,18)
        monthly_income = st.number_input("Monthly Income",1000,20000,7000)
        daily_rate     = st.number_input("Daily Rate",100,1500,800)
        hourly_rate    = st.number_input("Hourly Rate",20,100,50)
        submit         = st.form_submit_button("Predict & Explain")

    if submit:
        # build the one-row DataFrame
        ot_num = 1 if overtime=='Yes' else 0
        record = {
            'Age':                     age,
            'BusinessTravel':          business,
            'DailyRate':               daily_rate,
            'Department':              dept,
            'DistanceFromHome':        dist,
            'Education':               education_lvl,
            'EducationField':          edu_field,
            'EnvironmentSatisfaction': env_sat,
            'Gender':                  gender,
            'HourlyRate':              hourly_rate,
            'JobInvolvement':          job_sat,
            'JobLevel':                1,
            'JobRole':                 job_role,
            'JobSatisfaction':         job_sat,
            'MaritalStatus':           marital,
            'MonthlyIncome':           monthly_income,
            'MonthlyRate':             daily_rate * 24,
            'NumCompaniesWorked':      num_companies,
            'OverTime':                overtime,
            'PercentSalaryHike':       percent_hike,
            'PerformanceRating':       3,
            'RelationshipSatisfaction':rel_sat,
            'StockOptionLevel':        stock_option,
            'TotalWorkingYears':       total_years,
            'TrainingTimesLastYear':   training,
            'WorkLifeBalance':         work_balance,
            'YearsAtCompany':          yrs_company,
            'YearsInCurrentRole':      yrs_company,
            'YearsSinceLastPromotion': yrs_promo,
            'YearsWithCurrManager':    yrs_company,
            'OT_x_JobSat':             ot_num * job_sat,
            'TenureRatio':             yrs_promo / (yrs_company + 1),
            'RoleAttritionRate':       role_attr_rates.get(job_role, 0),
            'AvgSatisfaction':         np.mean([env_sat, job_sat, rel_sat]),
            'YearsPerCompany':         yrs_company / (num_companies + 1)
        }
        input_df = pd.DataFrame([record])

        # predict
        prob = model.predict_proba(input_df)[0,1]
        st.metric("Attrition Probability", f"{prob:.0%}")
        st.write(f"Threshold: **{threshold:.2f}** → **{'AT RISK' if prob>=threshold else 'Stay'}**")

        # ── Approximate SHAP: replace each feature with its background median/mode ─
        background = df_all.sample(100, random_state=42).drop(columns=['Attrition'])
        # precompute baselines
        baseline = {}
        for c in background.columns:
            if pd.api.types.is_numeric_dtype(background[c]):
                baseline[c] = background[c].median()
            else:
                baseline[c] = background[c].mode()[0]

        # measure one-by-one effect
        effects = {}
        for c in input_df.columns:
            x_ref = input_df.copy()
            x_ref.at[0, c] = baseline[c]
            effects[c] = prob - model.predict_proba(x_ref)[0,1]

        # pick top 5 by absolute effect
        top5 = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        names, vals = zip(*top5)
        labels = [f"{input_df.loc[0,f]} = {f}" for f in names]

        # plot
        fig, ax = plt.subplots(figsize=(6,4))
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Change in Attrition Probability")
        ax.set_title("Top 5 Risk Drivers (Approximate SHAP)")
        st.pyplot(fig)

# ── Tab 2: Bulk Upload & Batch Scoring ──────────────────────────────────────
with tab2:
    st.header("📂 Bulk Employee Scoring")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_bulk = pd.read_csv(uploaded)
        df_bulk = engineer_features(df_bulk)
        df_bulk['RiskScore'] = model.predict_proba(df_bulk.drop(columns=['Attrition']))[:,1]
        df_bulk['Flag']      = np.where(df_bulk['RiskScore']>=threshold, "At Risk", "Stay")
        st.dataframe(df_bulk[['EmployeeID','RiskScore','Flag']], height=400)
        st.download_button("Download Scored CSV",
                           df_bulk.to_csv(index=False).encode(),
                           file_name="scored_attrition.csv")

# ── Tab 3: Cohort Analytics & Calibration ──────────────────────────────────
with tab3:
    st.header("📈 Cohort & Calibration Analytics")
    df_cohort = df_all.copy()
    if dept_filter:
        df_cohort = df_cohort[df_cohort['Department'].isin(dept_filter)]
    df_cohort['RiskScore'] = model.predict_proba(df_cohort.drop(columns=['Attrition']))[:,1]
    df_cohort['Flag']      = np.where(df_cohort['RiskScore']>=threshold, "At Risk", "Stay")

    # Dept avg risk
    dept_avg = (
        df_cohort.groupby('Department')['RiskScore']
        .mean().reset_index().rename(columns={'RiskScore':'AvgRisk'})
    )
    bar = alt.Chart(dept_avg).mark_bar().encode(
        x=alt.X('Department:N', sort='-y'),
        y=alt.Y('AvgRisk:Q', axis=alt.Axis(format='%')),
        tooltip=['Department','AvgRisk']
    ).properties(width=650, height=300, title="Average Risk by Department")
    st.altair_chart(bar)

    # Distribution histogram
    hist = alt.Chart(df_cohort).mark_bar(opacity=0.6).encode(
        alt.X('RiskScore:Q', bin=alt.Bin(maxbins=30), title='Risk Score'),
        alt.Y('count():Q', title='Count'),
        color='Department:N'
    ).properties(width=650, height=200, title="Risk Score Distribution")
    st.altair_chart(hist)

    # Precision & Recall vs threshold
    pr_data = []
    for t in np.linspace(0,1,101):
        preds = (df_cohort['RiskScore']>=t).astype(int)
        tp = ((preds==1)&(df_cohort['Attrition']==1)).sum()
        fp = ((preds==1)&(df_cohort['Attrition']==0)).sum()
        fn = ((preds==0)&(df_cohort['Attrition']==1)).sum()
        recall    = tp/(tp+fn) if tp+fn>0 else 0
        precision = tp/(tp+fp) if tp+fp>0 else 0
        pr_data.append({'threshold':t,'precision':precision,'recall':recall})
    pr_df = pd.DataFrame(pr_data)
    pr_chart = alt.Chart(pr_df).transform_fold(
        ['precision','recall'], as_=['metric','value']
    ).mark_line().encode(
        x='threshold:Q', y='value:Q', color='metric:N'
    ).properties(width=650, height=250, title="Precision & Recall vs Threshold")
    st.altair_chart(pr_chart)

    # Calibration curve
    st.subheader("Calibration Curve")
    prob_true, prob_pred = calibration_curve(df_cohort['Attrition'], df_cohort['RiskScore'], n_bins=10)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(prob_pred, prob_true, marker='o')
    ax.plot([0,1],[0,1],'--', color='gray')
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Actual Attrition Rate")
    st.pyplot(fig)

    # Cohort summary
    st.markdown("**Cohort Summary**")
    summary = df_cohort['Flag'].value_counts().rename_axis('Flag').reset_index(name='Count')
    st.table(summary)