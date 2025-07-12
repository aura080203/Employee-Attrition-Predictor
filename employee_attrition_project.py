#!/usr/bin/env python3
import logging
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import joblib


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    df.drop(columns=[
        'EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'
    ], inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    df['RoleAttritionRate'] = df.groupby('JobRole')['Attrition'].transform('mean')
    df['OT_x_JobSat'] = ((df['OverTime'] == 'Yes').astype(int) * df['JobSatisfaction'])
    df['TenureRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    # Composite satisfaction
    df['AvgSatisfaction'] = df[['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
    # Years per company
    df['YearsPerCompany'] = df['YearsAtCompany'] / (df['NumCompaniesWorked'] + 1)
    # Drop raw columns not used
    return df


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def build_pipeline(numeric_features, categorical_features) -> ImbPipeline:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=42, scale_pos_weight=5)
    lgb = LGBMClassifier(random_state=42, is_unbalance=True)
    rf  = RandomForestClassifier(random_state=42, class_weight='balanced')
    lr_base = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    stack = StackingClassifier(
        estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf), ('lr', lr_base)],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5, n_jobs=-1
    )
    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),
        ('tomek', TomekLinks()),
        ('stack', stack)
    ])


def optimize_hyperparameters(pipeline, X_train, y_train) -> GridSearchCV:
    param_grid = {
        'stack__xgb__n_estimators': [100, 200],
        'stack__xgb__max_depth':    [3, 5],
        'stack__lgb__n_estimators': [100, 200],
        'stack__lgb__num_leaves':   [31, 50],
        'stack__rf__n_estimators':  [100, 200]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        error_score='raise'
    )
    grid.fit(X_train, y_train)
    logging.info(f"Best params: {grid.best_params_}")
    return grid


def calibrate_model(model, X_train, y_train) -> CalibratedClassifierCV:
    calib = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calib.fit(X_train, y_train)
    return calib


def compute_cost_threshold(y_true, y_proba, cost_fp=1, cost_fn=5):
    thresholds = np.linspace(0, 1, 101)
    best_thr, best_cost = 0, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_thr = cost, t
    return best_thr, best_cost


def evaluate_model(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred  = model.predict(X_test)
    logging.info(f"Accuracy @0.5 cutoff: {accuracy_score(y_test, y_pred):.4f}")
    logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    logging.info(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    logging.info(f"PR AUC: {auc(recall, precision):.4f}")
    # F1-optimal threshold
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    idx_f1 = np.nanargmax(f1[:-1])
    thr_f1 = thresholds[idx_f1]
    logging.info(f"Optimal F1 thr: {thr_f1:.3f}")
    # Recall ≥51%
    idx_r = np.where(recall >= 0.51)[0]
    thr_r51 = thresholds[idx_r[0]] if len(idx_r) else thr_f1
    logging.info(f"Rec≥51% thr: {thr_r51:.3f}")
    # Cost-based threshold
    thr_cost, cost_val = compute_cost_threshold(y_test, y_proba)
    logging.info(f"Cost-based thr: {thr_cost:.2f} (cost={cost_val})")
    # Max accuracy threshold
    grid_thr = np.linspace(0,1,101)
    accs = [(t, accuracy_score(y_test, (y_proba >= t).astype(int))) for t in grid_thr]
    thr_acc, _ = max(accs, key=lambda x: x[1])
    logging.info(f"Max-acc thr: {thr_acc:.2f}")
    return {
        'threshold_f1':       float(thr_f1),
        'threshold_51recall': float(thr_r51),
        'threshold_maxacc':   float(thr_acc),
        'threshold_cost':     float(thr_cost)
    }


def save_artifacts(model, thresholds, out_dir='artifacts'):
    path = Path(out_dir)
    path.mkdir(exist_ok=True)
    joblib.dump(model, path/'attrition_model.pkl')
    with open(path/'thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    logging.info(f"Artifacts saved to {path.resolve()}")


if __name__ == '__main__':
    setup_logging()
    DATA_URL = (
        'https://raw.githubusercontent.com/'
        'mragpavank/ibm-hr-analytics-attrition-dataset/'
        'master/WA_Fn-UseC_-HR-Employee-Attrition.csv'
    )
    df_raw = load_data(DATA_URL)
    df     = engineer_features(df_raw)
    X_tr, X_te, y_tr, y_te = split_data(df)
    num_feats = X_tr.select_dtypes(include='number').columns.tolist()
    cat_feats = X_tr.select_dtypes(include='object').columns.tolist()
    pipeline  = build_pipeline(num_feats, cat_feats)
    gridcv    = optimize_hyperparameters(pipeline, X_tr, y_tr)
    best      = gridcv.best_estimator_
    calib     = calibrate_model(best, X_tr, y_tr)
    thresholds= evaluate_model(calib, X_te, y_te)
    save_artifacts(calib, thresholds)
